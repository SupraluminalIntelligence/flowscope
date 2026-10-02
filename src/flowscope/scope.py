"""FlowScope: live view of the forward and backward pass through a residual-stream transformer.

Attach once, then train as usual; it hooks the model and optimizer and redraws every `every` steps.

    scope = FlowScope(model, optimizer, every=50, label="baseline")
    ... your normal training loop ...
    scope.close()   # optional: remove hooks

In a notebook it draws inline; in a plain script it serves a live page at http://127.0.0.1:8765
(or run an unmodified script with `flowscope run train.py`).

Top lane: the residual stream going forward (color = RMS of the activations).
Bottom lane: dloss/dx flowing back (color = size relative to the gradient at the last block).
A dark red bottom lane toward the left is a blockage: early layers get almost no learning signal.
"""
import base64, json, math, os, struct, zlib
import torch
import torch.nn as nn

from flowscope.tour import Tour, embed, rows, sample

# ---------------------------------------------------------------- colors / images

def _lerp(stops, t):
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    a, b = stops[i], stops[i + 1]
    return tuple(int(a[k] + (b[k] - a[k]) * f) for k in range(3))

_DIV = [(56, 132, 255), (20, 24, 36), (255, 128, 32)]                       # -1 .. 0 .. +1
_SEQ = [(12, 14, 24), (88, 40, 140), (220, 80, 90), (252, 210, 60)]         # attention 0 .. 1
_HEALTH = [(90, 12, 20), (170, 40, 40), (120, 120, 130), (40, 210, 235), (240, 80, 200)]

def _health(log_ratio):
    """log10(value/reference): -8 blocked (dark red), 0 healthy (cyan), +3 exploding (magenta)."""
    t = (log_ratio + 8) / 8 * 0.75 if log_ratio <= 0 else 0.75 + min(log_ratio, 3) / 3 * 0.25
    return "rgb(%d,%d,%d)" % _lerp(_HEALTH, t)

def _fwd_color(rms):
    """Forward activations: shrinking toward 0 is the problem, so penalize small RMS harder than large."""
    l = _log(rms)
    return _health(3 * l if l < 0 else 0.7 * l)

def _png(pixels, w, h):
    raw = b"".join(b"\x00" + bytes(c for px in pixels[r * w:(r + 1) * w] for c in px) for r in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()

def _strip(vec):
    """A vertical strip: one cell per channel, diverging colors, normalized to its own max."""
    m = max((abs(v) for v in vec), default=0) or 1.0
    return _png([_lerp(_DIV, 0.5 + 0.5 * v / m) for v in vec], 1, len(vec))

def _heat(mat):
    h, w = len(mat), len(mat[0])
    return _png([_lerp(_SEQ, v) for row in mat for v in row], w, h)

def _sci(x):
    if x is None or x != x:
        return "-"
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    return f"{x:.2f}" if -2 <= e <= 2 else f"{x / 10 ** e:.1f}e{e}"

def _log(x, floor=1e-30):
    return math.log10(max(x, floor))

# ---------------------------------------------------------------- model discovery

def _child(mod, *names):
    for n in names:
        if hasattr(mod, n) and isinstance(getattr(mod, n), nn.Module):
            return getattr(mod, n)
    return None

def _in_notebook():
    try:
        import sys
        from IPython import get_ipython
        # Jupyter / Colab kernels; plain python and the IPython terminal have no kernel
        return "google.colab" in sys.modules or hasattr(get_ipython(), "kernel")
    except Exception:
        return False

def _find_blocks(model):
    for path in ("blocks", "transformer.h", "layers", "h"):
        m = model
        for part in path.split("."):
            m = getattr(m, part, None)
        if isinstance(m, (nn.Sequential, nn.ModuleList)) and len(m):
            return list(m)
    raise ValueError("FlowScope: couldn't find the transformer blocks (looked for model.blocks / model.transformer.h)")

# ---------------------------------------------------------------- the scope

class FlowScope:
    runs = {}  # label -> latest gradient profile, shared across scopes so runs can be compared

    def __init__(self, model, optimizer, every=50, label=None, token=-1, sink="auto", port=8765,
                 runs_file=".flowscope/runs.json", tour_points=256, record=None, host="127.0.0.1"):
        old = getattr(model, "_flowscope", None)
        if old is not None:
            old.close()
        model._flowscope = self
        self.model, self.opt, self.every, self.token = model, optimizer, every, token
        self.label = label or f"run {len(FlowScope.runs) + 1}"
        self.blocks = _find_blocks(model)
        self.L = len(self.blocks)
        self.steps, self.active, self.cur, self.hist, self._hooks, self._snap = 0, False, None, [], [], {}
        self.tour_points, self._tour, self.tour_payload = tour_points, Tour() if tour_points else None, None
        self.record = record  # JSONL file: one tour frame per redraw, for replay (e.g. in GrandTourVision)
        self._install()
        self._handle = self._server = None
        waiting = self._frame('<div style="padding:24px;color:#8b93a7">FlowScope attached. Waiting for the first training step...</div>')
        if sink == "auto":
            sink = "notebook" if _in_notebook() else "server"
        if sink == "notebook":
            from IPython.display import HTML, display
            self._HTML = HTML
            self._handle = display(HTML(waiting), display_id=True)
        elif sink == "server" or hasattr(sink, "publish"):
            if sink == "server":
                from flowscope.server import LiveServer
                sink = LiveServer.shared(port, host)
            self._server = sink
            self._server.publish(waiting)
            self.runs_file = runs_file
            self._load_runs()

    # ---- hooks -------------------------------------------------------------------

    def _install(self):
        H = self._hooks
        H.append(self.model.register_forward_pre_hook(self._on_model_pre))
        H.append(self.model.register_forward_hook(self._on_model_out))
        for i, b in enumerate(self.blocks):
            if i == 0:
                H.append(b.register_forward_pre_hook(lambda m, a: self._cap(("s", 0), a[0])))
            H.append(b.register_forward_hook(lambda m, a, o, i=i: self._cap(("s", 2 * i + 2), o)))
            ln2 = _child(b, "ln2", "ln_2", "post_attention_layernorm")
            if ln2 is not None:
                H.append(ln2.register_forward_pre_hook(lambda m, a, i=i: self._cap(("s", 2 * i + 1), a[0])))
            sa, ff = _child(b, "sa", "attn", "self_attn", "attention"), _child(b, "ffwd", "mlp", "feed_forward")
            if sa is not None:
                H.append(sa.register_forward_hook(lambda m, a, o, i=i: self._cap(("a", i), o, grad=False)))
                heads = [h for h in sa.modules() if hasattr(h, "tril") and isinstance(getattr(h, "dropout", None), nn.Module)]
                for k, h in enumerate(heads):  # attention probs are the input to each head's dropout
                    H.append(h.dropout.register_forward_pre_hook(lambda m, a, i=i, k=k: self._cap_attn(i, k, a[0])))
            if ff is not None:
                H.append(ff.register_forward_hook(lambda m, a, o, i=i: self._cap(("f", i), o, grad=False)))
                for act in ff.modules():
                    if isinstance(act, (nn.ReLU, nn.GELU, nn.SiLU)):
                        H.append(act.register_forward_hook(lambda m, a, o, i=i: self._cap_relu(i, o)))
                        break
        head = _child(self.model, "lm_head", "head")
        if head is not None:
            H.append(head.register_forward_hook(lambda m, a, o: self._cap(("logits", 0), o)))
        H.append(self.opt.register_step_pre_hook(self._on_opt_pre))
        H.append(self.opt.register_step_post_hook(self._on_opt_post))
        # parameter groups for grad size and update/data ratio
        self._groups = [("emb", [p for n, p in self.model.named_parameters() if "embed" in n or n.startswith(("wte", "wpe"))])]
        for i, b in enumerate(self.blocks):
            sa, ff = _child(b, "sa", "attn", "self_attn", "attention"), _child(b, "ffwd", "mlp", "feed_forward")
            self._groups.append((("a", i), list(sa.parameters()) if sa is not None else []))
            self._groups.append((("f", i), list(ff.parameters()) if ff is not None else []))
        if head is not None:
            self._groups.append(("head", list(head.parameters())))

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        if getattr(self.model, "_flowscope", None) is self:
            del self.model._flowscope

    def _training(self):
        return self.model.training and torch.is_grad_enabled()

    def _on_model_pre(self, m, args):
        self.active = self._training() and self.steps % self.every == 0
        if self.active:
            self.cur = dict(fwd={}, grad={}, vec={}, gvec={}, attn={}, relu={}, cloud={}, tokens=None, loss=None)
            idx = args[0] if args else None
            if self._tour and torch.is_tensor(idx) and idx.dim() == 2 and not idx.is_floating_point():
                self.cur["tokens"] = idx[:rows(*idx.shape, self.tour_points)].reshape(-1).tolist()

    def _on_model_out(self, m, args, out):
        if self.active and isinstance(out, tuple) and len(out) == 2 and torch.is_tensor(out[1]):
            self.cur["loss"] = out[1].item()
        elif self.active:
            self.active = False  # e.g. generate(): no targets, no backward coming

    def _pick(self, t):
        return t.detach()[0, self.token].float().cpu().tolist() if t.dim() == 3 else []

    def _cap(self, key, t, grad=True):
        if not (self.active and torch.is_tensor(t)):
            return
        c = self.cur
        c["fwd"][key] = t.detach().float().pow(2).mean().sqrt().item()
        if self._tour and key[0] == "s" and key[1] % 2 == 0 and t.dim() == 3:  # block boundaries only
            c["cloud"][key[1] // 2] = sample(t, self.tour_points)
            c["T"] = t.shape[1]
        c["vec"][key] = self._pick(t)
        if grad and t.requires_grad:
            def on_grad(g, key=key):
                c["grad"][key] = g.detach().float().pow(2).mean().sqrt().item()
                c["gvec"][key] = self._pick(g)
            t.register_hook(on_grad)

    def _cap_attn(self, i, k, wei):
        if not self.active or wei.dim() != 3:
            return
        w = wei.detach().float()
        maxp = w.max(-1).values.mean().item()
        ent = -(w * torch.log(w.clamp_min(1e-12))).sum(-1).mean().item()
        d = self.cur["attn"].setdefault(i, dict(maxp=[], ent=[], map=None))
        d["maxp"].append(maxp)
        d["ent"].append(ent)
        if k == 0:
            d["map"] = w[0].cpu().tolist()

    def _cap_relu(self, i, h):
        if not self.active:
            return
        h = h.detach().reshape(-1, h.shape[-1])
        self.cur["relu"][i] = dict(zero=(h <= 0).float().mean().item(),
                                   dead=(~(h > 0).any(0)).float().mean().item())

    def _on_opt_pre(self, opt, args, kwargs):
        if not (self.active and self.cur and self.cur["loss"] is not None):
            return
        self._snap, pg = {}, {}
        for key, ps in self._groups:
            ps = [p for p in ps if p.grad is not None]
            self._snap[key] = [(p, p.detach().clone()) for p in ps if p.dim() >= 2]
            if ps:
                pg[key] = math.sqrt(sum(p.grad.float().pow(2).sum().item() for p in ps) / sum(p.numel() for p in ps))
        self.cur["pgrad"] = pg

    def _on_opt_post(self, opt, args, kwargs):
        if self.active and self.cur and self.cur["loss"] is not None:
            ur = {}
            for key, pairs in self._snap.items():
                vals = [_log(((p.detach() - old).std() / (old.std() + 1e-12)).item()) for p, old in pairs]
                if vals:
                    ur[key] = sum(vals) / len(vals)
            self.cur["update"], self.cur["step"] = ur, self.steps
            self._snap = {}
            self._commit()
        self.active = False
        self.steps += 1

    # ---- data --------------------------------------------------------------------

    def _points(self):
        """Stream positions in order: s0 (embeddings), then mid/out of each block."""
        return [("s", j) for j in range(2 * self.L + 1)]

    def _commit(self):
        c = self.cur
        pts = [p for p in self._points() if p in c["fwd"]]
        ref = c["grad"].get(pts[-1]) if pts else None
        c["profile"] = [(p[1], _log(c["grad"][p] / ref)) for p in pts if ref and p in c["grad"]]
        self.hist.append(dict(step=c["step"], loss=c["loss"], profile=c["profile"]))
        FlowScope.runs[self.label] = dict(profile=c["profile"], L=self.L, step=c["step"], loss=c["loss"])
        self._save_runs()
        if self._tour and len(c["cloud"]) == self.L + 1:
            clouds = [c["cloud"][j] for j in range(self.L + 1)]
            tokens = c["tokens"] if c["tokens"] and len(c["tokens"]) == clouds[0].shape[0] else None
            names = ["emb"] + [f"b{i + 1}" for i in range(self.L)]
            self.tour_payload = self._tour.payload(clouds, names, tokens, c["T"], c["step"])
            if self.record:
                os.makedirs(os.path.dirname(os.path.abspath(self.record)), exist_ok=True)
                with open(self.record, "a") as f:
                    f.write(json.dumps(dict(self.tour_payload, label=self.label)) + "\n")
        self.render()

    # runs are persisted in script mode so separate `python train.py` invocations can be compared
    def _load_runs(self):
        try:
            with open(self.runs_file) as f:
                for k, v in json.load(f).items():
                    FlowScope.runs.setdefault(k, v)
        except (OSError, ValueError):
            pass
        FlowScope.runs.pop(self.label, None)  # re-running the same config replaces its old line

    def _save_runs(self):
        if not getattr(self, "runs_file", None):
            return
        keep = dict(list(FlowScope.runs.items())[-6:])
        os.makedirs(os.path.dirname(self.runs_file) or ".", exist_ok=True)
        with open(self.runs_file, "w") as f:
            json.dump(keep, f)

    # ---- rendering ---------------------------------------------------------------

    def render(self):
        html = self.to_html()
        if self._handle is not None:
            self._handle.update(self._HTML(html + (embed(self.tour_payload) if self.tour_payload else "")))
        if self._server is not None:
            self._server.publish(html, self.tour_payload)
        return html

    def save(self, path="flowscope.html"):
        with open(path, "w") as f:
            f.write("<!doctype html><meta charset=utf-8><title>FlowScope</title>"
                    "<body style='margin:0;background:#0b0e16;padding:8px'>" + self.to_html()
                    + (embed(self.tour_payload) if self.tour_payload else "") + "</body>")
        return path

    @staticmethod
    def _frame(inner):
        return ('<div style="background:#0b0e16;color:#d5dae6;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;'
                'border-radius:10px;padding:10px 12px;max-width:100%;overflow-x:auto">' + inner + "</div>")

    def to_html(self):
        c = self.cur or {}
        if not c.get("profile") and not self.hist:
            return self._frame("waiting for data")
        return self._frame(self._header(c) + self._diagram(c) + self._charts())

    def _header(self, c):
        prof = dict(c.get("profile", []))
        r = prof.get(0)
        if r is None:
            verdict, col = "no gradient reached the embeddings", "#f87171"
        elif r < -3:
            verdict, col = f"BLOCKED: embeddings get 10^{r:.1f} of the output gradient", "#f87171"
        elif r < -1:
            verdict, col = f"weak: embeddings get 10^{r:.1f} of the output gradient", "#fbbf24"
        elif r > 2:
            verdict, col = f"EXPLODING: embeddings get 10^{r:+.1f} of the output gradient", "#e879f9"
        else:
            verdict, col = f"healthy: embeddings get {10 ** r:.2f}x of the output gradient", "#22d3ee"
        return (f'<div style="display:flex;gap:18px;flex-wrap:wrap;align-items:baseline;margin-bottom:4px">'
                f'<b style="font-size:14px;color:#fff">FlowScope · {self.label}</b>'
                f'<span>step {c.get("step", "-")}</span><span>loss {c.get("loss", 0):.3f}</span>'
                f'<span>{self.L} blocks</span><span style="color:{col}">{verdict}</span></div>')

    def _diagram(self, c):
        L = self.L
        dx = 128 if L <= 6 else max(64, 760 // L)
        X0, W = 150, 150 + L * dx + 190
        xs = {("s", j): X0 + j * dx / 2 for j in range(2 * L + 1)}
        xlog, xloss = xs[("s", 2 * L)] + 80, xs[("s", 2 * L)] + 150
        YA, YF, YB, YS = 46, 116, 206, 238
        out, anim = [], []
        T = lambda x, y, s, fill="#8b93a7", anchor="middle", size=10: out.append(
            f'<text x="{x:.0f}" y="{y:.0f}" fill="{fill}" font-size="{size}" text-anchor="{anchor}">{s}</text>')
        T(12, YF + 4, "forward x", "#d5dae6", "start", 11)
        T(12, YF + 18, "RMS", "#6b7280", "start")
        T(12, YB + 4, "backward ∇x", "#d5dae6", "start", 11)
        T(12, YB + 18, "vs output", "#6b7280", "start")

        pts = self._points()
        fwd, grad = c.get("fwd", {}), c.get("grad", {})
        gref = grad.get(pts[-1])
        glog = {p: (_log(grad[p] / gref) if gref and p in grad else None) for p in pts}
        chain = [p for p in pts if p in fwd]
        lane = [(xs[p], p) for p in chain] + [(xlog, ("logits", 0)), (xloss, "loss")]

        # branch boxes (attention above, MLP below), drawn first so the lanes sit on top
        for i in range(L):
            for kind, x_add, ybox, name in (("a", xs[("s", 2 * i + 1)], YA, "attn"), ("f", xs[("s", 2 * i + 2)], YF + 45, "mlp")):
                key = (kind, i)
                x_tap = x_add - dx / 2 + 10
                bw = min(56, dx / 2 - 14)
                bx = (x_tap + x_add) / 2
                contrib = fwd.get(key, 0) / (fwd.get(("s", 2 * i + (0 if kind == "a" else 1)), 0) or 1)
                ur = c.get("update", {}).get(key)
                # box outline = update/data ratio: cyan around 1e-3, red when the sublayer barely moves, magenta when too fast
                edge = "#374151" if ur is None else _health(0 if -4 <= ur <= -2 else (-5 if ur < -4 else 2))
                out.append(f'<path d="M{x_tap:.0f},{YF} L{x_tap:.0f},{ybox} L{x_add:.0f},{ybox} L{x_add:.0f},{YF}" '
                           f'fill="none" stroke="#2a3142" stroke-width="2"/>')
                out.append(f'<rect x="{bx - bw / 2:.0f}" y="{ybox - 13}" width="{bw:.0f}" height="26" rx="5" '
                           f'fill="#151a26" stroke="{edge}" stroke-width="1.5"/>')
                T(bx, ybox - 1, f"{name}{i + 1}" if dx >= 90 else name[0] + str(i + 1), "#cbd5e1")
                if dx >= 90:
                    T(bx, ybox + 9, f"×{contrib:.2f}", "#6b7280", size=8)
                T(x_add, YF - 8 if kind == "a" else YF + 14, "⊕", "#4b5563", size=11)

        # lanes
        for (xa, pa), (xb, pb) in zip(lane, lane[1:]):
            fa = fwd.get(pa) if pa != "loss" else None
            fcol = _fwd_color(fa) if fa else "#374151"
            gb = glog.get(pa) if isinstance(pa, tuple) and pa[0] == "s" else (
                _log(grad[pa] / gref) if gref and pa in grad else (0 if pa == ("s", 2 * L) else None))
            gcol = _health(gb) if gb is not None else "#374151"
            out.append(f'<line x1="{xa:.0f}" y1="{YF}" x2="{xb:.0f}" y2="{YF}" stroke="{fcol}" stroke-width="7" stroke-linecap="round" opacity=".6"/>')
            out.append(f'<line x1="{xa:.0f}" y1="{YB}" x2="{xb:.0f}" y2="{YB}" stroke="{gcol}" stroke-width="7" stroke-linecap="round" opacity=".6"/>')

        # nodes + numbers
        for x, p in lane:
            label = {"logits": "logits", "loss": "loss"}.get(p if p == "loss" else p[0])
            if label is None:
                j = p[1]
                label = "emb" if j == 0 else (f"b{(j + 1) // 2}" if j % 2 == 0 else "")
            out.append(f'<circle cx="{x:.0f}" cy="{YF}" r="4" fill="#0b0e16" stroke="#9ca3af"/>')
            out.append(f'<circle cx="{x:.0f}" cy="{YB}" r="4" fill="#0b0e16" stroke="#9ca3af"/>')
            if p == "loss":
                T(x, YF - 12, "loss", "#d5dae6")
                T(x, YF + 22, f"{c.get('loss', 0):.2f}", "#fff", size=11)
                T(x, YB + 22, "1.0", "#6b7280")
                continue
            if label:
                T(x, YF - 12 if p[0] != "s" or p[1] % 2 == 0 else YF - 12, label, "#d5dae6")
            if p[0] == "logits" or p[1] % 2 == 0 or dx >= 110:
                T(x, YF + 24 if p[0] == "s" and p[1] % 2 else YF + 24, _sci(fwd.get(p)), "#9ca3af", size=9)
                gtxt = _sci(grad.get(p) / gref) if gref and grad.get(p) is not None else "-"
                T(x, YB + 22, gtxt, "#9ca3af", size=9)

        # pulses: forward sweep left to right, then the gradient sweeps back right to left
        cyc, n = 7.0, len(lane) - 1
        tf, tb0, tb = 0.40, 0.48, 0.40
        for j in range(n):
            (xa, pa), (xb, pb) = lane[j], lane[j + 1]
            fa = fwd.get(pa) if pa != "loss" else None
            r = 3 + 3 * min(1, max(0, (_log(fa) + 1) / 2)) if fa else 3
            a, b = 0.005 + tf * j / n, 0.005 + tf * (j + 1) / n
            anim.append(self._pulse(xa, xb, YF, r, _fwd_color(fa) if fa else "#9ca3af", a, b, cyc))
            # backward: segment j is traversed from xb to xa, in reverse order
            k = n - 1 - j
            g = glog.get(pa) if isinstance(pa, tuple) and pa in glog else None
            if g is None and pa == ("logits", 0):
                g = 0
            rb = max(1.2, 6 + g * 0.6) if g is not None else 1.2
            a, b = tb0 + tb * k / n, tb0 + tb * (k + 1) / n
            anim.append(self._pulse(xb, xa, YB, rb, _health(g) if g is not None else "#7f1d1d", a, b, cyc))

        # vector strips: the actual numbers of one token's residual vector and its gradient
        vec, gvec = c.get("vec", {}), c.get("gvec", {})
        for p in chain:
            if p[1] % 2 == 0 or dx >= 110:
                x = xs[p]
                if vec.get(p):
                    out.append(f'<image href="{_strip(vec[p])}" x="{x - 11:.0f}" y="{YS}" width="9" height="96" '
                               f'preserveAspectRatio="none" style="image-rendering:pixelated"/>')
                if gvec.get(p):
                    out.append(f'<image href="{_strip(gvec[p])}" x="{x + 2:.0f}" y="{YS}" width="9" height="96" '
                               f'preserveAspectRatio="none" style="image-rendering:pixelated"/>')
        T(12, YS + 44, "one token:", "#6b7280", "start")
        T(12, YS + 58, "x | ∇x", "#6b7280", "start")

        # per-block internals: attention pattern (head 1) + saturation, MLP dead units, update ratio
        YI = YS + 118
        T(12, YI + 26, "attn head 1", "#6b7280", "start")
        T(12, YI + 40, "max-p · dead", "#6b7280", "start")
        for i in range(L):
            bx = (xs[("s", 2 * i)] + xs[("s", 2 * i + 2)]) / 2
            att = c.get("attn", {}).get(i)
            sz = min(64, dx - 16)
            if att and att["map"]:
                out.append(f'<image href="{_heat(att["map"])}" x="{bx - sz / 2:.0f}" y="{YI}" width="{sz:.0f}" height="{sz:.0f}" '
                           f'style="image-rendering:pixelated"/>')
                mp = sum(att["maxp"]) / len(att["maxp"])
                T(bx, YI + sz + 12, f"p̂ {mp:.2f}", "#fca5a5" if mp > 0.9 else "#9ca3af", size=9)
            rl = c.get("relu", {}).get(i)
            if rl:
                T(bx, YI + sz + 23, f"dead {rl['dead'] * 100:.0f}%", "#fca5a5" if rl["dead"] > 0.2 else "#9ca3af", size=9)
            ur = [c.get("update", {}).get((k, i)) for k in "af"]
            ur = [u for u in ur if u is not None]
            if ur:
                u = sum(ur) / len(ur)
                T(bx, YI + sz + 34, f"upd 10^{u:.1f}", "#fca5a5" if u > -2 or u < -4.5 else "#9ca3af", size=9)
        H = YI + 100
        legend = ("pulse size = signal size · lane color: <span style='color:#22d3ee'>healthy</span> "
                  "<span style='color:#9ca3af'>fading</span> <span style='color:#dc2626'>blocked</span> "
                  "<span style='color:#e879f9'>exploding</span> · p̂ = avg max attention prob (→1 = saturated softmax) · "
                  "dead = MLP units off for the whole batch · upd = update/data ratio (~10^-3 is healthy; box outline too)")
        return (f'<div style="color:#6b7280;font-size:10px;margin:2px 0 6px">{legend}</div>' f'<svg viewBox="0 0 {W:.0f} {H}" width="{W:.0f}" style="max-width:100%;height:auto;display:block" '
                f'xmlns="http://www.w3.org/2000/svg">' + "".join(out) + "".join(anim) + "</svg>")

    @staticmethod
    def _pulse(x0, x1, y, r, col, a, b, cyc):
        e = 0.004
        return (f'<circle cx="{x0:.0f}" cy="{y}" r="{r:.1f}" fill="#f8fafc" stroke="{col}" stroke-width="2" opacity="0" '
                f'style="filter:drop-shadow(0 0 5px {col})">'
                f'<animate attributeName="cx" values="{x0:.0f};{x0:.0f};{x1:.0f};{x1:.0f}" keyTimes="0;{a:.3f};{b:.3f};1" '
                f'dur="{cyc}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0;0;1;1;0;0" '
                f'keyTimes="0;{a:.3f};{a + e:.3f};{b - e:.3f};{b:.3f};1" dur="{cyc}s" repeatCount="indefinite"/></circle>')

    def _charts(self):
        W, H, pad = 460, 190, 34
        # left: gradient reaching each depth (log10, relative to the last block), this run over time + other runs
        lo = -1.0
        for h in self.hist[-12:]:
            lo = min([lo] + [v for _, v in h["profile"]])
        others = {k: v for k, v in FlowScope.runs.items() if k != self.label and v["profile"]}
        for v in others.values():
            lo = min([lo] + [y for _, y in v["profile"]])
        lo = math.floor(max(lo, -16))
        hi = 1.0

        def path(profile, L):
            n = 2 * L
            pts = [(pad + (W - pad - 10) * j / n, 10 + (H - 30) * (hi - max(y, lo)) / (hi - lo)) for j, y in profile]
            return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) if pts else ""

        g = [f'<text x="{pad}" y="-2" fill="#d5dae6" font-size="11">gradient reaching each depth (log10, vs output)</text>']
        for t in range(int(lo), int(hi) + 1, max(1, int((hi - lo) // 6))):
            y = 10 + (H - 30) * (hi - t) / (hi - lo)
            g.append(f'<line x1="{pad}" x2="{W - 10}" y1="{y:.0f}" y2="{y:.0f}" stroke="#1f2533"/>'
                     f'<text x="{pad - 4}" y="{y + 3:.0f}" fill="#6b7280" font-size="9" text-anchor="end">{t}</text>')
        g.append(f'<text x="{pad}" y="{H - 4}" fill="#6b7280" font-size="9">emb</text>'
                 f'<text x="{W - 10}" y="{H - 4}" fill="#6b7280" font-size="9" text-anchor="end">last block</text>')
        for k, (name, v) in enumerate(others.items()):
            g.append(f'<path d="{path(v["profile"], v["L"])}" fill="none" stroke="#f59e0b" stroke-width="1.5" '
                     f'stroke-dasharray="4 3" opacity=".8"/>')
        recent = self.hist[-12:]
        for k, h in enumerate(recent):
            last = k == len(recent) - 1
            g.append(f'<path d="{path(h["profile"], self.L)}" fill="none" stroke="#22d3ee" '
                     f'stroke-width="{2.2 if last else 1}" opacity="{1 if last else 0.12 + 0.5 * k / len(recent)}"/>')

        # right: loss
        ls = [(h["step"], h["loss"]) for h in self.hist]
        s0, s1 = ls[0][0], max(ls[-1][0], ls[0][0] + 1)
        l0, l1 = min(l for _, l in ls), max(l for _, l in ls)
        l1 = l1 if l1 > l0 else l0 + 1
        lp = "M" + " L".join(f"{pad + (W - pad - 10) * (s - s0) / (s1 - s0):.1f},{10 + (H - 30) * (l1 - l) / (l1 - l0):.1f}"
                             for s, l in ls)
        lchart = [f'<text x="{pad}" y="-2" fill="#d5dae6" font-size="11">training loss (sampled)</text>',
                  f'<text x="{pad - 4}" y="13" fill="#6b7280" font-size="9" text-anchor="end">{l1:.2f}</text>',
                  f'<text x="{pad - 4}" y="{H - 20}" fill="#6b7280" font-size="9" text-anchor="end">{l0:.2f}</text>',
                  f'<text x="{W - 10}" y="{H - 4}" fill="#6b7280" font-size="9" text-anchor="end">step {s1}</text>',
                  f'<path d="{lp}" fill="none" stroke="#a78bfa" stroke-width="1.6"/>']
        svg = lambda body: (f'<svg viewBox="0 -14 {W} {H + 14}" width="{W}" style="max-width:100%;height:auto" '
                            f'xmlns="http://www.w3.org/2000/svg">{"".join(body)}</svg>')
        key = [f'<div style="color:#22d3ee">━ {self.label} (faded = earlier steps)</div>']
        key += [f'<div style="color:#f59e0b">╍ {n} ({v["L"]} blocks, step {v["step"]}, loss {v["loss"]:.2f})</div>'
                for n, v in others.items()]
        return (f'<div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px">'
                f'<div>{svg(g)}<div style="font-size:10px;margin-left:{pad}px">{"".join(key)}</div></div>'
                f'<div>{svg(lchart)}</div></div>')
