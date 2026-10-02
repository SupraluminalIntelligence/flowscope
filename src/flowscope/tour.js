// FlowScope representation tour: a grand tour (Asimov's torus method) through the residual stream.
// Every panel is one block's token cloud; all panels share the same moving 3D window.
(function () {
  if (window.FlowTour) return;

  const CSS = `
  .fs-tour{margin-top:14px;color:#d5dae6;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
  .fs-tour-head{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
  .fs-tour-head b{font-size:13px;color:#fff}
  .fs-tour-ctl{display:flex;gap:16px;flex-wrap:wrap;align-items:center;color:#8b93a7;margin-bottom:8px}
  .fs-tour-ctl input[type=range]{vertical-align:middle;width:110px;accent-color:#22d3ee}
  .fs-tour-ctl button,.fs-tour-ctl select{background:#151a26;color:#d5dae6;border:1px solid #2a3142;border-radius:5px;
    padding:3px 8px;font:inherit;cursor:pointer}
  .fs-tour-grid{display:flex;flex-wrap:wrap;gap:8px}
  .fs-panel{background:#0f1320;border:1px solid #1f2533;border-radius:8px;padding:4px;cursor:zoom-in}
  .fs-panel.big{cursor:zoom-out;border-color:#2f3a52}
  .fs-panel .lbl{display:flex;justify-content:space-between;color:#8b93a7;font-size:10px;padding:0 3px 2px}
  .fs-panel .lbl b{color:#d5dae6;font-weight:600}
  .fs-panel canvas{display:block}
  .fs-tour-note{color:#6b7280;font-size:10px;margin-top:6px;max-width:900px;line-height:1.5}`;

  const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  const SMALL = 168, BIG = 420;

  function viridis(t) {
    t = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
    const i = Math.min(VIRIDIS.length - 2, Math.floor(t)), f = t - i, a = VIRIDIS[i], b = VIRIDIS[i + 1];
    return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
  }

  function decode(layer, N, C) {
    const bin = atob(layer.data), z = new Float32Array(N * C);
    for (let i = 0; i < z.length; i++) {
      const b = bin.charCodeAt(i);
      z[i] = (b > 127 ? b - 256 : b) * layer.scale;
    }
    const v = new Float64Array(C);  // variance along each principal direction (data is centered)
    for (let i = 0; i < z.length; i++) v[i % C] += z[i] * z[i] / N;
    return { name: layer.name, dim: layer.dim, z, v, total: v.reduce((a, b) => a + b, 0) || 1 };
  }

  // Torus method: rotate in every coordinate plane (i, j) at its own speed; with rationally independent
  // speeds the path never repeats and comes arbitrarily close to every 3D projection.
  function speeds(n) {
    const out = [];
    let norm = 0, m = 2;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        while (Number.isInteger(Math.sqrt(m))) m++;
        const v = 0.25 + 0.75 * (Math.sqrt(m) % 1);  // fractional parts of sqrt(non-squares): irrational
        out.push([i, j, v]);
        norm += v * v;
        m++;
      }
    const k = 0.5 * Math.sqrt(3) / Math.sqrt(norm);  // keep the visible motion calm whatever n is
    return out.map(([i, j, v]) => [i, j, v * k * Math.sqrt(n)]);
  }

  class Engine {
    constructor() {
      if (!document.getElementById("fs-tour-css")) {
        const st = document.createElement("style");
        st.id = "fs-tour-css";
        st.textContent = CSS;
        document.head.appendChild(st);
      }
      this.el = document.createElement("div");
      this.el.className = "fs-tour";
      this.el.innerHTML = `
        <div class="fs-tour-head"><b>representation tour</b><span class="info"></span></div>
        <div class="fs-tour-ctl">
          <label>dims toured <input class="n" type="range" min="3" max="64" value="8"> <span class="nv">8</span></label>
          <label>speed <input class="sp" type="range" min="0" max="3" step="0.1" value="1"></label>
          <button class="pause">pause</button><button class="reset">back to PC1-3</button>
          <label>color <select class="col"><option value="position">position in sequence</option>
            <option value="token">token id</option></select></label>
        </div>
        <div class="fs-tour-grid"></div>
        <div class="fs-tour-note">Each dot is one token's residual vector at that block, normalized to unit RMS (shape only;
          size is in the lanes above). All panels look through the same moving 3D window, touring the top principal
          directions of the residual stream. <b>eff.dim</b> = how many directions the cloud really uses (participation
          ratio). Low at step 0 = rank collapse. <b>% seen</b> = share of the block's variance inside the toured
          dims; when it is low, raise "dims toured" or the panel is mostly showing you what's left over. Click a panel to enlarge.</div>`;
      const q = s => this.el.querySelector(s);
      this.grid = q(".fs-tour-grid");
      this.info = q(".info");
      this.t = 0; this.speed = 1; this.paused = false; this.n = null; this.color = "position";
      this.layers = []; this.prev = []; this.fadeStart = 0; this.panels = [];
      q(".n").oninput = e => { this.n = +e.target.value; q(".nv").textContent = this.n; this.sp = speeds(this.n); this.label(); };
      q(".sp").oninput = e => { this.speed = +e.target.value; };
      q(".pause").onclick = e => { this.paused = !this.paused; e.target.textContent = this.paused ? "play" : "pause"; };
      q(".reset").onclick = () => { this.t = 0; };
      q(".col").onchange = e => { this.color = e.target.value; this.paint(); };
      this.last = performance.now();
      requestAnimationFrame(this.tick.bind(this));
    }

    update(p) {
      if (!p || p.C < 3) return;
      const first = this.n === null;
      this.p = p;
      this.prev = this.layers;
      this.layers = p.layers.map(l => decode(l, p.N, p.C));
      this.fadeStart = performance.now();
      const nIn = this.el.querySelector(".n");
      nIn.max = p.C;
      if (first || this.n > p.C) {
        this.n = Math.min(p.k, p.C);
        nIn.value = this.n;
        this.el.querySelector(".nv").textContent = this.n;
        this.sp = speeds(this.n);
      }
      this.el.querySelector(".col").disabled = !p.tokens;
      if (!p.tokens) this.color = "position";
      this.info.textContent = `step ${p.step} · ${p.N} tokens × ${p.C} dims · ${this.layers.length} panels`;
      if (this.panels.length !== this.layers.length) this.buildPanels();
      this.label();
      this.paint();
    }

    // Zoom: a typical 3D view of an n-dim cloud carries ~3/n of its variance, so size panels by that, but cap it
    // so a collapsed cloud still fits when the window lines up with it. Also report how much of the block is toured.
    zoom(l) {
      const { N, C } = this.p, n = this.n;
      let inView = 0;
      for (let d = 0; d < n; d++) inView += l.v[d];
      const r = new Float64Array(N);
      for (let p = 0; p < N; p++)
        for (let d = 0; d < n; d++) r[p] += l.z[p * C + d] ** 2;
      const r95 = Math.sqrt(r.sort()[Math.floor(0.95 * (N - 1))]);  // a few outlier tokens may leave the panel
      l.fit = 1 / Math.max(2.6 * Math.sqrt(inView / n), r95 / 1.1, 1e-9);  // view units -> half-panel
      return inView;
    }

    label() {
      this.prev.forEach(l => this.zoom(l));
      this.layers.forEach((l, i) => {
        const pct = Math.round(100 * this.zoom(l) / l.total);
        this.panels[i].lbl.innerHTML = `<b>${l.name}</b><span title="eff.dim: directions the cloud really uses · `
          + `% seen: share of its variance inside the toured dims">dim ${l.dim} · <span style="color:${
          pct < 50 ? "#fbbf24" : "inherit"}">${pct}% seen</span></span>`;
      });
    }

    paint() {
      const p = this.p;
      if (!p) return;
      this.colors = Array.from({ length: p.N }, (_, i) => this.color === "token" && p.tokens
        ? `hsl(${(p.tokens[i] * 137.508) % 360},70%,62%)`
        : viridis((i % p.T) / Math.max(1, p.T - 1)));
    }

    buildPanels() {
      this.grid.innerHTML = "";
      this.panels = this.layers.map(() => {
        const div = document.createElement("div");
        div.className = "fs-panel";
        const lbl = document.createElement("div");
        lbl.className = "lbl";
        const cv = document.createElement("canvas");
        div.append(lbl, cv);
        const panel = { div, lbl, cv, big: false };
        div.onclick = () => { panel.big = !panel.big; div.classList.toggle("big", panel.big); this.size(panel); };
        this.size(panel);
        this.grid.appendChild(div);
        return panel;
      });
    }

    size(panel) {
      const s = panel.big ? BIG : SMALL, dpr = window.devicePixelRatio || 1;
      panel.cv.width = panel.cv.height = s * dpr;
      panel.cv.style.width = panel.cv.style.height = s + "px";
      panel.px = s;
    }

    basis() {
      const n = this.n, v = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
      v[0][0] = v[1][1] = v[2][2] = 1;
      for (const [i, j, w] of this.sp) {
        const th = w * this.t, c = Math.cos(th), s = Math.sin(th);
        for (const b of v) {
          const a = b[i], d = b[j];
          b[i] = c * a - s * d;
          b[j] = s * a + c * d;
        }
      }
      return v;
    }

    tick(now) {
      const dt = Math.min(0.1, (now - this.last) / 1000);
      this.last = now;
      if (!this.paused) this.t += dt * this.speed;
      if (this.layers.length && this.el.isConnected) {
        const B = this.basis(), f = Math.min(1, (now - this.fadeStart) / 700);
        this.panels.forEach((panel, i) => {
          const ctx = panel.cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, panel.px, panel.px);
          if (panel.big) this.spokes(ctx, panel.px, B);
          if (f < 1 && this.prev[i]) this.draw(ctx, panel.px, this.prev[i], B, 1 - f);
          this.draw(ctx, panel.px, this.layers[i], B, f);
        });
      }
      requestAnimationFrame(this.tick.bind(this));
    }

    draw(ctx, S, layer, B, alpha) {
      const { N, C } = this.p, n = this.n, z = layer.z, R = S * 0.44 * layer.fit;
      const pts = new Array(N);
      for (let p = 0; p < N; p++) {
        let x = 0, y = 0, w = 0;
        const o = p * C;
        for (let d = 0; d < n; d++) {
          const v = z[o + d];
          x += B[0][d] * v; y += B[1][d] * v; w += B[2][d] * v;
        }
        pts[p] = [x, y, w * layer.fit, p];
      }
      pts.sort((a, b) => a[2] - b[2]);  // far points first
      for (const [x, y, w, p] of pts) {
        const depth = Math.min(1, Math.max(0, (w + 1) / 2)), k = 4 / (4 - Math.max(-1.5, Math.min(1.5, w)));
        ctx.globalAlpha = alpha * (0.35 + 0.65 * depth);
        ctx.fillStyle = this.colors[p];
        ctx.beginPath();
        ctx.arc(S / 2 + x * R * k, S / 2 - y * R * k, (1.1 + 1.6 * depth) * (S > SMALL ? 1.5 : 1), 0, 6.2832);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    spokes(ctx, S, B) {
      // where each principal direction points in the current view (length = how much of it you are seeing)
      ctx.font = "10px ui-monospace,Menlo,monospace";
      for (let d = 0; d < Math.min(this.n, 8); d++) {
        const x = S / 2 + B[0][d] * S * 0.42, y = S / 2 - B[1][d] * S * 0.42;
        ctx.strokeStyle = ctx.fillStyle = "rgba(139,147,167,0.45)";
        ctx.beginPath(); ctx.moveTo(S / 2, S / 2); ctx.lineTo(x, y); ctx.stroke();
        ctx.fillText("PC" + (d + 1), x + 3, y - 3);
      }
    }
  }

  window.FlowTour = {
    attach(slot, payload) {
      if (!window.__fsTour) window.__fsTour = new Engine();
      const eng = window.__fsTour;
      if (slot && eng.el.parentNode !== slot) slot.appendChild(eng.el);
      eng.update(payload);
    },
  };
})();
