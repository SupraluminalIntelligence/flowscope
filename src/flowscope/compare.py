"""Compare how several checkpoints represent the same tokens, for FlowScope's tour and GrandTourVision.

capture() records the residual stream at every layer for chosen token positions from any Hugging Face model. It
uses output_hidden_states, so there are no architecture-specific hooks (hybrid conv/attention models work too).
frames() turns several captures into tour frames that share one principal basis, so the same 3D view compares
checkpoints directly, and adds two per-layer measures against the first checkpoint: linear CKA (similarity of the
representation up to rotation and scale) and relative change (how far each token's vector actually moved).
"""
import base64, gc

import torch

from flowscope.tour import effective_dim


def capture(model_id, sequences, windows, device, dtype=torch.bfloat16, revision=None):
    """Hidden states at the window positions of each sequence, for embeddings plus every layer.

    sequences: token-id lists; windows: (start, length) per sequence. The model sees each sequence up to the end of
    its window, so the window tokens carry their full context. Returns (layers, layer_types), where layers is a list of
    float32 tensors [total window tokens, hidden]. Loads the base model only (no LM head), then frees it.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, revision=revision, dtype=dtype).to(device).eval()
    per_layer = None
    with torch.no_grad():
        for ids, (start, n) in zip(sequences, windows):
            out = model(input_ids=torch.tensor([ids[:start + n]], device=device), output_hidden_states=True, use_cache=False)
            hidden = [h[0, start:start + n].float().cpu() for h in out.hidden_states]
            per_layer = per_layer or [[] for _ in hidden]
            for i, h in enumerate(hidden):
                per_layer[i].append(h)
            del out
    layer_types = getattr(model.config, "layer_types", None)
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return [torch.cat(layer) for layer in per_layer], layer_types


def linear_cka(x, y):
    """Linear centered kernel alignment: 1 = same representation up to rotation and scale, 0 = unrelated."""
    x = (x - x.mean(0)).double()
    y = (y - y.mean(0)).double()
    return float((x.T @ y).norm() ** 2 / ((x.T @ x).norm() * (y.T @ y).norm()))


def relative_change(ref, h):
    """Median over tokens of |h - ref| / |ref|: how far each token's vector moved, rotation and scale included."""
    return float(((h - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-12)).median())


def _unit(h):
    h = h - h.mean(0)
    return h / h.pow(2).mean().sqrt().clamp_min(1e-12)  # shape only, as in the live tour


def frames(captures, layer_names, *, tokens, texts, groups, group_names, T, title, k=128, toured=8):
    """One tour frame per checkpoint (in the order given), all in one shared basis of the top-k principal directions.

    captures: {checkpoint name: [layer tensors [N, hidden]]}. The first checkpoint is the reference for CKA.
    """
    names = list(captures)
    unit = {m: [_unit(h) for h in hs] for m, hs in captures.items()}
    pooled = torch.cat([h for hs in unit.values() for h in hs]).double()
    evals, evecs = torch.linalg.eigh(pooled.T @ pooled)
    basis = evecs[:, -k:].flip(1).float()  # hidden x k, most spread first
    retained = float(evals[-k:].sum() / evals.sum())
    ref = captures[names[0]]
    out = []
    for step, m in enumerate(names):
        layers = []
        for name, h in zip(layer_names, unit[m]):
            z = h @ basis
            s = float(z.abs().max().clamp_min(1e-12))
            q = (z / s * 127).round().to(torch.int8).view(torch.uint8).flatten().tolist()
            layers.append(dict(name=name, dim=round(effective_dim(h), 1), scale=s / 127, data=base64.b64encode(bytes(q)).decode()))
        out.append(dict(step=step, C=k, N=len(tokens), T=T, k=toured, layers=layers, tokens=tokens, label=title, name=m,
                        texts=texts, groups=groups, groupNames=group_names, retained=round(retained, 4),
                        similarity=[round(linear_cka(a, b), 4) for a, b in zip(ref, captures[m])],
                        change=[round(relative_change(a, b), 4) for a, b in zip(ref, captures[m])]))
    return out
