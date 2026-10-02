"""Representation tour: the residual-stream point cloud at every block, packed for a grand tour in the browser.

All blocks are expressed in one shared principal basis of the residual stream, so the same 3D window
looks at every block at the same moment and you can watch the cloud get reorganized with depth.
"""
import base64, json, os

import torch

JS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tour.js")).read()


def sample(t, n_points):
    """The first few whole sequences of a (B, T, C) activation, as (N, C) float32 on the cpu."""
    B, T, C = t.shape
    return t.detach()[:rows(B, T, n_points)].reshape(-1, C).float().cpu()


def rows(B, T, n_points):
    return max(1, min(B, n_points // T))


def effective_dim(X):
    """Participation ratio of the covariance spectrum: ~C for an isotropic cloud, ~1 for a line."""
    s = torch.linalg.svdvals(X - X.mean(0)) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum()) if s.sum() > 0 else 0.0


def _procrustes(A, ref):
    """Rotate A's columns within their own span to best match ref, so the view doesn't jump between updates."""
    U, _, Vh = torch.linalg.svd(A.T @ ref)
    return A @ (U @ Vh)


class Tour:
    def __init__(self, k=8):
        self.k, self.W = k, None

    def payload(self, clouds, names, tokens, T, step):
        Xs = []
        for X in clouds:
            X = X - X.mean(0)
            Xs.append(X / X.pow(2).mean().sqrt().clamp_min(1e-12))  # shape only; magnitude is in the lanes above
        W = torch.linalg.svd(torch.cat(Xs), full_matrices=False).Vh.T  # C x C, columns = principal directions
        k = min(self.k, W.shape[1])
        if self.W is not None and self.W.shape == W.shape:
            W = torch.cat([_procrustes(W[:, :k], self.W[:, :k]), _procrustes(W[:, k:], self.W[:, k:])], 1)
        self.W = W
        layers = []
        for name, X in zip(names, Xs):
            Z = X @ W
            m = float(Z.abs().max().clamp_min(1e-12))
            q = (Z / m * 127).round().to(torch.int8).view(torch.uint8).flatten().tolist()
            layers.append(dict(name=name, dim=round(effective_dim(X), 1), scale=m / 127,
                               data=base64.b64encode(bytes(q)).decode()))
        return dict(step=step, C=W.shape[0], N=Xs[0].shape[0], T=T, k=k, layers=layers, tokens=tokens)


def embed(payload):
    """Self-contained HTML for notebooks: the engine plus this frame's data."""
    data = json.dumps(payload).replace("</", "<\\/")
    return (f'<div class="fs-tour-slot"></div><script>{JS}\n'
            f'FlowTour.attach(document.currentScript ? document.currentScript.previousElementSibling : '
            f'[...document.querySelectorAll(".fs-tour-slot")].pop(), {data});</script>')
