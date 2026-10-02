# FlowScope

Watch the numbers flow through your network while it trains: the forward pass through the residual
stream, and the gradient flowing back. Blockages (vanishing gradients), explosions, saturated attention
and dead units show up as they happen. Example frames: `docs/*.html`.

## Run any training script, unmodified

```bash
uv sync
uv run flowscope run examples/gpt.py                             # healthy baseline
uv run flowscope run examples/gpt.py --layers 12 --no-residual   # watch it block
```

Open http://127.0.0.1:8765 in any browser pane (VS Code/Cursor: "Simple Browser: Show"). Leave the tab
open: it keeps the last frame when a run ends and picks up the next run on its own. Each run is saved in
`.flowscope/runs.json` and drawn as a dashed line, so runs compare across invocations.

Watch from a Vision Pro (GrandTourVision's layer gallery) or another machine: `--host 0.0.0.0` serves on your
network and prints the address to use. Save a run for replay with `--record run.jsonl`.

`flowscope run` waits for the first `optimizer.step()`, finds the model that owns the optimizer's
parameters, and attaches hooks. Options: `--every N` (redraw interval), `--port`, `--label`.

## Or attach explicitly

```python
from flowscope import FlowScope
scope = FlowScope(model, optimizer, every=50, label="baseline")
# ... normal training loop ...
```

In Jupyter/Colab it draws inline; in a script it serves the live page.
`notebooks/gpt-dev-flowscope.ipynb` is Karpathy's gpt-dev notebook with FlowScope cells and experiments.

## What you're looking at

- **Forward lane**: residual stream RMS at each block boundary. Red = signal shrinking toward zero.
- **Backward lane**: `dloss/dx` relative to the last block. Dark red toward the input = early layers get
  almost no learning signal.
- **Pulses**: sweep forward, then backward; their size is the signal size.
- **Strips**: one token's actual residual vector and its gradient at each point.
- **Per block**: attention map (head 1), `p̂` = average max attention probability (near 1 = saturated
  softmax), dead MLP units, update/data ratio (~1e-3 is healthy).
- **Representation tour**: every token's residual vector at each block (256 tokens), all seen through one
  slowly rotating 3D window. It's a grand tour (Asimov's torus method: rotate in every coordinate plane at
  once, at irrationally related speeds, so the window eventually passes near every 3D view) through the
  top principal directions of the residual stream. `dim` = effective dimensionality (participation ratio);
  `% seen` = share of the block's variance inside the toured dims. Without residuals the stream collapses
  to ~2 dims within a few blocks, already at step 0 (rank collapse). Healthy nets also narrow with depth
  as they train; compare against step 0.

## Scope today

Residual-stream transformers shaped like nanoGPT / Karpathy's GPT (`blocks` or `transformer.h`, with
`sa`/`attn` and `ffwd`/`mlp` children). Attention maps need the attention probs to pass through a
`dropout` module (true for the Zero-to-Hero code, not for fused `scaled_dot_product_attention`).

## Development

After changing `src/flowscope`, regenerate the notebook's install cell: `uv run python scripts/build_notebook.py`.


```bash
uv run pytest -q
```
