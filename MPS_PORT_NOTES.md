# TRELLIS.2 MPS Port Notes

This private package is a source-only Apple Silicon/MPS packaging of the
locally validated TRELLIS.2 path. It is based on:

- `shivampkumar/trellis-mac` at local source commit `257397f`.
- Nested `microsoft/TRELLIS.2` at local source commit `5565d24`.

Model checkpoints, local virtual environments, caches, generated outputs, and
workspace secrets are not included. Users must authenticate with Hugging Face
and download gated weights under the upstream model terms.

## What Was Changed

- Default attention backends are set to PyTorch SDPA.
- Sparse convolution uses a pure PyTorch fallback backend instead of CUDA.
- Mesh extraction uses a Python fallback for the flexible dual-grid path.
- Texture baking prefers the Apple/Metal stack:
  `mtldiffrast`, `mtlbvh`, `mtlmesh`/`cumesh`, `mtlgemm`/`flex_gemm`, and an
  Apple-compatible `o_voxel.postprocess.to_glb` path.
- If the Metal toolchain is unavailable, texture baking can fall back to the
  included xatlas/KDTree path.
- Hard CUDA-only paths in the nested TRELLIS.2 checkout were patched to select
  the active local device where possible.
- `generate.py` loads Hugging Face tokens from a local `.env` file found in the
  current directory or one of the repo parents. It never prints token values.

## Local Validation

The included manifest is `manifests/trellis_mac_probe.json`.

Validated on local Apple Silicon:

- Torch MPS available: true.
- Dense attention on MPS: output shape `[1, 4, 2, 8]`.
- Sparse attention on MPS: output shape `[5, 2, 4]`.
- Pure PyTorch sparse convolution on MPS: output shape `[8, 3]`.
- Mesh extraction fallback: vertices `[4, 3]`, faces `[2, 3]`.
- Metal `o_voxel.postprocess.to_glb`: generated a valid tiny GLB.
- KDTree texture baker fallback: generated a valid tiny textured GLB.

Full image-to-3D validation completed with:

- Input: `assets/shoe_input.png`.
- Pipeline: `512`.
- Seed: `42`.
- Texture size: `512`.
- Output GLB in the original workspace: 7.6 MB.
- Blender validation: 1 mesh, 1 material, 153,708 vertices, 197,529 triangles,
  no findings.

## Requirements

- Apple Silicon Mac.
- Python 3.11+.
- Xcode command line tools.
- Metal toolchain if you want the accelerated texture baker:

```bash
xcodebuild -downloadComponent MetalToolchain
```

- Hugging Face access to:
  - `microsoft/TRELLIS.2-4B`
  - `facebook/dinov3-vitl16-pretrain-lvd1689m`
  - `briaai/RMBG-2.0`

## Recommended Setup

The upstream setup script creates `.venv` and installs the patched dependencies:

```bash
bash setup.sh
source .venv/bin/activate
python generate.py assets/shoe_input.png --output shoe_probe --pipeline-type 512 --texture-size 512
```

For a `uv`-managed environment, use:

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
bash setup.sh
python generate.py assets/shoe_input.png --output shoe_probe --pipeline-type 512 --texture-size 512
```

Place secrets in a local `.env` file that is not committed:

```bash
HF_TOKEN=hf_xxx
```

## Known Limits

- This is an MPS/Metal port, not a CUDA-equivalent rewrite of every upstream
  kernel.
- Very high-resolution modes can exceed 24 GB unified memory.
- The Apple `o_voxel` path is validated for GLB postprocessing, but it is not a
  full drop-in replacement for every upstream CUDA submodule.
- PyTorch can emit MPS CPU fallback warnings for unsupported ops. The validated
  generation path still completed successfully.

## Files Intentionally Excluded

- Model weights and checkpoints.
- `.venv`, package caches, Hugging Face caches, and build artifacts.
- Generated GLB/OBJ/video/render outputs.
- Local workspace `.env` and any secrets.
