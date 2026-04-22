# TRELLIS.2 MPS Port Notes

This private package is a source-only Apple Silicon/MPS packaging of Microsoft
TRELLIS.2. It keeps the original upstream source under `TRELLIS.2/` and adds
local Mac compatibility patches, setup scripts, backends, and validation
manifests.

Upstream source:

- `microsoft/TRELLIS.2`
- Local vendored source commit: `5565d24`

Model checkpoints, local virtual environments, caches, generated outputs, and
workspace secrets are not included. Users must authenticate with Hugging Face
and download gated weights under the upstream model terms.

## What Changed

- Default attention backends are set to PyTorch SDPA.
- Sparse convolution uses a pure PyTorch fallback backend instead of CUDA.
- Mesh extraction uses a Python fallback for the flexible dual-grid path.
- Texture baking can use the available Metal path and can fall back to the
  included xatlas/KDTree path when Metal components are unavailable.
- Hard CUDA-only paths in the vendored TRELLIS.2 checkout were patched to select
  the active local device where possible.
- `generate.py` loads Hugging Face tokens from a local `.env` file found in the
  current directory or one of the repo parents. It never prints token values.
- Astronaut visual review assets were added under `assets/astronaut/` for a
  compact README example of input image to generated 3D asset.

## Local Validation

The included manifest is `manifests/trellis_mac_probe.json`.

Validated on local Apple Silicon:

- Torch MPS available: true.
- Dense attention on MPS: output shape `[1, 4, 2, 8]`.
- Sparse attention on MPS: output shape `[5, 2, 4]`.
- Pure PyTorch sparse convolution on MPS: output shape `[8, 3]`.
- Mesh extraction fallback: vertices `[4, 3]`, faces `[2, 3]`.
- GLB postprocessing path: generated a valid tiny GLB.
- KDTree texture baker fallback: generated a valid tiny textured GLB.

Full image-to-3D validation completed with:

- Input: `assets/shoe_input.png`.
- Pipeline: `512`.
- Seed: `42`.
- Texture size: `512`.
- Output GLB in the original workspace: 7.6 MB.
- Blender validation: 1 mesh, 1 material, 153,708 vertices, 197,529 triangles,
  no findings.

Astronaut visual assets:

- `assets/astronaut/astronaut_input_front.png`
- `assets/astronaut/astronaut_input_to_3d_turntable.gif`
- `assets/astronaut/astronaut_generated_contact_sheet.png`

## Requirements

- Apple Silicon Mac.
- Python 3.11+.
- Xcode command line tools.
- Metal toolchain if you want the accelerated texture baking path:

```bash
xcodebuild -downloadComponent MetalToolchain
```

- Hugging Face access to:
  - `microsoft/TRELLIS.2-4B`
  - `facebook/dinov3-vitl16-pretrain-lvd1689m`
  - `briaai/RMBG-2.0`

## Recommended Setup

```bash
bash setup.sh
source .venv/bin/activate
python generate.py assets/shoe_input.png --output outputs/shoe_probe --pipeline-type 512 --texture-size 512 --seed 42
```

Place secrets in a local `.env` file that is not committed:

```bash
HF_TOKEN=hf_xxx
```

## Known Limits

- This is an MPS/Metal compatibility package, not a CUDA-equivalent rewrite of
  every upstream kernel.
- Very high-resolution modes can exceed 24 GB unified memory.
- Some unsupported PyTorch MPS operations can fall back to CPU.
- Texture baking and mesh cleanup quality can vary depending on whether the
  Metal toolchain and optional acceleration components are available.
- Training is not supported in this Mac package.

## Files Intentionally Excluded

- Model weights and checkpoints.
- `.venv`, package caches, Hugging Face caches, and build artifacts.
- Generated GLB/OBJ/video/render outputs.
- Local workspace `.env` and any secrets.
