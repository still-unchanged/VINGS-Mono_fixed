# Deploying VINGS-Mono on a new machine

Everything below is what this fork changed so a fresh clone runs without
editing yaml or tracing import errors.

## 1. Clone with submodules

```bash
git clone --recursive https://github.com/still-unchanged/VINGS-Mono_fixed.git
cd VINGS-Mono_fixed
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

## 2. Environment

Python 3.9, CUDA 11.8, torch 2.0.1.

```bash
conda create -n vings_vio python=3.9 -y
conda activate vings_vio
bash set_env.sh
```

`set_env.sh` installs torch, requirements, the CUDA extensions
(`diff-surfel-rasterization`, `dbaf`/`droid_backends`) and gtsam, then asserts
the VIO interfaces exist. It must **not** be run with `sudo` — the original
script did, which installs into root's python instead of your env.

### gtsam is the thing that breaks first

`mode: vio` needs `BA2GTSAM`, `CustomHessianFactor` and
`CombinedImuFactor.evaluateErrorCustom`. These exist **only** in the
`Promethe-us/gtsam` **`vio`** branch, which this repo pins as a submodule.
A plain `pip install gtsam` installs upstream 4.x, which lacks all three, and
you get a Python `AttributeError` at the first VIO alignment.

Check it:

```bash
python -c "import gtsam; print(hasattr(gtsam,'BA2GTSAM'), hasattr(gtsam.CombinedImuFactor,'evaluateErrorCustom'))"
# -> True True
```

## 3. Weights

Only `droid.pth` is needed for `mode: vio` / `vo`.

```bash
mkdir -p ckpts && cd ckpts
wget https://huggingface.co/Promethe-us/VINGS-Mono-Checkpoints/resolve/main/droid.pth
```

Optional: `metric_depth_vit_small_800k.pth` (only with `use_metric: True`),
`lightglue/*.onnx` (only with `use_loop: True`), `FastSAM-x.pt` (only with
`use_dynamic: True`).

## 4. Datasets

Put them under `./data/`, keeping the author's relative layout, e.g.

```
data/DATA/2024/RAW_KITTI/2011_09_30/2011_09_30_drive_0027_sync/
```

`load_config` rewrites the author's `/data/wuke/...` prefixes onto this
checkout automatically, so **you do not edit any yaml**:

| author path | resolves to |
|---|---|
| `/data/wuke/workspace/VINGS-Mono/x` | `<repo>/x` |
| `/data/wuke/DATA/x` | `<repo>/data/DATA/x` |
| `/data/wuke/workspace/x` | `<repo>/data/x` |

Dataset layout per dataset is in `docs/PREPARE_DATA.md`. Note the repo uses the
**processed** KITTI layout (`image_02/data` + `metadata/{calib,c2i,camstamp,imu}.txt`),
not the raw KITTI download.

Set `VINGS_ROOT` if the tree lives outside the checkout.

## 5. Run

**From the repo root**, using the `scripts/...` path:

```bash
python scripts/run.py configs/kitti/sync/kitti_2011_09_30_drive_0027.yaml
```

Running it from inside `configs/` fails with `can't open file`; the entry
points rely on `sys.path[0] == scripts/`.

## 6. Things that bite

**`image_size` must match the source aspect ratio** and both dimensions must be
divisible by 8 (the buffers are `h//8`). Every shipped config is now derived
from its own `intrinsic` H/W. A 16:9 `[344, 616]` on a 1226x370 panoramic strip
squashes horizontally by 1.85x: the tracker survives, but depth packaged for
the mapper collapses (13.7% valid pixels, 16/45 packages fully empty on KITTI
0027) and `run.py` dies with `cannot convert float NaN to integer`.

**VRAM.** The map only grows: 40k gaussians per keyframe, never evicted, and
Adam keeps full-size state, ~237 B/gaussian. 1106 keyframes ≈ 9.8 GiB. Keep
`use_storage_manager: True` (it offloads far-away gaussians to CPU) and/or
lower the `40000` in `gaussian_model.py:137`. On an 8 GiB card the full
sequence OOMs around keyframe 690 without the storage manager.

**numpy.** `pip install` over an existing numpy can leave a mixed tree
(`numpy/_core/*.so` from 2.x next to 1.26 python files), which surfaces as
`ValueError: object __array__ method not producing an array` from matplotlib.
If plotting dies, reinstall numpy cleanly inside the env.

**Multi-GPU is not supported.** There is no `DataParallel`/`DistributedDataParallel`
anywhere; the configs pin `cuda:0`. To use several cards, run one process per
card with `CUDA_VISIBLE_DEVICES=N`.
