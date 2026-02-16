# Install and run extend_Video.py (traveler example)

Use this to extend the video in `example/traveler/` with LongVie.

---

## 1. Environment

- **Python**: 3.10  
- **CUDA**: 12.1 (for PyTorch and flash-attention)  
- **Recommended**: Conda

```bash
conda create -n longvie python=3.10 -y
conda activate longvie
conda install psutil -y
```

---

## 2. PyTorch (CUDA 12.1)

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install ninja
```

---

## 3. Flash Attention

Either use a **prebuilt wheel** (no CUDA toolkit needed):

```bash
# Python 3.10, PyTorch 2.5, CUDA 12.x (try cu128 wheel on cu121)
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.2/flash_attn-2.7.4+cu128torch2.5-cp310-cp310-linux_x86_64.whl --no-deps
```

Or **build from source** (requires `CUDA_HOME` and `nvcc`):

```bash
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2.post1
```

---

## 4. LongVie repo and Python deps

From the **LongVie** project root:

```bash
cd /home/ubuntu/LongVie
pip install -e .
```

This installs `diffsynth` and everything in `requirements.txt` (torch, torchvision, transformers, imageio, imageio[ffmpeg], safetensors, einops, sentencepiece, protobuf, modelscope, ftfy, pynvml, pandas, accelerate, peft, decord).

---

## 5. Extra dependencies for utils (depth + track)

Used by `utils/get_depth.py` and `utils/get_track.py`:

```bash
pip install opencv-python Pillow numpy mediapy flow-vis moviepy timm cupy-cuda12x easydict prettytable scikit-image
```

- **flow-vis**: if not found, try `pip install flow_vis` or install from [flow_vis](https://github.com/tomrunia/OpticalFlow_Visualization).
- **cupy**: match your CUDA version (e.g. `cupy-cuda12x` for CUDA 12).

---

## 6. Download weights and checkpoints

### 6.1 Base video model (Wan2.1 I2V)

From **LongVie** root:

```bash
python download_wan2.1.py
```

Puts models under `models/Wan-AI/Wan2.1-I2V-14B-480P` and `Wan2.1-T2V-1.3B`.

### 6.2 LongVie control weights

Download [LongVie2 weights](https://huggingface.co/Vchitect/LongVie2) and place:

- `control.safetensors` → `LongVie/models/LongVie/control.safetensors`
- `dit.safetensors` → `LongVie/models/LongVie/dit.safetensors`

### 6.3 Depth + track checkpoints (one command)

From **LongVie** root, run:

```bash
python3 utils/download_checkpoints.py
```

This downloads **video_depth_anything_vitl.pth** (~1.5 GB) from Hugging Face into `utils/checkpoints/`.  
If **spaT_final.pth** is missing, the script prints a link: download it from [SpaTracker Google Drive](https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ?usp=sharing) and put it in `LongVie/utils/checkpoints/`.

### 6.4 Manual checkpoint locations

| Checkpoint | Path | Source |
|------------|------|--------|
| Depth (vitl) | `utils/checkpoints/video_depth_anything_vitl.pth` | [HF: depth-anything/Video-Depth-Anything-Large](https://huggingface.co/depth-anything/Video-Depth-Anything-Large) or `python3 utils/download_checkpoints.py` |
| Track | `utils/checkpoints/spaT_final.pth` | [SpaTracker Google Drive](https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ?usp=sharing) |

---

## 7. Run: extend the traveler video

From the **LongVie** root:

```bash
cd /home/ubuntu/LongVie

python extend_Video.py \
  --input_video ./example/traveler/traveler.mp4 \
  --target_duration 10
```

- **Input**: `example/traveler/traveler.mp4`  
- **Target length**: 10 seconds (two ~5 s segments). Change `--target_duration` as needed (e.g. 15 for 15 s).  
- **Output**: `./gen_videos/traveler_extended/` (and a working dir `example/traveler/traveler_ext/`).

Optional arguments:

- `--output_name NAME` – output subdir name (default: from input filename, e.g. `traveler_extended`).  
- `--prompt "Your text prompt"` – override the default continuation prompt.  
- `--work_dir PATH` – custom working directory.  
- `--control_weight_path PATH` – default `./models/LongVie/control.safetensors`.  
- `--dit_weight_path PATH` – default `./models/LongVie/dit.safetensors`.  
- `--skip_extraction` – skip depth/track extraction and use existing files in `work_dir` (e.g. `first.png`, depth, track).

Example with custom prompt and output name:

```bash
python extend_Video.py \
  --input_video ./example/traveler/traveler.mp4 \
  --target_duration 10 \
  --output_name traveler_10s \
  --prompt "The traveler keeps walking along the path with the same lighting and style."
```

---

## 8. Quick checklist

| Item | Location / Command |
|------|--------------------|
| Conda env | `conda activate longvie` |
| PyTorch 2.5.1 + CUDA 12.1 | `pip install torch==2.5.1 ... cu121` |
| Flash attention | Prebuilt wheel or `pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2.post1` |
| LongVie package | `cd LongVie && pip install -e .` |
| Utils deps | `pip install opencv-python Pillow numpy mediapy flow-vis moviepy timm cupy-cuda12x easydict prettytable scikit-image` |
| Wan2.1 | `python download_wan2.1.py` |
| LongVie weights | `models/LongVie/control.safetensors`, `dit.safetensors` |
| Depth checkpoint | `utils/checkpoints/video_depth_anything_vitl.pth` |
| Track checkpoint | `utils/checkpoints/spaT_final.pth` |
| Run | `python extend_Video.py --input_video ./example/traveler/traveler.mp4 --target_duration 10` |

---

## 9. Troubleshooting

- **"CUDA_HOME is not set"** when building flash-attention: set `CUDA_HOME` to your CUDA root (e.g. `/usr/local/cuda-12.2`) and ensure `nvcc` is on `PATH`, or use the prebuilt wheel.  
- **"No module named 'models.vda'"** (or similar): run from **LongVie** root and use `pip install -e .` so the package and `utils` are on the path.  
- **"checkpoint not found"**: ensure `utils/checkpoints/` has `video_depth_anything_vitl.pth` and `spaT_final.pth`; paths are relative to `LongVie/utils` when running `get_depth.py` / `get_track.py`.  
- **Out of GPU memory**: reduce resolution or segment count; extension uses ~5 s per segment, so `--target_duration 5` uses one segment.
