#!/usr/bin/env python3
"""
Video extension script using LongViePipeline.

Loops over prompt segments and feeds the last frame and a short history back
into the pipeline to continue generation. Uses TemporalTiler_BCTHW (via
sliding_window_size / sliding_window_stride) for long-sequence processing.

Modes:
  - I2V extension: input_image + history (video[-8:]) + optional dense/sparse control.
  - V2V extension: pass input_video and denoising_strength for video-to-video step.

Usage (JSON segments, like inference.py):
  python video_extension.py --json_file cond.json --video_name out --image_path first.png \\
    --control_weight_path ./models/LongVie/control.safetensors

Usage (single prompt, N segments):
  python video_extension.py --prompt "Scene continues smoothly." --num_segments 3 \\
    --image_path first.png --video_name out --control_weight_path ...

Usage (V2V extension with denoising):
  python video_extension.py --json_file cond.json --input_video ./clip.mp4 \\
    --denoising_strength 0.85 --video_name out --control_weight_path ...
"""

import os
import json
import argparse
import torch
import torch.distributed as dist
from PIL import Image
import decord
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new_longvie import LongViePipeline, ModelConfig

# Target resolution (width, height); must match pipeline expectations
TARGET_SIZE = (640, 352)
# Frames per segment (must satisfy time_division: 4*k+1)
NUM_FRAMES_DEFAULT = 81
# History length fed back into pipeline (last N frames)
HISTORY_LEN = 8

# Sliding window for long sequences (latent time steps). Pipeline uses
# TemporalTiler_BCTHW.run(sliding_window_size, sliding_window_stride).
# For 81 frames, latent length = (81-1)//4+1 = 21. Overlap helps continuity.
SLIDING_WINDOW_SIZE_DEFAULT = 21
SLIDING_WINDOW_STRIDE_DEFAULT = 17


def resolve_media_path(path):
    """Resolve path so it works when cwd is longvie/ and user passes longvie/... or example/..."""
    if not path or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    # When cwd is longvie/, "longvie/example/..." does not exist; try "example/..."
    if path.startswith("longvie/"):
        alt = path[len("longvie/"):]
        if os.path.exists(alt):
            return os.path.abspath(alt)
    return os.path.abspath(path)


def load_image(path):
    path = resolve_media_path(path)
    return Image.open(path).convert("RGB").resize(TARGET_SIZE)


def resize_video_frames(video_np):
    return [Image.fromarray(frame).resize(TARGET_SIZE) for frame in video_np]


def load_video_frames(path, max_frames=None):
    """Load video as list of PIL Images (RGB), resized to TARGET_SIZE."""
    path = resolve_media_path(path)
    vr = decord.VideoReader(path)
    n = len(vr)
    if max_frames is not None:
        n = min(n, max_frames)
    frames = vr.get_batch(list(range(n))).asnumpy()
    return resize_video_frames(frames)


def main(args):
    pipe = LongViePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        use_usp=args.use_usp,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", skip_download=True),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu", skip_download=True),
        ],
        redirect_common_files=False,
        control_weight_path=args.control_weight_path,
        dit_weight_path=args.dit_weight_path,
        ring_degree=args.ring_degree,
        ulysses_degree=args.ulysses_degree,
    )
    pipe.enable_vram_management()

    # Build segment list: either from JSON or from single prompt
    if args.json_file:
        with open(args.json_file, "r") as f:
            samples = json.load(f)
    else:
        if not args.prompt:
            raise ValueError("Provide either --json_file or --prompt")
        samples = [{"text": args.prompt}] * args.num_segments
        if args.depth or args.track:
            depth_list = [args.depth] if isinstance(args.depth, str) else (args.depth or [])
            track_list = [args.track] if isinstance(args.track, str) else (args.track or [])
            for i, s in enumerate(samples):
                s["depth"] = depth_list[i] if i < len(depth_list) else (depth_list[0] if depth_list else None)
                s["track"] = track_list[i] if i < len(track_list) else (track_list[0] if track_list else None)

    if not args.image_path and not args.input_video:
        raise ValueError("Provide --image_path or --input_video for initial condition")

    # Initial condition: image + optional input_video (V2V) + history
    if args.input_video:
        video_frames = load_video_frames(args.input_video)
        image = video_frames[-1]
        history = video_frames[-HISTORY_LEN:] if len(video_frames) >= HISTORY_LEN else video_frames
    else:
        image = load_image(args.image_path) if args.image_path else None
        history = []

    noise = None
    prev_segment = None  # full previous segment for V2V mode
    num_frames = getattr(args, "num_frames", None) or NUM_FRAMES_DEFAULT
    sliding_window_size = getattr(args, "sliding_window_size", None) or SLIDING_WINDOW_SIZE_DEFAULT
    sliding_window_stride = getattr(args, "sliding_window_stride", None) or SLIDING_WINDOW_STRIDE_DEFAULT

    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    )

    for i, sample in enumerate(samples):
        prompt = sample.get("text", args.prompt or "")
        dense_frames = None
        sparse_frames = None
        if sample.get("depth") and sample.get("track"):
            dense_vr = decord.VideoReader(sample["depth"])
            sparse_vr = decord.VideoReader(sample["track"])
            dense_frames = resize_video_frames(dense_vr[:].asnumpy())
            sparse_frames = resize_video_frames(sparse_vr[:].asnumpy())

        # V2V: pass previous full segment as input_video and denoising_strength for continuity
        input_video = None
        denoising_strength = 1.0
        if args.v2v_mode and i > 0 and prev_segment is not None:
            input_video = prev_segment
            denoising_strength = args.denoising_strength

        pipe_kw = dict(
            input_image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=args.seed,
            tiled=args.tiled,
            height=TARGET_SIZE[1],
            width=TARGET_SIZE[0],
            num_frames=num_frames,
            history=history,
            noise=noise,
            sliding_window_size=sliding_window_size,
            sliding_window_stride=sliding_window_stride,
            input_video=input_video,
            denoising_strength=denoising_strength,
        )
        if dense_frames is not None and sparse_frames is not None:
            pipe_kw["dense_video"] = dense_frames
            pipe_kw["sparse_video"] = sparse_frames

        video, noise = pipe(**pipe_kw)

        # Chain for next segment: last frame as image, last HISTORY_LEN frames as history
        image = video[-1]
        history = video[-HISTORY_LEN:]
        prev_segment = video

        if not dist.is_initialized() or dist.get_rank() == 0:
            save_dir = os.path.join(args.output_dir or "./gen_videos", args.video_name)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{i}.mp4")
            save_video(video, save_path, fps=args.fps, quality=args.quality)
            print(f"[Saved] {save_path}")

        if dist.is_initialized():
            dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongVie video extension (I2V chain + optional V2V)")
    parser.add_argument("--json_file", type=str, default=None, help="JSON with list of {text, depth?, track?} (like inference.py)")
    parser.add_argument("--video_name", type=str, required=True, help="Output folder name under output_dir")
    parser.add_argument("--output_dir", type=str, default="./gen_videos", help="Base output directory")
    parser.add_argument("--image_path", type=str, default="", help="Initial image for I2V (ignored if --input_video set)")
    parser.add_argument("--input_video", type=str, default=None, help="Initial video: use last frame + history for first segment (or V2V with --denoising_strength)")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt when not using --json_file")
    parser.add_argument("--num_segments", type=int, default=1, help="Number of segments when using --prompt only")
    parser.add_argument("--depth", type=str, nargs="*", default=None, help="Depth video path(s) for control (when not using json)")
    parser.add_argument("--track", type=str, nargs="*", default=None, help="Track video path(s) (when not using json)")
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES_DEFAULT, help=f"Frames per segment (default {NUM_FRAMES_DEFAULT}, must be 4*k+1)")
    parser.add_argument("--sliding_window_size", type=int, default=SLIDING_WINDOW_SIZE_DEFAULT, help="TemporalTiler window size (latent steps)")
    parser.add_argument("--sliding_window_stride", type=int, default=SLIDING_WINDOW_STRIDE_DEFAULT, help="TemporalTiler stride (latent steps)")
    parser.add_argument("--v2v_mode", action="store_true", help="Use V2V for extension steps: input_video=previous segment, denoising_strength")
    parser.add_argument("--denoising_strength", type=float, default=0.85, help="Denoising strength for V2V steps (default 0.85)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiled", action="store_true", help="Use VAE tiling")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--quality", type=int, default=10)
    parser.add_argument("--control_weight_path", type=str, required=True)
    parser.add_argument("--dit_weight_path", type=str, default="")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--ulysses_degree", type=int, default=1)
    parser.add_argument("--ring_degree", type=int, default=1)
    parser.add_argument("--use_usp", action="store_true", help="Enable USP")
    args = parser.parse_args()
    main(args)
