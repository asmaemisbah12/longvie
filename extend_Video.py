#!/usr/bin/env python3
"""
Extend a video using LongVie - single script that runs the full pipeline.
Takes an input video (e.g. 5 sec) and extends it (e.g. to 10 sec) by generating
additional content guided by depth and track extracted from the input.

Usage:
    python extend_video.py --input_video ./input_video/traveler.mp4 --target_duration 10
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import decord
import imageio
import numpy as np
from PIL import Image

# Segment length: LongVie generates 81 frames (~5 sec at 16fps) per segment
FRAMES_PER_SEGMENT = 81
FPS = 16


def extract_first_frame(video_path: str, output_path: str, use_last: bool = True):
    """Extract first or last frame from video as PNG."""
    vr = decord.VideoReader(video_path)
    frame_idx = len(vr) - 1 if use_last else 0
    frame = vr[frame_idx].asnumpy()
    Image.fromarray(frame).convert("RGB").save(output_path)
    print(f"[✓] Extracted {'last' if use_last else 'first'} frame to {output_path}")


def run_get_depth(video_path: str, output_dir: str, utils_dir: Path):
    """Run utils/get_depth.py to extract depth maps."""
    output_path = os.path.join(output_dir, "depth_raw")
    os.makedirs(output_path, exist_ok=True)
    npy_path = os.path.join(output_path, os.path.splitext(os.path.basename(video_path))[0] + "_depth.npy")

    cmd = [
        sys.executable,
        "get_depth.py",
        "--input_video", os.path.abspath(video_path),
        "--output_dir", os.path.abspath(output_path),
    ]
    result = subprocess.run(cmd, cwd=utils_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"get_depth.py failed: {result.stderr}")

    if not os.path.exists(npy_path):
        raise RuntimeError(f"Depth output not found: {npy_path}")
    print(f"[✓] Depth extracted to {npy_path}")
    return npy_path


def depth_npy_to_segments(
    npy_path: str,
    output_dir: str,
    video_name: str,
    num_segments: int = 1,
):
    """
    Convert depth .npy to depth_XX.mp4 segments (81 frames each).
    Pads if needed for short videos. For extension, uses last 81 frames (repeated if needed).
    """
    depth = np.load(npy_path)
    n_frames = len(depth)

    # Preprocess: clip and normalize
    p95, p5 = np.percentile(depth, [95, 5])
    depth = np.clip(depth, p5, p95)
    depth = (p95 - depth) / (p95 - p5 + 1e-8)

    seg_dir = os.path.join(output_dir, video_name)
    os.makedirs(seg_dir, exist_ok=True)

    # For extension: use last 81 frames for all segments (extrapolate continuation)
    if n_frames >= FRAMES_PER_SEGMENT:
        base_depth = depth[-FRAMES_PER_SEGMENT:]
    else:
        base_depth = np.concatenate([
            depth,
            np.tile(depth[-1:], (FRAMES_PER_SEGMENT - n_frames, 1, 1))
        ], axis=0)

    for i in range(num_segments):
        sub_depth = base_depth.copy()

        sub_depth_uint8 = (sub_depth * 255).astype(np.uint8)
        sub_depth_rgb = np.stack([sub_depth_uint8] * 3, axis=-1)

        out_path = os.path.join(seg_dir, f"depth_{i:02d}.mp4")
        writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=10, macro_block_size=1)
        for frame in sub_depth_rgb:
            writer.append_data(frame)
        writer.close()
        print(f"[✓] Saved {out_path}")

    return seg_dir


def extract_video_segment(video_path: str, output_path: str, num_frames: int = FRAMES_PER_SEGMENT):
    """Extract last N frames from video as a segment (for get_track)."""
    vr = decord.VideoReader(video_path)
    n = len(vr)
    start = max(0, n - num_frames)
    indices = list(range(start, min(n, start + num_frames)))
    frames = vr.get_batch(indices).asnumpy()

    # Pad if needed
    if len(frames) < num_frames:
        pad = np.tile(frames[-1:], (num_frames - len(frames), 1, 1, 1))
        frames = np.concatenate([frames, pad], axis=0)
    frames = frames[:num_frames]

    writer = imageio.get_writer(output_path, fps=FPS, codec="libx264", quality=10, macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"[✓] Extracted video segment to {output_path}")


def run_get_track(video_path: str, depth_path: str, save_path: str, utils_dir: Path):
    """Run utils/get_track.py to extract track visualization."""
    video_abs = os.path.abspath(video_path)
    depth_abs = os.path.abspath(depth_path)
    save_abs = os.path.abspath(save_path)

    cmd = [
        sys.executable,
        "get_track.py",
        "--video", video_abs,
        "--depth", depth_abs,
        "--save_path", save_abs,
    ]
    result = subprocess.run(cmd, cwd=str(utils_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"get_track.py failed: {result.stderr}")
    print(f"[✓] Track saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Extend video using LongVie (single script)")
    parser.add_argument("--input_video", type=str, required=True, help="Path to input video (e.g. 5 sec)")
    parser.add_argument("--target_duration", type=int, default=10, help="Target duration in seconds (default: 10)")
    parser.add_argument("--output_name", type=str, default=None, help="Output video name (default: from input filename)")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for extension (default: generic continuation)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control_weight_path", type=str, default="./models/LongVie/control.safetensors")
    parser.add_argument("--dit_weight_path", type=str, default="./models/LongVie/dit.safetensors")
    parser.add_argument("--work_dir", type=str, default=None, help="Working directory (default: input_video/<name>_ext)")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip depth/track extraction (use existing)")
    args = parser.parse_args()

    input_path = Path(args.input_video).resolve()
    if not input_path.exists():
        sys.exit(f"Error: Input video not found: {input_path}")

    video_name = args.output_name or input_path.stem
    work_dir = Path(args.work_dir or str(input_path.parent / f"{video_name}_ext"))
    work_dir.mkdir(parents=True, exist_ok=True)
    utils_dir = Path(__file__).resolve().parent / "utils"

    # Compute number of extension segments: target_duration / 5 sec per segment
    num_segments = max(1, args.target_duration // 5)
    print(f"[*] Extending to {args.target_duration}s = {num_segments} segment(s) (~5 sec each)")

    default_prompt = (
        "The video continues the scene with smooth, natural motion. "
        "The camera maintains a steady perspective as the action flows seamlessly. "
        "Consistent lighting and atmosphere throughout the sequence."
    )
    prompt = args.prompt or default_prompt

    if not args.skip_extraction:
        # 1. Extract last frame as starting image for generation
        first_frame_path = work_dir / "first.png"
        extract_first_frame(str(input_path), str(first_frame_path), use_last=True)

        # 2. Run get_depth
        depth_npy = run_get_depth(str(input_path), str(work_dir), utils_dir)

        # 3. Convert depth to MP4 segments
        depth_seg_dir = depth_npy_to_segments(
            depth_npy,
            str(work_dir),
            video_name,
            num_segments=num_segments,
        )

        # 4. Extract video segment for track (last 81 frames)
        rgb_segment_path = work_dir / "rgb_segment.mp4"
        extract_video_segment(str(input_path), str(rgb_segment_path))

        # 5. Run get_track for each segment (use same track for continuation - extrapolated motion)
        depth_00 = os.path.join(depth_seg_dir, "depth_00.mp4")
        track_00_path = work_dir / video_name / "track_00.mp4"
        track_00_path.parent.mkdir(parents=True, exist_ok=True)
        run_get_track(str(rgb_segment_path), depth_00, str(track_00_path), utils_dir)

        # For additional segments: reuse depth/track (extrapolation)
        for i in range(1, num_segments):
            depth_i = os.path.join(depth_seg_dir, f"depth_{i:02d}.mp4")
            track_i = work_dir / video_name / f"track_{i:02d}.mp4"
            if not os.path.exists(depth_i):
                shutil.copy(depth_00, depth_i)
            if not os.path.exists(track_i):
                shutil.copy(track_00_path, track_i)
    else:
        first_frame_path = work_dir / "first.png"
        if not first_frame_path.exists():
            sys.exit("Error: --skip_extraction requires existing first.png")

    # 6. Create cond.json
    cond_path = work_dir / "cond.json"
    cond_data = []
    for i in range(num_segments):
        depth_path = work_dir / video_name / f"depth_{i:02d}.mp4"
        track_path = work_dir / video_name / f"track_{i:02d}.mp4"
        cond_data.append({
            "text": prompt,
            "depth": str(depth_path.resolve()),
            "track": str(track_path.resolve()),
        })

    with open(cond_path, "w") as f:
        json.dump(cond_data, f, indent=2)
    print(f"[✓] Created {cond_path}")

    # 7. Run inference
    print("[*] Running LongVie inference...")
    infer_cmd = [
        sys.executable,
        "inference.py",
        "--json_file", str(cond_path),
        "--image_path", str(first_frame_path),
        "--video_name", f"{video_name}_extended",
        "--control_weight_path", args.control_weight_path,
        "--dit_weight_path", args.dit_weight_path,
        "--seed", str(args.seed),
    ]
    result = subprocess.run(infer_cmd, cwd=Path(__file__).resolve().parent)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print(f"\n[✓] Done! Extended video saved to ./gen_videos/{video_name}_extended/")


if __name__ == "__main__":
    main()
