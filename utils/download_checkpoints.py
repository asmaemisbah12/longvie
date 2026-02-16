#!/usr/bin/env python3
"""
Download checkpoints required for extend_Video.py (depth + track).
Run from LongVie root: python3 utils/download_checkpoints.py
Or from utils: python3 download_checkpoints.py
"""
import os
import sys

CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

def download_depth():
    """Download Video Depth Anything (vitl) from Hugging Face."""
    path = os.path.join(CHECKPOINTS_DIR, "video_depth_anything_vitl.pth")
    if os.path.isfile(path):
        print(f"[skip] Already exists: {path}")
        return path
    try:
        from huggingface_hub import hf_hub_download
        print("Downloading video_depth_anything_vitl.pth (~1.5 GB)...")
        downloaded = hf_hub_download(
            repo_id="depth-anything/Video-Depth-Anything-Large",
            filename="video_depth_anything_vitl.pth",
            local_dir=CHECKPOINTS_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"[ok] Saved to {downloaded}")
        return downloaded
    except Exception as e:
        print(f"[error] {e}")
        print("Manual download: https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth")
        print(f"Save to: {path}")
        return None

def main():
    print(f"Checkpoints directory: {CHECKPOINTS_DIR}\n")
    download_depth()
    spa_path = os.path.join(CHECKPOINTS_DIR, "spaT_final.pth")
    if not os.path.isfile(spa_path):
        print(f"\n[note] spaT_final.pth not found at {spa_path}")
        print("Track extraction (get_track) needs it. Download from SpaTracker Google Drive:")
        print("  https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ?usp=sharing")
        print(f"  then save spaT_final.pth to: {spa_path}")

if __name__ == "__main__":
    main()
