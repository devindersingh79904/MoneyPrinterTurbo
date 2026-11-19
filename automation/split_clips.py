"""Split raw gameplay recordings into ~10 second clips for MoneyPrinterTurbo's local video source.

Usage:  python automation/split_clips.py [--seconds 10]

Reads every .mp4 in ../raw_recordings/ and writes <name>_000.mp4, <name>_001.mp4, ... to
../local_clips/. Stream copy (-c copy) is fast and lossless, but cuts land on keyframes, so a
segment can be a little longer or shorter than --seconds.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_recordings"
CLIPS = ROOT / "local_clips"


def split(video, seconds):
    pattern = CLIPS / f"{video.stem}_%03d.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
           "-map", "0:v:0", "-map", "0:a?", "-c", "copy", "-f", "segment",
           "-segment_time", str(seconds), "-reset_timestamps", "1", str(pattern)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        print(f"{video.name}: ffmpeg failed: {result.stderr.strip()[-300:]}")
        return 0
    count = len(list(CLIPS.glob(f"{video.stem}_[0-9][0-9][0-9].mp4")))
    print(f"{video.name}: {count} clip(s)")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=int, default=10)
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not on PATH.")
    RAW.mkdir(exist_ok=True)
    CLIPS.mkdir(exist_ok=True)
    videos = sorted(RAW.glob("*.mp4"))
    if not videos:
        sys.exit(f"No .mp4 files in {RAW}")
    total = sum(split(video, args.seconds) for video in videos)
    print(f"done: {total} clip(s) in {CLIPS}")


if __name__ == "__main__":
    main()
