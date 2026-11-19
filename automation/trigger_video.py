"""Ask the local MoneyPrinterTurbo API to make one 9:16 video from local clips, then wait for it.

Usage:
  python automation/trigger_video.py
  python automation/trigger_video.py --subject "Best GTA 6 police escapes" --script "optional narration"

Clips in ../local_clips/ are copied into storage/local_videos/ (the only folder the app reads
local materials from) and passed as video_materials. Prints the output MP4 path(s).
"""
import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ROOT / "local_clips"
LOCAL_VIDEOS = ROOT / "storage" / "local_videos"
API = "http://127.0.0.1:8080"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(API + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode())


def stage_clips(limit):
    LOCAL_VIDEOS.mkdir(parents=True, exist_ok=True)
    names = []
    for clip in sorted(CLIPS.glob("*.mp4"))[:limit]:
        target = LOCAL_VIDEOS / clip.name
        if not target.exists() or target.stat().st_size != clip.stat().st_size:
            shutil.copy2(clip, target)
        names.append(clip.name)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", default="Top 3 craziest GTA moments")
    parser.add_argument("--script", default="", help="narration; empty lets the local LLM write it")
    parser.add_argument("--voice", default="en-US-GuyNeural-Male")
    parser.add_argument("--max-clips", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds to wait for the task")
    args = parser.parse_args()

    names = stage_clips(args.max_clips)
    if not names:
        sys.exit(f"No clips in {CLIPS}. Put recordings in raw_recordings/ and run split_clips.py first.")
    body = {
        "video_subject": args.subject,
        "video_script": args.script,
        "video_aspect": "9:16",
        "video_source": "local",
        "video_materials": [{"provider": "local", "url": name, "duration": 0} for name in names],
        "video_clip_duration": 5,
        "video_count": 1,
        "video_language": "en-US",
        "voice_name": args.voice,
        "subtitle_enabled": True,
    }
    try:
        created = call("POST", "/api/v1/videos", body)
    except urllib.error.HTTPError as error:
        sys.exit(f"API returned HTTP {error.code}: {error.read().decode()[:500]}")
    except urllib.error.URLError as error:
        sys.exit(f"Cannot reach {API} ({error.reason}). Start the API first (automation/start_services.ps1).")
    task_id = (created.get("data") or {}).get("task_id")
    if not task_id:
        sys.exit(f"No task_id in response: {created}")
    print(f"task {task_id} started with {len(names)} local clip(s)")

    deadline = time.time() + args.timeout
    last = None
    while time.time() < deadline:
        task = call("GET", f"/api/v1/tasks/{task_id}").get("data") or {}
        state, progress = task.get("state"), task.get("progress")
        if (state, progress) != last:
            print(f"  state {state}  progress {progress}")
            last = (state, progress)
        if state == 1:
            videos = task.get("videos") or task.get("combined_videos") or []
            for video in videos:
                print("output:", video)
            return
        if state == -1:
            sys.exit(f"task failed: {task.get('error') or task.get('message') or json.dumps(task)[:500]}")
        time.sleep(5)
    sys.exit(f"timed out after {args.timeout}s; check GET {API}/api/v1/tasks/{task_id}")


if __name__ == "__main__":
    main()
