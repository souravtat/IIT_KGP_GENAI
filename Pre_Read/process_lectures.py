#!/usr/bin/env python3
"""
AI4ICPS Lecture Pipeline: Download -> Extract Audio -> Transcribe (whisper.cpp) -> Markdown Notes

Produces SRT (timestamped) transcripts and generates structured notes with time ranges
on every section/subsection heading.

Usage:
    python3 process_lectures.py                  # Process all lectures
    python3 process_lectures.py 0 5             # Process lectures 0-4 (by index)
    python3 process_lectures.py --clean          # Delete video/audio after transcription
    python3 process_lectures.py --model medium   # Use medium model (if downloaded)
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path("/Users/u0j006f/Projects/ai4icps-notes")
VIDEOS_DIR = BASE_DIR / "videos"
AUDIO_DIR = BASE_DIR / "audio"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
NOTES_DIR = BASE_DIR / "notes"
MODELS_DIR = BASE_DIR / "models"

WHISPER_CLI = "whisper-cli"
DEFAULT_MODEL = "small"

for d in [VIDEOS_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, NOTES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── SRT Parsing ───────────────────────────────────────────────────────────────

def parse_srt(srt_path):
    """Parse an SRT file into a list of (start_seconds, end_seconds, text) tuples."""
    segments = []
    with open(srt_path, "r") as f:
        content = f.read()

    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        time_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1]
        )
        if not time_match:
            continue
        g = time_match.groups()
        start_s = int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2]) + int(g[3]) / 1000
        end_s = int(g[4]) * 3600 + int(g[5]) * 60 + int(g[6]) + int(g[7]) / 1000
        text = " ".join(lines[2:]).strip()
        if text:
            segments.append((start_s, end_s, text))

    return segments


def format_timestamp(seconds):
    """Convert seconds to MM:SS or H:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def get_full_text_from_segments(segments):
    """Join all segment text into one string."""
    return " ".join(seg[2] for seg in segments)


# ─── Pipeline Functions ────────────────────────────────────────────────────────

def load_video_urls():
    with open(BASE_DIR / "video_urls.json") as f:
        return json.load(f)


def download_video(name, url, index, total):
    output_path = VIDEOS_DIR / f"{name}.mp4"
    if output_path.exists() and output_path.stat().st_size > 10000:
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  [{index}/{total}] Already downloaded: {name} ({size_mb:.0f} MB)")
        return output_path

    print(f"  [{index}/{total}] Downloading: {name}...")
    result = subprocess.run(
        ["curl", "-L", "-o", str(output_path), "-#", url],
        capture_output=True, text=True, timeout=900
    )
    if result.returncode != 0 or not output_path.exists():
        print(f"    ERROR downloading: {result.stderr[:200]}")
        return None

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    Downloaded: {size_mb:.1f} MB")
    return output_path


def extract_audio(name, video_path):
    audio_path = AUDIO_DIR / f"{name}.wav"
    if audio_path.exists() and audio_path.stat().st_size > 10000:
        print(f"    Audio exists: {name}")
        return audio_path

    print(f"    Extracting audio...")
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
         "-ar", "16000", "-ac", "1", str(audio_path), "-y", "-loglevel", "error"],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"    ERROR extracting audio: {result.stderr[:200]}")
        return None
    return audio_path


def transcribe_audio(name, audio_path, model_name):
    """Transcribe audio, producing both .srt (timestamped) and .txt (plain) files."""
    srt_path = TRANSCRIPTS_DIR / f"{name}.srt"
    txt_path = TRANSCRIPTS_DIR / f"{name}.txt"

    if srt_path.exists() and srt_path.stat().st_size > 100:
        print(f"    SRT transcript exists: {name}")
        return srt_path

    model_path = MODELS_DIR / f"ggml-{model_name}.bin"
    if not model_path.exists():
        print(f"    ERROR: Model not found: {model_path}")
        return None

    print(f"    Transcribing with whisper.cpp ({model_name} model, Metal GPU)...")
    start = time.time()

    output_stem = TRANSCRIPTS_DIR / name
    # Produce both SRT and TXT
    result = subprocess.run(
        [WHISPER_CLI, "-m", str(model_path), "-f", str(audio_path),
         "-osrt", "-otxt", "-of", str(output_stem),
         "-t", "8", "-l", "en"],
        capture_output=True, text=True, timeout=1800
    )

    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"    ERROR transcribing: {result.stderr[:300]}")
        return None

    if srt_path.exists():
        size_kb = srt_path.stat().st_size / 1024
        print(f"    Transcribed in {elapsed:.0f}s (SRT: {size_kb:.0f} KB)")
        return srt_path

    return None


def process_single(name, info, index, total, model_name, clean):
    url = info.get("download_url")
    if not url:
        print(f"  [{index}/{total}] SKIPPING {name}: no video URL")
        return False

    video_path = download_video(name, url, index, total)
    if not video_path:
        return False

    audio_path = extract_audio(name, video_path)
    if not audio_path:
        return False

    if clean and video_path.exists():
        video_path.unlink()
        print(f"    Cleaned: video deleted")

    srt_path = transcribe_audio(name, audio_path, model_name)
    if not srt_path:
        return False

    if clean and audio_path.exists():
        audio_path.unlink()
        print(f"    Cleaned: audio deleted")

    print(f"    Done: {name}")
    return True


def main():
    args = sys.argv[1:]
    clean = "--clean" in args
    if clean:
        args.remove("--clean")

    model_name = DEFAULT_MODEL
    if "--model" in args:
        idx = args.index("--model")
        model_name = args[idx + 1]
        args = args[:idx] + args[idx+2:]

    data = load_video_urls()
    total = len(data)

    start_idx = int(args[0]) if len(args) > 0 else 0
    end_idx = int(args[1]) if len(args) > 1 else total

    print(f"{'=' * 60}")
    print(f"AI4ICPS Lecture Pipeline (whisper.cpp + Metal)")
    print(f"Model: {model_name} | Clean: {clean} | Output: SRT + TXT")
    print(f"Processing lectures {start_idx+1} to {end_idx} of {total}")
    print(f"{'=' * 60}")

    success = 0
    failed = 0
    start_time = time.time()

    items = list(data.items())[start_idx:end_idx]
    for i, (name, info) in enumerate(items, start=start_idx + 1):
        print(f"\n{'─' * 50}")
        if process_single(name, info, i, total, model_name, clean):
            success += 1
        else:
            failed += 1

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"  Success: {success} | Failed: {failed}")
    print(f"  Transcripts: {TRANSCRIPTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
