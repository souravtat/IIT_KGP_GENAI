# Video Scraping and Transcribing

Automated pipeline to scrape lecture videos from Moodle LMS, transcribe them using **whisper.cpp** (with Apple Silicon Metal GPU acceleration), and generate structured Markdown notes.

## Features

- **Video Extraction**: Scrapes video URLs from authenticated Moodle pages
- **Batch Download**: Downloads MP4 lectures from CloudFront CDN
- **Fast Transcription**: Uses whisper.cpp with Metal GPU (~60x realtime on M4 Pro)
- **Structured Notes**: Converts raw transcripts into organized Markdown with headings, tables, diagrams, and key takeaways

## Pipeline

```
Moodle Page → Extract Video URLs → Download MP4 → Extract Audio (ffmpeg)
    → Transcribe (whisper.cpp + Metal) → Generate Markdown Notes
```

## Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Homebrew

### Installation

```bash
# Install whisper.cpp (with Metal GPU support)
brew install whisper-cpp

# Install ffmpeg
brew install ffmpeg

# Install Python dependencies
pip3 install requests yt-dlp

# Download Whisper model (small, ~466MB)
mkdir -p models
curl -L -o models/ggml-small.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
```

## Usage

### 1. Extract video URLs from Moodle

```bash
python3 extract_videos.py
```

This produces `video_urls.json` with all lecture video URLs.

### 2. Run the full pipeline

```bash
# Process all lectures
python3 process_lectures.py

# Process specific range (e.g., lectures 1-5)
python3 process_lectures.py 0 5

# Process with disk cleanup (delete video/audio after transcription)
python3 process_lectures.py --clean

# Use a different model size
python3 process_lectures.py --model medium
```

### Output Structure

```
ai4icps-notes/
├── video_urls.json          # Extracted video URLs
├── extract_videos.py        # URL extraction script
├── process_lectures.py      # Main pipeline script
├── models/                  # Whisper GGML models
│   └── ggml-small.bin
├── transcripts/             # Raw transcription output
│   ├── 01_Introduction_to_AI.txt
│   └── ...
└── notes/                   # Structured Markdown notes
    ├── 01_Introduction_to_AI.md
    └── ...
```

## Performance

| Step | Time (per lecture) | Hardware |
|------|-------------------|----------|
| Download | ~30-60s | Depends on network |
| Audio extraction | ~5-10s | ffmpeg |
| Transcription | ~60-90s | whisper.cpp + M4 Pro Metal |
| Notes generation | <1s | Python |

**Total**: ~43 lectures processed in ~1.5-2 hours

## Models

| Model | Size | Accuracy | Speed |
|-------|------|----------|-------|
| base | 142 MB | Good for clear audio | Fastest |
| small | 466 MB | Good balance (default) | Fast |
| medium | 1.5 GB | Best for accented speech | Moderate |

## Notes Generation Prompt

The structured Markdown notes were generated using a detailed prompt that instructs the AI to act as an AI/ML Subject Matter Expert. The full prompt is available in [`PROMPT.md`](PROMPT.md).

Key aspects of the prompt:
- Timestamps mapped from SRT segments to section headings
- Every formula rendered in LaTeX + Python pseudocode + plain English
- Every technical term gets a "Jargon" callout with a software-engineer-friendly explanation
- Mermaid diagrams for pipelines, architectures, and concept relationships
- Comparison tables for side-by-side analysis
- Glossary and concept map at the end of each file
- AI expert corrections for transcription errors and missing context

## Course Content

This pipeline was built for the **Gen AI Course for Walmart** by IIT Kharagpur AI4ICPS, covering:

1. Introduction to AI
2. Mathematical Foundations for AI/ML
3. Python for ML (Hands-on)
4. Linear Models (Regression & Classification)
5. Supervised ML (Bayes, KNN, SVM, Decision Trees, Random Forest)
6. Unsupervised ML (K-Means, PCA)
7. Neural Networks & Deep Learning (Feed-Forward, CNN, RNN)
8. Transfer Learning
9. Computer Vision with DL
10. NLP (Theory & Hands-on)
11. Advanced DL (Transformers, BERT)
12. Prompting with Open-Source LLMs (Llama2, Mistral)
13. Foundations of Generative AI
14. LLM Finetuning & Agentic Workflows
15. GANs & VAEs

## License

MIT
