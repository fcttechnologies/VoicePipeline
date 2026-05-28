#!/usr/bin/env python3
"""Synthesize speech with F5-TTS — generic CLI wrapper.

Renders one piece of text in a target voice using F5-TTS finetune inference.
Reads (text + reference audio + reference text + checkpoint) and writes a WAV.

Usage:
    synthesize.py \\
        --text "Good morning, sir." \\
        --ref-audio /path/to/ref.wav \\
        --ref-text "Reference clip text" \\
        --output /tmp/out.wav \\
        [--ckpt /path/to/model_last.pt] \\
        [--model F5TTS_v1_Base] \\
        [--vocab /path/to/vocab.txt] \\
        [--speed 1.0] \\
        [--remove-silence] \\
        [--f5-bin /path/to/f5-tts_infer-cli]

If --ckpt is omitted, F5-TTS uses the bundled pretrained checkpoint for the
selected model. Pass --ckpt to use a finetuned model.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_f5_bin(explicit: str | None) -> str:
    if explicit:
        if Path(explicit).exists():
            return explicit
        sys.exit(f"--f5-bin path does not exist: {explicit}")
    # Search candidates: sibling venv in VoicePipeline, then PATH.
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / ".venv" / "bin" / "f5-tts_infer-cli",
        repo_root / ".venv-f5" / "bin" / "f5-tts_infer-cli",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    on_path = shutil.which("f5-tts_infer-cli")
    if on_path:
        return on_path
    sys.exit(
        "f5-tts_infer-cli not found. Install F5-TTS in this repo's .venv "
        "(`python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`) "
        "or pass --f5-bin."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize speech with F5-TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--ref-audio", required=True, help="Reference audio WAV path.")
    parser.add_argument("--ref-text", required=True, help="Transcript of the reference audio.")
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument("--ckpt", help="Custom finetune checkpoint .pt (optional).")
    parser.add_argument("--model", default="F5TTS_v1_Base", help="F5-TTS model name (default F5TTS_v1_Base).")
    parser.add_argument("--vocab", help="Custom vocab .txt (optional; defaults to F5-TTS bundled vocab).")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier (default 1.0).")
    parser.add_argument("--remove-silence", action="store_true", help="Strip long silences from the output.")
    parser.add_argument("--f5-bin", help="Path to f5-tts_infer-cli (auto-detected if omitted).")
    parser.add_argument("--verbose", action="store_true", help="Print the underlying f5-tts_infer-cli command.")
    args = parser.parse_args()

    f5_bin = find_f5_bin(args.f5_bin)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        f5_bin,
        "-m", args.model,
        "-r", args.ref_audio,
        "-s", args.ref_text,
        "-t", args.text,
        "-o", str(output_path.parent),
        "-w", output_path.name,
        "--speed", str(args.speed),
    ]
    if args.ckpt:
        cmd.extend(["-p", args.ckpt])
    if args.vocab:
        cmd.extend(["-v", args.vocab])
    if args.remove_silence:
        cmd.append("--remove_silence")

    if args.verbose:
        print(f"$ {' '.join(cmd)}", file=sys.stderr)

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        return result.returncode

    if not output_path.exists():
        # F5-TTS sometimes adds a suffix to the output filename. Find the produced file.
        produced = sorted(output_path.parent.glob(f"{output_path.stem}*{output_path.suffix}"))
        if produced and produced[0] != output_path:
            produced[0].rename(output_path)
        if not output_path.exists():
            print(f"warning: expected output at {output_path} but it was not produced", file=sys.stderr)
            return 2

    print(str(output_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
