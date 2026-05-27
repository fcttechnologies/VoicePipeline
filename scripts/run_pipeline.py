#!/usr/bin/env python3
"""Voice-training pipeline runner: acquire source audio, transcribe, extract speaker-isolated clips, and finetune F5-TTS.

Override default output locations with:
  VOICE_PIPELINE_OUTPUT_ROOT   absolute path for runs/, datasets/, voice-models/, drop-zone/
  VOICE_PIPELINE_STATE_DIR     absolute path for source-material.json, processed.json, run-log.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
AUTOMATION_ROOT = PIPELINE_ROOT  # legacy alias for skill/setup paths inside this script
OUTPUT_ROOT = Path(os.environ.get("VOICE_PIPELINE_OUTPUT_ROOT", str(PIPELINE_ROOT / "output"))).expanduser().resolve()
STATE_DIR = Path(os.environ.get("VOICE_PIPELINE_STATE_DIR", str(PIPELINE_ROOT / "state"))).expanduser().resolve()
SOURCE_STATE = STATE_DIR / "source-material.json"
PROCESSED_STATE = STATE_DIR / "processed.json"
RUN_LOG_STATE = STATE_DIR / "run-log.json"


@dataclass
class RunResult:
    status: str
    run_id: str
    source_id: str | None
    source_title: str | None
    work_dir: Path
    dataset_dir: Path | None
    clips_written: int
    training_status: str
    message: str


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp.replace(path)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "command failed"
        raise RuntimeError(f"{cmd[0]} failed: {detail}")
    return result


def run_cmd_logged(cmd: list[str], log_path: Path, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        process = subprocess.run(cmd, cwd=cwd, text=True, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed; see {log_path}")


def require_binary(name: str, explicit: str | None = None) -> str:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return str(path)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Missing required command: {name}")
    return found


def next_source(smoke: bool) -> dict[str, Any] | None:
    sources_state = load_json(SOURCE_STATE, {"sources": []})
    processed_state = load_json(PROCESSED_STATE, {"processed": {}})
    sources = sorted(sources_state.get("sources", []), key=lambda item: item.get("priority", 999))
    if smoke:
        return sources[0] if sources else None
    processed = processed_state.get("processed", {})
    for source in sources:
        if source.get("id") not in processed:
            return source
    return None


def ffmpeg_location_arg(ffmpeg_path: str) -> str:
    path = Path(ffmpeg_path)
    return str(path.parent if path.name == "ffmpeg" else path)


def download_audio(source: dict[str, Any], work_dir: Path, ffmpeg_path: str, yt_dlp_path: str, smoke: bool, smoke_seconds: int) -> Path:
    source_type = source.get("type", "youtube")

    if source_type == "local-file":
        url = source["url"]
        local_path = Path(url.replace("file://", "")) if url.startswith("file://") else Path(url)
        if not local_path.exists():
            raise RuntimeError(f"Local file not found: {local_path}. Place the audio file in the drop-zone and verify the URL in source-material.json.")
        dest = work_dir / "source.wav"
        if smoke:
            cmd = [
                ffmpeg_path, "-y",
                "-ss", "00:00:00", "-t", str(smoke_seconds),
                "-i", str(local_path),
                "-vn", "-c:a", "pcm_s16le",
                str(dest),
            ]
            run_cmd(cmd)
        else:
            import shutil as _shutil
            _shutil.copy2(local_path, dest)
        return dest

    output_template = str(work_dir / "source.%(ext)s")
    cmd = [
        yt_dlp_path,
        "--no-playlist",
        "--force-overwrites",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--ffmpeg-location",
        ffmpeg_location_arg(ffmpeg_path),
        "-o",
        output_template,
    ]
    if smoke:
        cmd.extend(["--download-sections", f"*00:00-{smoke_seconds:02d}"])
    cmd.append(source["url"])
    run_cmd(cmd, cwd=work_dir)
    matches = sorted(work_dir.glob("source*.wav"))
    if not matches:
        raise RuntimeError("yt-dlp completed but no WAV audio was produced")
    return matches[0]


def normalize_audio(input_audio: Path, output_audio: Path, ffmpeg_path: str) -> Path:
    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_audio),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-af",
        "loudnorm=I=-20:TP=-1.5:LRA=11",
        str(output_audio),
    ]
    run_cmd(cmd)
    return output_audio


def transcribe_with_timestamps(audio_path: Path, model_name: str) -> list[dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Python package faster-whisper is not installed in this environment") from exc

    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=False,
        word_timestamps=True,
    )

    output: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        text = " ".join((segment.text or "").strip().split())
        if not text:
            continue
        words = []
        for word in segment.words or []:
            word_text = " ".join((word.word or "").strip().split())
            if not word_text:
                continue
            words.append({"word": word_text, "start": word.start, "end": word.end})
        output.append(
            {
                "index": index,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "words": words,
            }
        )
    if not output:
        raise RuntimeError("Transcription produced no timestamped segments")
    return output


def valid_segment(segment: dict[str, Any], min_sec: float, max_sec: float) -> bool:
    duration = float(segment["end"]) - float(segment["start"])
    text = segment.get("text", "").strip()
    if duration < min_sec or duration > max_sec:
        return False
    if len(text.split()) < 2:
        return False
    return True


def extract_dataset_clips(
    source: dict[str, Any],
    normalized_audio: Path,
    transcript_segments: list[dict[str, Any]],
    dataset_dir: Path,
    work_dir: Path,
    ffmpeg_path: str,
    max_segments: int | None,
) -> tuple[int, list[dict[str, Any]]]:
    wav_dir = dataset_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = dataset_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    clip_records: list[dict[str, Any]] = []
    metadata_lines: list[str] = []
    eligible = [segment for segment in transcript_segments if valid_segment(segment, 0.75, 15.0)]
    if max_segments is not None:
        eligible = eligible[:max_segments]

    source_id = source["id"].replace("/", "-")
    for ordinal, segment in enumerate(eligible, start=1):
        clip_name = f"{source_id}_{ordinal:04d}.wav"
        clip_path = wav_dir / clip_name
        start = max(0.0, float(segment["start"]) - 0.08)
        end = float(segment["end"]) + 0.08
        cmd = [
            ffmpeg_path,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(normalized_audio),
            "-ac",
            "1",
            "-ar",
            "24000",
            "-af",
            "loudnorm=I=-20:TP=-1.5:LRA=11",
            str(clip_path),
        ]
        run_cmd(cmd)
        text = segment["text"].replace("|", " ").strip()
        text_path = clip_path.with_suffix(".txt")
        text_path.write_text(text + "\n", encoding="utf-8")
        metadata_lines.append(f"{clip_path}|{text}")
        clip_records.append(
            {
                "clip": str(clip_path),
                "text": text,
                "start": start,
                "end": end,
                "source_id": source["id"],
            }
        )

    if not clip_records:
        raise RuntimeError("No valid speech segments were eligible for dataset extraction")

    metadata_path = dataset_dir / "metadata.csv"
    mode = "a" if metadata_path.exists() else "w"
    with metadata_path.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write("audio_file|text\n")
        for line in metadata_lines:
            handle.write(line + "\n")

    relative_metadata_path = dataset_dir / "metadata-relative.csv"
    relative_mode = "a" if relative_metadata_path.exists() else "w"
    with relative_metadata_path.open(relative_mode, encoding="utf-8") as handle:
        if relative_mode == "w":
            handle.write("audio_file|text\n")
        for record in clip_records:
            handle.write(f"wavs/{Path(record['clip']).name}|{record['text']}\n")

    save_json(work_dir / "transcript-segments.json", transcript_segments)
    save_json(manifests_dir / f"{source_id}-{now_iso().replace(':', '').replace('+', '_')}.json", clip_records)
    return len(clip_records), clip_records


def write_voice_profile(dataset_dir: Path, clip_records: list[dict[str, Any]]) -> Path:
    best = max(clip_records, key=lambda record: float(record["end"]) - float(record["start"]))
    profile = {
        "schema_version": 1,
        "tool": "f5-tts",
        "purpose": "Reference prompt for local F5-TTS inference and a seed asset for finetuning.",
        "ref_audio": best["clip"],
        "ref_text": best["text"],
        "dataset_dir": str(dataset_dir),
        "created_at": now_iso(),
    }
    profile_path = dataset_dir / "voice-profile.json"
    save_json(profile_path, profile)
    return profile_path


def f5_paths() -> dict[str, Path]:
    f5_bin = AUTOMATION_ROOT / ".venv-f5" / "bin"
    f5_python = f5_bin / "python"
    f5_cli = f5_bin / "f5-tts_finetune-cli"
    prepare_script = (
        AUTOMATION_ROOT
        / ".venv-f5"
        / "lib/python3.11/site-packages/f5_tts/train/datasets/prepare_csv_wavs.py"
    )
    example_vocab = (
        AUTOMATION_ROOT
        / ".venv-f5"
        / "lib/python3.11/site-packages/f5_tts/infer/examples/vocab.txt"
    )
    data_root = AUTOMATION_ROOT / ".venv-f5" / "lib/python3.11/data"
    ckpt_root = AUTOMATION_ROOT / ".venv-f5" / "lib/python3.11/ckpts"
    return {
        "python": f5_python,
        "cli": f5_cli,
        "prepare_script": prepare_script,
        "example_vocab": example_vocab,
        "data_root": data_root,
        "ckpt_root": ckpt_root,
    }


def training_step(dataset_dir: Path, model_dir: Path, skip_training: bool) -> str:
    model_dir.mkdir(parents=True, exist_ok=True)
    if skip_training:
        save_json(
            model_dir / "training-skipped.json",
            {
                "schema_version": 1,
                "status": "skipped",
                "reason": "Smoke test or explicit skip requested. Dataset and voice profile were prepared; full F5-TTS finetuning was not run.",
                "dataset_dir": str(dataset_dir),
                "created_at": now_iso(),
            },
        )
        return "skipped"

    paths = f5_paths()
    missing = [name for name, path in paths.items() if name in {"python", "cli", "prepare_script", "example_vocab"} and not path.exists()]
    if missing:
        raise RuntimeError(f"F5-TTS environment is incomplete; missing {', '.join(missing)}. See setup.md.")

    dataset_name = f"voice_{dataset_dir.name.replace('-', '_')}"
    prepared_dir = paths["data_root"] / f"{dataset_name}_pinyin"
    pretrained_vocab = paths["data_root"] / "Emilia_ZH_EN_pinyin" / "vocab.txt"
    if not pretrained_vocab.exists():
        pretrained_vocab.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths["example_vocab"], pretrained_vocab)
    training_log = model_dir / "training.log"
    run_cmd_logged(
        [
            str(paths["python"]),
            str(paths["prepare_script"]),
            str(dataset_dir / "metadata.csv"),
            str(prepared_dir),
            "--workers",
            "2",
        ],
        training_log,
    )
    run_cmd_logged(
        [
            str(paths["cli"]),
            "--exp_name",
            "F5TTS_v1_Base",
            "--dataset_name",
            dataset_name,
            "--finetune",
            "--learning_rate",
            "0.00001",
            "--batch_size_per_gpu",
            "512",
            "--batch_size_type",
            "frame",
            "--max_samples",
            "8",
            "--grad_accumulation_steps",
            "1",
            "--max_grad_norm",
            "1.0",
            "--epochs",
            "1",
            "--num_warmup_updates",
            "10",
            "--save_per_updates",
            "100",
            "--last_per_updates",
            "20",
            "--keep_last_n_checkpoints",
            "1",
        ],
        training_log,
    )

    ckpt_dir = paths["ckpt_root"] / dataset_name
    if ckpt_dir.exists():
        target = model_dir / "checkpoints"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(ckpt_dir, target)
    save_json(
        model_dir / "training.json",
        {
            "schema_version": 1,
            "status": "complete",
            "tool": "f5-tts",
            "dataset_name": dataset_name,
            "dataset_dir": str(dataset_dir),
            "prepared_dir": str(prepared_dir),
            "checkpoint_dir": str(model_dir / "checkpoints"),
            "training_log": str(training_log),
            "created_at": now_iso(),
        },
    )
    return "complete"


def append_run_log(result: RunResult) -> None:
    state = load_json(RUN_LOG_STATE, {"schema_version": 1, "runs": []})
    state.setdefault("runs", []).append(
        {
            "run_id": result.run_id,
            "created_at": now_iso(),
            "status": result.status,
            "source_id": result.source_id,
            "source_title": result.source_title,
            "work_dir": str(result.work_dir),
            "dataset_dir": str(result.dataset_dir) if result.dataset_dir else None,
            "clips_written": result.clips_written,
            "training_status": result.training_status,
            "message": result.message,
        }
    )
    save_json(RUN_LOG_STATE, state)


def mark_processed(source: dict[str, Any], dataset_dir: Path, clips_written: int) -> None:
    state = load_json(PROCESSED_STATE, {"schema_version": 1, "dataset_version": 1, "processed": {}})
    state.setdefault("processed", {})[source["id"]] = {
        "title": source.get("title"),
        "url": source.get("url"),
        "processed_at": now_iso(),
        "dataset_dir": str(dataset_dir),
        "clips_written": clips_written,
    }
    save_json(PROCESSED_STATE, state)


def send_telegram(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    subprocess.run(
        [
            "curl",
            "-s",
            "-X",
            "POST",
            f"https://api.telegram.org/bot{token}/sendMessage",
            "-d",
            f"chat_id={chat_id}",
            "-d",
            f"text={text}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def run_pipeline(args: argparse.Namespace) -> RunResult:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = OUTPUT_ROOT / "runs" / ("smoke-" + run_id if args.smoke_test else ("train-" + run_id if args.train_only else run_id))
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.train_only:
        processed_state = load_json(PROCESSED_STATE, {"dataset_version": 1})
        dataset_name = f"v{int(processed_state.get('dataset_version', 1)):03d}"
        dataset_dir = OUTPUT_ROOT / "datasets" / dataset_name
        model_dir = OUTPUT_ROOT / "voice-models" / dataset_name
        if not (dataset_dir / "metadata.csv").exists():
            result = RunResult("blocked", run_id, None, None, work_dir, dataset_dir, 0, "failed", f"No dataset at {dataset_dir}. Build it before --train-only.")
            append_run_log(result)
            return result
        try:
            training_status = training_step(dataset_dir, model_dir, skip_training=False)
            with open(dataset_dir / "metadata.csv") as fh:
                clip_count = sum(1 for _ in fh) - 1
            message = f"Trained on existing dataset {dataset_dir} ({clip_count} clips). Model at {model_dir}."
            result = RunResult("done", run_id, None, None, work_dir, dataset_dir, clip_count, training_status, message)
            append_run_log(result)
            return result
        except Exception as exc:
            result = RunResult("blocked", run_id, None, None, work_dir, dataset_dir, 0, "failed", str(exc))
            append_run_log(result)
            return result

    ffmpeg_path = require_binary("ffmpeg", args.ffmpeg)
    require_binary("ffprobe", args.ffprobe)
    yt_dlp_path = require_binary("yt-dlp", args.yt_dlp)

    source = next_source(args.smoke_test)
    if not source:
        result = RunResult("done", run_id, None, None, work_dir, None, 0, "not_started", "No unprocessed source material remains.")
        append_run_log(result)
        return result

    processed_state = load_json(PROCESSED_STATE, {"dataset_version": 1})
    dataset_name = "smoke" if args.smoke_test else f"v{int(processed_state.get('dataset_version', 1)):03d}"
    dataset_dir = OUTPUT_ROOT / "datasets" / dataset_name
    model_dir = OUTPUT_ROOT / "voice-models" / dataset_name

    try:
        raw_audio = download_audio(source, work_dir, ffmpeg_path, yt_dlp_path, args.smoke_test, args.smoke_seconds)
        normalized_audio = normalize_audio(raw_audio, work_dir / "normalized.wav", ffmpeg_path)
        segments = transcribe_with_timestamps(normalized_audio, args.whisper_model)
        save_json(work_dir / "source.json", source)
        clips_written, clip_records = extract_dataset_clips(
            source,
            normalized_audio,
            segments,
            dataset_dir,
            work_dir,
            ffmpeg_path,
            args.max_segments if args.max_segments else (1 if args.smoke_test else None),
        )
        profile_path = write_voice_profile(dataset_dir, clip_records)
        training_status = training_step(dataset_dir, model_dir, args.skip_training or args.smoke_test)
        if not args.smoke_test:
            mark_processed(source, dataset_dir, clips_written)
        message = f"Prepared {clips_written} clip(s), dataset {dataset_dir}, profile {profile_path}, training {training_status}."
        result = RunResult("done", run_id, source["id"], source.get("title"), work_dir, dataset_dir, clips_written, training_status, message)
        append_run_log(result)
        return result
    except Exception as exc:
        result = RunResult("blocked", run_id, source.get("id"), source.get("title"), work_dir, dataset_dir, 0, "failed", str(exc))
        append_run_log(result)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Jarvis voice-training automation.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny sample and skip full training.")
    parser.add_argument("--skip-training", action="store_true", help="Build dataset only; do not invoke the training step.")
    parser.add_argument("--train-only", action="store_true", help="Train F5-TTS on the existing v{dataset_version} dataset without acquiring new source audio.")
    parser.add_argument("--smoke-seconds", type=int, default=20, help="Seconds to download during smoke tests.")
    parser.add_argument("--max-segments", type=int, default=0, help="Maximum transcript segments to extract.")
    parser.add_argument("--whisper-model", default=os.environ.get("VOICE_PIPELINE_WHISPER_MODEL", "base.en"))
    parser.add_argument("--ffmpeg", default=os.environ.get("VOICE_PIPELINE_FFMPEG", "/opt/homebrew/bin/ffmpeg"))
    parser.add_argument("--ffprobe", default=os.environ.get("VOICE_PIPELINE_FFPROBE", "/opt/homebrew/bin/ffprobe"))
    parser.add_argument("--yt-dlp", default=os.environ.get("VOICE_PIPELINE_YT_DLP", "yt-dlp"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(args)
    digest = (
        f"Jarvis Voice Training - {result.status}\n"
        f"Run: {result.run_id}\n"
        f"Source: {result.source_title or 'none'}\n"
        f"Clips: {result.clips_written}\n"
        f"Training: {result.training_status}\n"
        f"Output: {result.work_dir}\n"
        f"{result.message}"
    )
    print(digest)
    send_telegram(digest)
    return 0 if result.status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
