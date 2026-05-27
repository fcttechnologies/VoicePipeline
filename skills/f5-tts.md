# F5-TTS

Primary local TTS and finetuning tool used by VoicePipeline.

## Install

```bash
uv venv --python 3.11.15 .venv-f5
uv pip install --python .venv-f5/bin/python f5-tts
```

Verify:

```bash
.venv-f5/bin/f5-tts_infer-cli --help
ls .venv-f5/bin | grep f5-tts
```

Expected commands:

- `f5-tts_infer-cli`
- `f5-tts_infer-gradio`
- `f5-tts_finetune-cli`
- `f5-tts_finetune-gradio`

## Dataset Format

Raw automation dataset:

```text
datasets/v001/
├── metadata.csv
└── wavs/
```

`metadata.csv`:

```text
audio_file|text
/absolute/path/to/wavs/source_0001.wav|Spoken transcript.
```

F5-TTS preparation converts this into:

```text
.venv-f5/lib/python3.11/data/voice_v001_pinyin/
├── raw.arrow
├── duration.json
└── vocab.txt
```

## Finetune Command

The runner invokes:

```bash
.venv-f5/bin/f5-tts_finetune-cli \
  --exp_name F5TTS_v1_Base \
  --dataset_name voice_v001 \
  --finetune \
  --learning_rate 0.00001 \
  --batch_size_per_gpu 512 \
  --batch_size_type frame \
  --max_samples 8 \
  --grad_accumulation_steps 1 \
  --max_grad_norm 1.0 \
  --epochs 1 \
  --num_warmup_updates 10 \
  --save_per_updates 100 \
  --last_per_updates 20 \
  --keep_last_n_checkpoints 1
```

These settings are conservative for the Mac mini. Increase them only after a successful first real finetune.

## Inference

Before finetuning is good enough, use the generated `voice-profile.json` for zero-shot inference:

```bash
.venv-f5/bin/f5-tts_infer-cli \
  --model F5TTS_v1_Base \
  --ref_audio "/path/to/ref.wav" \
  --ref_text "Reference transcript." \
  --gen_text "Hello, world."
```

## Failure Handling

- If `f5-tts_finetune-cli` is missing, run setup again.
- If base-weight download fails, rerun with working network.
- If training is too slow or unstable on local CPU, keep the dataset and test XTTS-v2 as the fallback local engine.
