# Proposal for mlx-audio: make the ICL repetition penalty configurable

Draft for an issue/PR against `Blaizzy/mlx-audio` (Qwen3-TTS model). Not posted yet.

## Problem

`mlx_audio/tts/models/qwen3_tts/qwen3_tts.py`, in `generate()`, routes voice
cloning with a reference clip and transcript to `_generate_icl()` and forces
the repetition penalty:

```python
# ICL mode needs stronger repetition penalty to prevent code
# degeneration with long reference audio prefills
icl_rep_penalty = max(repetition_penalty, 1.5)
```

`_sample_token()` applies that penalty to the set of every codebook-0 token
generated so far in the segment, with no window. Two consequences for
callers that clone from short, clean references:

1. The public `repetition_penalty` argument is silently ignored below 1.5 in
   the one mode where most users are (cloning). The official `qwen_tts`
   implementation defaults to 1.05 for the same checkpoint.
2. On long segments (500+ characters, ~800-1000 codec steps) an ever-growing
   share of the 2048-code vocabulary is penalised, which is a plausible,
   code-grounded mechanism for the end-of-segment speed and prosody drift that
   users report with Qwen3-TTS clones.

The floor was added to prevent degeneration with *long* reference prefills.
Callers that keep references to 10-15 s do not need it, but cannot opt out.

## Proposal

Add an explicit knob and keep the current default:

```python
def generate(
    self,
    text: str,
    ...
    repetition_penalty: float = 1.05,
    icl_repetition_penalty: float | None = None,
    **kwargs,
):
    ...
    if use_icl:
        # Default keeps today's behaviour; callers with short references can lower it.
        penalty = icl_repetition_penalty if icl_repetition_penalty is not None else max(repetition_penalty, 1.5)
        yield from self._generate_icl(..., repetition_penalty=penalty, ...)
```

Optionally document the trade-off in the docstring: lower values give more
natural prosody on short references; values below ~1.2 with references longer
than ~20 s can loop.

## Evidence we can attach

- Voicebox (fork `CarlosGamezF/voicebox`) has an experiment flag,
  `VOICEBOX_MLX_ICL_REPETITION_PENALTY`, that calls `_generate_icl` directly,
  plus an evaluation harness (`backend/tools/prosody_eval.py`) that measures
  WER against Whisper large, speaking rate and its drift, pauses and clipping
  across seeds.
- Measured on 2026-09-14 (M5 Pro, mlx-audio 0.4.1, `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`,
  one Spanish speaker, 28 s reference, 4 texts of 137-551 characters, seeds
  1-3, chunks of 550 characters, Whisper large for WER after normalising
  accents, punctuation and abbreviations):

  | `repetition_penalty` | takes | WER | chars/s (min-max) | pauses > 300 ms | longest pause | clipping > 0.85 |
  |---|---|---|---|---|---|---|
  | 1.5 (current floor) | 12 | 0.063 | 15.2 (13.3-17.7) | 2.75 | 0.67 s | 0.033 % |
  | 1.2 | 12 | 0.064 | 15.1 (13.0-17.9) | 2.67 | 0.68 s | 0.031 % |
  | 1.1 | 12 | 0.068 | 15.1 (12.7-18.0) | 2.83 | 0.66 s | 0.020 % |

  No take degenerated at 1.1 or 1.2: longest output 32.0-32.6 s in every
  condition, no repeated phrases in the Whisper transcripts. On this setup
  the 1.5 floor is not needed for stability, which is the reason the code
  gives for clamping; whether a lower value sounds more natural is a
  listening question (blind A/B pairs were produced, not yet rated). That is
  the case for exposing the value instead of clamping it: callers who see
  loops can keep 1.5, callers who do not can lower it.
