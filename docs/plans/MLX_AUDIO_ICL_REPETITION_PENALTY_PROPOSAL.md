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
  across seeds. Results for 1.5 vs 1.2 vs 1.1 on Spanish clones will be added
  here once measured.
