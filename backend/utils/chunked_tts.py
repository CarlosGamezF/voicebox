"""
Chunked TTS generation utilities.

Splits long text into sentence-boundary chunks, generates audio per-chunk
via any TTSBackend, and concatenates with crossfade.  All logic is
engine-agnostic — it wraps the standard ``TTSBackend.generate()`` interface.

Short text (≤ max_chunk_chars) uses the single-shot fast path with zero
overhead.
"""

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .audio import save_audio

logger = logging.getLogger("voicebox.chunked-tts")

# Default chunk size in characters.  Can be overridden per-request via
# the ``max_chunk_chars`` field on GenerationRequest.
DEFAULT_MAX_CHUNK_CHARS = 800
MAX_RUNAWAY_RETRIES = 2
MIN_RUNAWAY_RETRY_CHARS = 100
# Chunks shorter than this are merged into a neighbour when the result still
# fits: Qwen renders two- or three-word tail chunks with odd prosody.
MIN_CHUNK_CHARS = 40

# Common abbreviations that should NOT be treated as sentence endings.
# Lowercase for case-insensitive matching.
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "ave",
        "blvd",
        "inc",
        "ltd",
        "corp",
        "dept",
        "est",
        "approx",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "u.s",
        "u.s.a",
        "u.k",
        # Spanish
        "sra",
        "srta",
        "dra",
        "dres",
        "dña",
        "ud",
        "uds",
        "vd",
        "vds",
        "avda",
        "dpto",
        "aprox",
        "ej",
        # Doubled-letter plurals: EE. UU., FF. CC.
        "ee",
        "uu",
        "ff",
        "cc",
    }
)

# Abbreviations that are also ordinary words ("a work of art.") and only
# abbreviate when a number or Roman numeral follows: art. 5, cap. 3, vol. IV.
_NUMBER_ABBREVIATIONS = frozenset({"art", "cap", "fig", "vol", "pág", "págs", "núm", "nº", "tel", "pp"})
_NUMBER_AFTER_RE = re.compile(r"\s*(\d|[IVXLCDM]+\b)")
_INITIAL_AFTER_RE = re.compile(r"\s*[A-ZÁÉÍÓÚÑ]\.")

# A sentence end: terminal punctuation (runs like "..." included), then any
# closing quotes or brackets that belong to the sentence, then whitespace
# or the end of the text.
_SENTENCE_END_RE = re.compile(r"[.!?\u2026]+([\"\u00bb\u201d\u2019')\]]*)(?=\s|$)")

# Paralinguistic tags used by Chatterbox Turbo.  The splitter must never
# cut inside one of these.
_PARA_TAG_RE = re.compile(r"\[[^\]]*\]")


def split_text_into_chunks(
    text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS, merge_tiny: bool = True
) -> list[str]:
    """Split *text* at natural boundaries into chunks of at most *max_chars*.

    Paragraph breaks (a blank line) are chunk boundaries and single newlines
    inside a paragraph become spaces, so the model never receives a raw
    newline. Within a paragraph the priority is sentence-end (``.!?…`` plus
    any closing quote, not after an abbreviation, a list marker or an
    initial, and not inside brackets) → clause boundary (``;:,—``) →
    whitespace → hard cut. With *merge_tiny*, chunks shorter than
    ``MIN_CHUNK_CHARS`` are folded into a neighbour when the result still
    fits, across paragraphs too (a one-word line of dialogue is not a take).

    Paralinguistic tags like ``[laugh]`` are treated as atomic and will not
    be split across chunks.
    """
    chunks: list[str] = []
    for paragraph in _split_paragraphs(text):
        chunks.extend(_split_paragraph(paragraph, max_chars))
    return _merge_tiny_chunks(chunks, max_chars) if merge_tiny else chunks


def _split_paragraphs(text: str) -> list[str]:
    """Blank-line separated paragraphs, each with inner newlines collapsed."""
    paragraphs = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return [re.sub(r"\s*\n\s*", " ", p).strip() for p in paragraphs if p.strip()]


def _split_paragraph(text: str, max_chars: int) -> list[str]:
    """Cut one paragraph into chunks of at most *max_chars* at the best boundary in each window."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        segment = remaining[:max_chars]

        # Try to split at the last real sentence ending
        split_pos = _find_last_sentence_end(segment)
        if split_pos == -1:
            split_pos = _find_last_clause_boundary(segment)
        if split_pos == -1:
            split_pos = segment.rfind(" ")
        if split_pos == -1:
            # Absolute fallback: hard cut but avoid splitting inside a tag
            split_pos = _safe_hard_cut(segment, max_chars)

        chunk = remaining[: split_pos + 1].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos + 1 :]

    return chunks


def _merge_tiny_chunks(chunks: list[str], max_chars: int, min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Fold chunks shorter than *min_chars* into the previous chunk, else the next, when the result fits."""
    merged: list[str] = []
    pending: str | None = None
    for chunk in chunks:
        if pending is not None:
            if len(pending) + 1 + len(chunk) <= max_chars:
                chunk = f"{pending} {chunk}"
            else:
                merged.append(pending)
            pending = None
        if len(chunk) >= min_chars:
            merged.append(chunk)
        elif merged and len(merged[-1]) + 1 + len(chunk) <= max_chars:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            pending = chunk
    if pending is not None:
        merged.append(pending)
    return merged


def _find_last_sentence_end(text: str) -> int:
    """Return the index of the last character of the last sentence end in *text*.

    That character is the terminal punctuation or the closing quote that
    follows it (``«¿Vienes?»`` ends at ``»``). Periods after abbreviations
    (``Dr.``, ``Sra.``, ``núm.``), after single-letter initials (``J. R.``,
    ``p. ej.``) and inside bracket tags (``[laugh]``) are skipped. A period
    after a number is a boundary when whitespace follows it (``en 1990.
    Luego``); decimals such as ``3.5`` never match because no whitespace
    follows the period. CJK sentence-ending punctuation (``。！？``) counts too.
    """
    best = -1
    for m in _SENTENCE_END_RE.finditer(text):
        pos = m.start()
        run = m.group(0)[: len(m.group(0)) - len(m.group(1))]
        if run == "." and _period_is_abbreviation(text, pos):
            continue
        if _inside_bracket_tag(text, pos):
            continue
        best = m.end() - 1
    # CJK sentence-ending punctuation
    for m in re.finditer(r"[\u3002\uff01\uff1f]", text):
        if m.start() > best:
            best = m.start()
    return best


def _period_is_abbreviation(text: str, pos: int) -> bool:
    """True when the period at *pos* does not end a sentence.

    Covers abbreviations (``Sra.``, ``etc.``), number-referencing ones only
    when a number follows (``art. 5`` but not ``a work of art.``), list
    markers (``1.``, never years like ``1990.``) and initials (``J. R.``,
    ``p. ej.``, ``D. Manuel``) while keeping ``plan B.`` or ``vitamina C.``
    as sentence ends.
    """
    token_start = pos
    while token_start > 0 and (text[token_start - 1].isalnum() or text[token_start - 1] in "ºª"):
        token_start -= 1
    token = text[token_start:pos]
    if not token:
        return False
    after = text[pos + 1 :]
    if token.isdigit():
        before = text[token_start - 1] if token_start > 0 else ""
        return len(token) <= 2 and (before == "" or before.isspace() or before == ":")
    low = token.lower()
    if low in _NUMBER_ABBREVIATIONS:
        return bool(_NUMBER_AFTER_RE.match(after))
    if low in _ABBREVIATIONS:
        return True
    if len(token) == 1 and token.isalpha():
        return _single_letter_is_initial(text, token, token_start, after)
    return False


def _single_letter_is_initial(text: str, token: str, token_start: int, after: str) -> bool:
    if token.islower() or token == "D":
        return True
    previous = text[:token_start].rstrip()
    chained = (
        len(previous) >= 2
        and previous[-1] == "."
        and previous[-2].isupper()
        and (len(previous) == 2 or not previous[-3].isalnum())
    )
    return chained or bool(_INITIAL_AFTER_RE.match(after))


def _find_last_clause_boundary(text: str) -> int:
    """Return the index of the last clause-boundary punctuation."""
    best = -1
    for m in re.finditer(r"[;:,\u2014](?:\s|$)", text):
        pos = m.start()
        # Skip if inside a bracket tag
        if _inside_bracket_tag(text, pos):
            continue
        best = pos
    return best


def _inside_bracket_tag(text: str, pos: int) -> bool:
    """Return True if *pos* falls inside a ``[...]`` tag."""
    for m in _PARA_TAG_RE.finditer(text):
        if m.start() < pos < m.end():
            return True
    return False


def _safe_hard_cut(segment: str, max_chars: int) -> int:
    """Find a hard-cut position that doesn't split a ``[tag]``."""
    cut = max_chars - 1
    # Check if the cut falls inside a bracket tag; if so, move before it
    for m in _PARA_TAG_RE.finditer(segment):
        if m.start() < cut < m.end():
            return m.start() - 1 if m.start() > 0 else cut
    return cut


def concatenate_audio_chunks(
    chunks: List[np.ndarray],
    sample_rate: int,
    crossfade_ms: int = 50,
) -> np.ndarray:
    """Concatenate audio arrays with a short crossfade to eliminate clicks.

    Each chunk is expected to be a 1-D float32 ndarray at *sample_rate* Hz.
    """
    if not chunks:
        return np.array([], dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    crossfade_samples = int(sample_rate * crossfade_ms / 1000)
    result = np.array(chunks[0], dtype=np.float32, copy=True)

    for chunk in chunks[1:]:
        if len(chunk) == 0:
            continue
        overlap = min(crossfade_samples, len(result), len(chunk))
        if overlap > 0:
            fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
            fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            result[-overlap:] = result[-overlap:] * fade_out + chunk[:overlap] * fade_in
            result = np.concatenate([result, chunk[overlap:]])
        else:
            result = np.concatenate([result, chunk])

    return result


def _chunk_dump_dir(text: str, seed: int | None) -> Path | None:
    """Directory for per-chunk dumps when ``VOICEBOX_DUMP_CHUNKS`` is set, else None.

    Seeded runs share one sub-directory per (text, seed): identical requests
    are reproducible, so their dumps may overwrite each other. Unseeded runs
    (including "regenerate") get a fresh suffix so takes never collide.
    """
    root = os.environ.get("VOICEBOX_DUMP_CHUNKS")
    if not root:
        return None
    digest = hashlib.sha1(f"{seed}:{text}".encode()).hexdigest()[:10]
    if seed is None:
        digest = f"{digest}-{uuid.uuid4().hex[:6]}"
    return Path(root) / digest


def _dump_chunk(dump_dir: Path | None, index: int, chunk_text: str, audio: np.ndarray, sample_rate: int) -> None:
    """Write one chunk's audio (before crossfading) and its text for offline analysis.

    Dumping is diagnostics only: any failure is logged and generation goes on.
    """
    if dump_dir is None:
        return
    try:
        dump_dir.mkdir(parents=True, exist_ok=True)
        save_audio(audio, str(dump_dir / f"chunk_{index:03d}.wav"), sample_rate)
        (dump_dir / f"chunk_{index:03d}.txt").write_text(chunk_text, encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not dump chunk %d to %s: %s", index, dump_dir, exc)


async def generate_chunked(
    backend,
    text: str,
    voice_prompt: dict,
    language: str = "en",
    seed: int | None = None,
    instruct: str | None = None,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    crossfade_ms: int = 50,
    trim_fn=None,
    runaway_detector=None,
) -> Tuple[np.ndarray, int]:
    """Generate audio with automatic chunking for long text.

    For text shorter than *max_chunk_chars* this is a thin wrapper around
    ``backend.generate()`` with zero overhead.

    For longer text the input is split at natural sentence boundaries,
    each chunk is generated independently, optionally trimmed (useful for
    Chatterbox engines that hallucinate trailing noise), and the results
    are concatenated with a crossfade (or hard cut if *crossfade_ms* is 0).

    Parameters
    ----------
    backend : TTSBackend
        Any backend implementing the ``generate()`` protocol.
    text : str
        Input text (may be arbitrarily long).
    voice_prompt, language, seed, instruct
        Forwarded to ``backend.generate()`` verbatim.
    max_chunk_chars : int
        Maximum characters per chunk (default 800).
    crossfade_ms : int
        Crossfade duration in milliseconds between chunks.  0 for a hard
        cut with no overlap (default 50).
    trim_fn : callable | None
        Optional ``(audio, sample_rate) -> audio`` post-processing
        function applied to each chunk before concatenation (e.g.
        ``trim_tts_output`` for Chatterbox engines).
    runaway_detector : callable | None
        Optional ``(audio, sample_rate) -> bool`` detector. When it flags
        unstable output, the affected text is split in half and retried.

    Returns
    -------
    (audio, sample_rate) : Tuple[np.ndarray, int]
    """
    async def generate_one(
        chunk_text: str,
        chunk_seed: int | None,
        retry_depth: int = 0,
    ) -> tuple[np.ndarray, int]:
        chunk_audio, chunk_sr = await backend.generate(
            chunk_text,
            voice_prompt,
            language,
            chunk_seed,
            instruct,
        )

        if runaway_detector is not None and runaway_detector(chunk_audio, chunk_sr):
            if retry_depth >= MAX_RUNAWAY_RETRIES or len(chunk_text) <= MIN_RUNAWAY_RETRY_CHARS:
                raise RuntimeError(
                    "TTS output remained unstable after retrying smaller text chunks"
                )

            retry_max_chars = max(MIN_RUNAWAY_RETRY_CHARS, len(chunk_text) // 2)
            retry_chunks = split_text_into_chunks(chunk_text, retry_max_chars, merge_tiny=False)
            if len(retry_chunks) <= 1:
                raise RuntimeError("Unable to split unstable TTS output for retry")

            logger.warning(
                "Detected unstable TTS output for %d chars; retrying as %d smaller chunks",
                len(chunk_text),
                len(retry_chunks),
            )
            retry_audio: list[np.ndarray] = []
            for i, retry_text in enumerate(retry_chunks):
                retry_seed = (
                    chunk_seed + ((retry_depth + 1) * 1000) + i
                    if chunk_seed is not None
                    else None
                )
                audio, sample_rate = await generate_one(
                    retry_text,
                    retry_seed,
                    retry_depth + 1,
                )
                retry_audio.append(np.asarray(audio, dtype=np.float32))

            return (
                concatenate_audio_chunks(
                    retry_audio,
                    sample_rate,
                    crossfade_ms=crossfade_ms,
                ),
                sample_rate,
            )

        if trim_fn is not None:
            chunk_audio = trim_fn(chunk_audio, chunk_sr)
        return np.asarray(chunk_audio, dtype=np.float32), chunk_sr

    chunks = split_text_into_chunks(text, max_chunk_chars)
    dump_dir = _chunk_dump_dir(text, seed)

    if len(chunks) <= 1:
        # Short text — single-shot fast path, on the normalised text so a
        # raw newline never reaches the model.
        single = chunks[0] if chunks else text
        audio, sample_rate = await generate_one(single, seed)
        _dump_chunk(dump_dir, 0, single, audio, sample_rate)
        return audio, sample_rate

    # Long text — chunked generation
    logger.info(
        "Splitting %d chars into %d chunks (max %d chars each)",
        len(text),
        len(chunks),
        max_chunk_chars,
    )
    audio_chunks: List[np.ndarray] = []
    sample_rate: int | None = None

    for i, chunk_text in enumerate(chunks):
        logger.info(
            "Generating chunk %d/%d (%d chars)",
            i + 1,
            len(chunks),
            len(chunk_text),
        )
        # Vary the seed per chunk to avoid correlated RNG artefacts,
        # but keep it deterministic so the same (text, seed) pair
        # always produces the same output.
        chunk_seed = (seed + i) if seed is not None else None

        chunk_audio, chunk_sr = await generate_one(
            chunk_text,
            chunk_seed,
        )
        _dump_chunk(dump_dir, i, chunk_text, chunk_audio, chunk_sr)

        audio_chunks.append(chunk_audio)
        if sample_rate is None:
            sample_rate = chunk_sr

    audio = concatenate_audio_chunks(audio_chunks, sample_rate, crossfade_ms=crossfade_ms)
    return audio, sample_rate
