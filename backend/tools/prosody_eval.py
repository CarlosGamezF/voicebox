"""Local evaluation harness for Spanish prosody experiments.

Drives a running Voicebox backend over REST: for every condition, text and
seed it requests a generation, downloads the audio, measures it, transcribes
it with the configured Whisper and computes the word error rate against the
input. It also writes blind A/B pairs so a listener can rate conditions
without knowing which is which.

    backend/venv/bin/python -m backend.tools.prosody_eval \\
        --profile Carlos --seeds 1 2 3 --out ./eval-out \\
        --condition chunk550:max_chunk_chars=550 --condition chunk300:max_chunk_chars=300

Never use retry/regenerate for comparisons: they re-seed and re-chunk.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

from backend.utils.text_normalize import normalize_text

CORPUS = [
    {"id": "short_question", "text": "¿Vienes conmigo esta tarde al mercado?"},
    {
        "id": "narrative_date",
        "text": (
            "El 15 de septiembre de 2026 la Sra. García llegó tarde a la reunión. Nadie dijo nada, "
            "pero todos miraron el reloj cuando entró con el café en la mano y una sonrisa de disculpa."
        ),
    },
    {
        "id": "long_paragraph",
        "text": (
            "El consejo de administración se reunió el martes por la tarde para analizar los resultados del primer "
            "semestre, que fueron mejores de lo esperado en casi todas las áreas. Sin embargo, el director financiero "
            "advirtió de que el segundo semestre sería más complicado, porque los costes de la energía siguen subiendo "
            "y la demanda en los mercados europeos se está enfriando. ¿Qué medidas se tomarán? Por ahora, ninguna "
            "definitiva: se estudiará una reducción gradual de gastos y se aplazarán algunas inversiones hasta que la "
            "situación se aclare. ¡Paciencia!"
        ),
    },
    {
        "id": "two_paragraphs",
        "text": (
            "Primero, una advertencia: nada de esto es definitivo. Los datos cambian cada semana y las conclusiones "
            "también.\n\nSegundo, una promesa: en cuanto haya novedades concretas, la plantilla será la primera en saberlo."
        ),
    },
    {
        "id": "numbers_currency",
        "text": (
            "La factura ascendió a 1.234,56 € y se pagó el 15/09/2026 con un descuento del 12,5 % por pronto pago, "
            "según consta en el recibo número 48."
        ),
    },
    {
        "id": "abbreviations",
        "text": (
            "La Dra. Pérez, núm. 3 de la lista, recomendó, p. ej., caminar veinte minutos al día. "
            "El Sr. Ortega y la Srta. Ruiz estuvieron de acuerdo."
        ),
    },
]

DEFAULT_PAYLOAD = {"engine": "qwen", "model_size": "1.7B", "normalize": True}


@dataclass
class Condition:
    name: str
    overrides: dict = field(default_factory=dict)


def build_generate_payload(cond: Condition, *, profile_id: str, text: str, seed: int, language: str = "es") -> dict:
    payload = {"profile_id": profile_id, "text": text, "seed": seed, "language": language, **DEFAULT_PAYLOAD}
    payload.update(cond.overrides)
    return payload


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate via Levenshtein distance over normalised words."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return 1.0 if hyp else 0.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


def audio_metrics(audio: np.ndarray, sample_rate: int, text: str, silence_db: float = -40.0, pause_min_s: float = 0.3) -> dict:
    """Duration, speaking rate, edge silence, internal pauses, level and clipping of one take."""
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    duration = n / sample_rate
    threshold = 10 ** (silence_db / 20)
    loud = np.flatnonzero(np.abs(audio) > threshold)
    lead = loud[0] / sample_rate if loud.size else duration
    trail = (n - 1 - loud[-1]) / sample_rate if loud.size else 0.0
    pauses = _internal_pauses(audio, sample_rate, threshold, loud) if loud.size else []
    long_pauses = [p for p in pauses if p >= pause_min_s]
    rms = float(np.sqrt(np.mean(audio**2))) if n else 0.0
    return {
        "duration_s": round(duration, 3),
        "chars_per_s": round(len(text) / duration, 2) if duration else 0.0,
        "leading_silence_s": round(float(lead), 3),
        "trailing_silence_s": round(float(trail), 3),
        "pauses_over_300ms": len(long_pauses),
        "longest_pause_s": round(max(long_pauses), 3) if long_pauses else 0.0,
        "rms_dbfs": round(20 * np.log10(rms), 1) if rms > 0 else -120.0,
        "peak": round(float(np.abs(audio).max()), 3) if n else 0.0,
        "clipped_085_ratio": float(np.mean(np.abs(audio) >= 0.849)) if n else 0.0,
        "clipped_099_ratio": float(np.mean(np.abs(audio) >= 0.99)) if n else 0.0,
    }


def _internal_pauses(audio: np.ndarray, sample_rate: int, threshold: float, loud: np.ndarray) -> list[float]:
    """Lengths in seconds of the quiet stretches between the first and last loud sample."""
    frame = max(1, int(0.01 * sample_rate))
    region = audio[loud[0] : loud[-1] + 1]
    frames = region[: len(region) - len(region) % frame].reshape(-1, frame)
    quiet = np.abs(frames).max(axis=1) <= threshold
    pauses, run = [], 0
    for q in quiet:
        if q:
            run += 1
        elif run:
            pauses.append(run * frame / sample_rate)
            run = 0
    return pauses


def make_blind_pairs(rows: list[dict], rng_seed: int = 0) -> tuple[list[dict], dict]:
    """A/B pairs of takes per (text, seed) with the condition hidden in a separate key."""
    rng = random.Random(rng_seed)
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["text_id"], row["seed"]), []).append(row)
    pairs, key = [], {}
    for (text_id, seed), takes in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        takes = sorted(takes, key=lambda r: r["condition"])
        for i, (left, right) in enumerate(combinations(takes, 2)):
            a, b = (left, right) if rng.random() < 0.5 else (right, left)
            pair_id = f"{text_id}-s{seed}-{i}"
            pairs.append({"pair_id": pair_id, "text_id": text_id, "seed": seed, "A": a["wav"], "B": b["wav"]})
            key[pair_id] = {"A": a["condition"], "B": b["condition"]}
    return pairs, key


# ── REST runner ──────────────────────────────────────────────────────────


def parse_condition(spec: str) -> Condition:
    """``name:key=value,key=value`` with int/float/bool coercion."""
    name, _, rest = spec.partition(":")
    overrides: dict = {}
    for item in filter(None, rest.split(",")):
        key, _, raw = item.partition("=")
        overrides[key.strip()] = _coerce(raw.strip())
    return Condition(name=name.strip(), overrides=overrides)


def _coerce(raw: str):
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def _resolve_profile_id(session, base_url: str, profile: str) -> str:
    profiles = session.get(f"{base_url}/profiles", timeout=30).json()
    for p in profiles:
        if p["id"] == profile or p["name"].lower() == profile.lower():
            return p["id"]
    raise SystemExit(f"Profile {profile!r} not found; available: {[p['name'] for p in profiles]}")


def _wait_for_generation(session, base_url: str, generation_id: str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with session.get(f"{base_url}/generate/{generation_id}/status", stream=True, timeout=timeout_s) as resp:
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    payload = json.loads(line[5:])
                    if payload.get("status") in {"completed", "failed"}:
                        return payload
        time.sleep(1)
    raise TimeoutError(f"generation {generation_id} did not finish in {timeout_s}s")


def _transcribe(session, base_url: str, wav_path: Path, stt_model: str, language: str) -> str:
    for _ in range(60):
        with wav_path.open("rb") as fh:
            resp = session.post(
                f"{base_url}/transcribe",
                files={"file": (wav_path.name, fh, "audio/wav")},
                data={"language": language, "model": stt_model},
                timeout=600,
            )
        if resp.status_code == 202:  # model still downloading/loading
            time.sleep(5)
            continue
        resp.raise_for_status()
        return resp.json()["text"]
    raise TimeoutError("transcription model never became available")


def run(args: argparse.Namespace) -> None:
    import requests

    session = requests.Session()
    base_url = args.base_url.rstrip("/")
    profile_id = _resolve_profile_id(session, base_url, args.profile)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = [c for c in CORPUS if not args.texts or c["id"] in args.texts]
    rows: list[dict] = []
    for cond in args.conditions:
        for item in corpus:
            for seed in args.seeds:
                rows.append(_run_take(session, base_url, out, cond, item, seed, profile_id, args))
                _write_results(out, rows)
    pairs, key = make_blind_pairs(rows, rng_seed=args.pair_seed)
    _write_csv(out / "pairs.csv", pairs)
    (out / "pairs_key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    _print_summary(rows)


def _run_take(session, base_url, out, cond, item, seed, profile_id, args) -> dict:
    import soundfile as sf

    payload = build_generate_payload(cond, profile_id=profile_id, text=item["text"], seed=seed, language=args.language)
    started = time.time()
    gen = session.post(f"{base_url}/generate", json=payload, timeout=60).json()
    status = _wait_for_generation(session, base_url, gen["id"], args.timeout)
    wall = round(time.time() - started, 1)
    row = {"condition": cond.name, "text_id": item["id"], "seed": seed, "generation_id": gen["id"], "wall_s": wall, "status": status.get("status"), "error": status.get("error") or ""}
    if status.get("status") != "completed":
        print(f"[{cond.name}] {item['id']} seed {seed}: FAILED {row['error']}", file=sys.stderr)
        return row
    wav_path = out / cond.name / f"{item['id']}_seed{seed}.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(session.get(f"{base_url}/audio/{gen['id']}", timeout=120).content)
    audio, sr = sf.read(wav_path)
    row.update(audio_metrics(audio if audio.ndim == 1 else audio.mean(axis=1), sr, item["text"]))
    row["wav"] = str(wav_path.relative_to(out))
    if not args.skip_wer:
        hypothesis = _transcribe(session, base_url, wav_path, args.stt_model, args.language)
        row["wer"] = round(wer(item["text"], hypothesis), 4)
        row["transcript"] = hypothesis
    print(f"[{cond.name}] {item['id']} seed {seed}: {row['duration_s']} s, {row['chars_per_s']} chars/s, WER {row.get('wer', 'n/a')}, {wall} s wall")
    return row


def _write_results(out: Path, rows: list[dict]) -> None:
    _write_csv(out / "results.csv", rows)
    (out / "results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict]) -> None:
    by_condition: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") == "completed":
            by_condition.setdefault(row["condition"], []).append(row)
    print("\ncondition        takes  WER    chars/s  pauses>0.3s  clip@0.85  wall s")
    for name, takes in by_condition.items():
        print(
            f"{name:16} {len(takes):5d}  {_mean(takes, 'wer'):.3f}  {_mean(takes, 'chars_per_s'):7.2f}  "
            f"{_mean(takes, 'pauses_over_300ms'):11.2f}  {_mean(takes, 'clipped_085_ratio'):9.5f}  {_mean(takes, 'wall_s'):6.1f}"
        )


def _mean(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows if key in row]
    return sum(values) / len(values) if values else float("nan")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:17493")
    parser.add_argument("--profile", required=True, help="voice profile name or id")
    parser.add_argument("--condition", dest="conditions", action="append", type=parse_condition, required=True, help="name:key=value,... (repeatable)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--texts", nargs="*", default=None, help="corpus ids to run (default: all)")
    parser.add_argument("--language", default="es")
    parser.add_argument("--stt-model", default="large")
    parser.add_argument("--skip-wer", action="store_true")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
