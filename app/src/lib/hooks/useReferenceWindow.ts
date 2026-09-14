import { useEffect, useState } from 'react';
import type { ReferenceWindow } from '@/lib/api/types';
import { getAudioDuration } from '@/lib/utils/audio';

/** Clips longer than this get a trim control; shorter ones are always used whole. */
export const REFERENCE_WINDOW_MIN_CLIP_S = 16;

export type ReferenceRange = [start: number, end: number];

/**
 * Length of the selected clip in seconds, or null while unknown or unreadable.
 * Recordings carry `recordedDuration`; uploads are decoded by getAudioDuration.
 */
export function useClipDuration(file: File | undefined): number | null {
  const [measured, setMeasured] = useState<{ file: File; seconds: number } | null>(null);

  useEffect(() => {
    if (!file) return;
    let cancelled = false;
    getAudioDuration(file)
      .then((seconds) => {
        if (!cancelled && Number.isFinite(seconds)) {
          setMeasured({ file, seconds });
        }
      })
      .catch(() => {
        // Unreadable clip: leave the duration unknown so the trim control stays hidden.
      });
    return () => {
      cancelled = true;
    };
  }, [file]);

  return file && measured?.file === file ? measured.seconds : null;
}

/**
 * Start/end selection over a clip that is long enough to trim. The range is keyed to the
 * file, so choosing another clip resets it to the whole clip. `request` is only set once
 * the user narrowed the window, ready to be passed to the add-sample mutation.
 */
export function useReferenceWindow(file: File | undefined, durationS: number | null) {
  const [selection, setSelection] = useState<{ file: File; range: ReferenceRange } | null>(null);

  const isTrimmable =
    durationS !== null && Number.isFinite(durationS) && durationS > REFERENCE_WINDOW_MIN_CLIP_S;

  let range: ReferenceRange | null = null;
  if (file && durationS !== null && isTrimmable) {
    range = selection?.file === file ? selection.range : [0, durationS];
  }

  const setRange = (next: ReferenceRange) => {
    if (file) {
      setSelection({ file, range: next });
    }
  };

  const isNarrowed = range !== null && durationS !== null && (range[0] > 0 || range[1] < durationS);
  const request: Required<ReferenceWindow> | undefined =
    range && isNarrowed
      ? { startS: roundSeconds(range[0]), endS: roundSeconds(range[1]) }
      : undefined;

  return { range, setRange, isNarrowed, request };
}

function roundSeconds(seconds: number): number {
  return Math.round(seconds * 100) / 100;
}
