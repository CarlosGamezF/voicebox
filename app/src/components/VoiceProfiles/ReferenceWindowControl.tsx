import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import type { ReferenceRange } from '@/lib/hooks/useReferenceWindow';

const STEP_S = 0.1;
const MIN_WINDOW_S = 1;

interface ReferenceWindowControlProps {
  /** Length of the whole clip, in seconds. */
  durationS: number;
  range: ReferenceRange;
  onChange: (range: ReferenceRange) => void;
  isNarrowed: boolean;
  disabled?: boolean;
}

/** Dual-thumb picker for the part of a long clip to keep as the voice reference. */
export function ReferenceWindowControl({
  durationS,
  range,
  onChange,
  isNarrowed,
  disabled = false,
}: ReferenceWindowControlProps) {
  const { t } = useTranslation();
  const [start, end] = range;

  return (
    <div className="space-y-2 rounded-lg border border-border p-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium leading-none">{t('sampleQuality.window.label')}</span>
        {isNarrowed && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs"
            onClick={() => onChange([0, durationS])}
            disabled={disabled}
          >
            {t('sampleQuality.window.reset')}
          </Button>
        )}
      </div>
      <Slider
        value={[start, end]}
        onValueChange={([nextStart = start, nextEnd = end]) => onChange([nextStart, nextEnd])}
        min={0}
        max={durationS}
        step={STEP_S}
        minStepsBetweenThumbs={MIN_WINDOW_S / STEP_S}
        disabled={disabled}
        aria-label={t('sampleQuality.window.label')}
      />
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-xs text-muted-foreground">
        <span className="font-mono">
          {t('sampleQuality.window.range', {
            start: start.toFixed(1),
            end: end.toFixed(1),
            length: (end - start).toFixed(1),
          })}
        </span>
        <span>{t('sampleQuality.window.hint')}</span>
      </div>
    </div>
  );
}
