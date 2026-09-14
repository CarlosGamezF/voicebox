import { useTranslation } from 'react-i18next';
import { Badge } from '@/components/ui/badge';
import { useSampleAnalysis } from '@/lib/hooks/useProfiles';
import { cn } from '@/lib/utils/cn';

const BADGE_CLASS = 'px-2 py-0 text-[10px] font-medium';

interface SampleQualityBadgeProps {
  sampleId: string;
}

/** Server-side quality verdict and length of a stored sample. */
export function SampleQualityBadge({ sampleId }: SampleQualityBadgeProps) {
  const { t } = useTranslation();
  const { data, isError } = useSampleAnalysis(sampleId);

  if (isError) {
    return null;
  }

  if (!data) {
    return (
      <Badge variant="secondary" className={BADGE_CLASS}>
        {t('sampleQuality.badge.checking')}
      </Badge>
    );
  }

  const hasNotes = data.warnings.length > 0;

  return (
    <div className="flex items-center gap-2">
      <Badge
        variant="outline"
        className={cn(
          BADGE_CLASS,
          hasNotes
            ? 'cursor-help border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400'
            : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
        )}
        title={hasNotes ? data.warnings.join('\n') : undefined}
      >
        {hasNotes
          ? t('sampleQuality.badge.notes', { count: data.warnings.length })
          : t('sampleQuality.badge.good')}
      </Badge>
      <span className="font-mono text-[10px] text-muted-foreground">
        {t('sampleQuality.duration', { seconds: data.duration_s.toFixed(1) })}
      </span>
    </div>
  );
}
