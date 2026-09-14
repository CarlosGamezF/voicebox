interface SampleWarningListProps {
  warnings: string[];
}

/** Quality notes returned by the server, one per line (used inside toasts). */
export function SampleWarningList({ warnings }: SampleWarningListProps) {
  return (
    <ul className="list-disc space-y-0.5 pl-4">
      {warnings.map((warning) => (
        <li key={warning}>{warning}</li>
      ))}
    </ul>
  );
}
