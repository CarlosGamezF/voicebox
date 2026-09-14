import { useMutation } from '@tanstack/react-query';
import { apiClient } from '@/lib/api/client';
import type { ReferenceWindow, WhisperModelSize } from '@/lib/api/types';
import type { LanguageCode } from '@/lib/constants/languages';

export function useTranscription() {
  return useMutation({
    mutationFn: ({
      file,
      language,
      model,
      referenceWindow,
    }: {
      file: File;
      language?: LanguageCode;
      model?: WhisperModelSize;
      /** Transcribe only this part of the clip, matching the window the sample will keep. */
      referenceWindow?: ReferenceWindow;
    }) => apiClient.transcribeAudio(file, language, model, referenceWindow),
  });
}
