"""Slow progressing PCM, stalled input and hard deadlines without ASR or network."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import yaml
from scripts import production_qwen as pipeline
from src.ai.qwen_transcriber import QwenTranscriber, RATE


class FakeVad:
    def __init__(self, *args, **kwargs): pass
    def empty(self): return True
    def accept_waveform(self, samples): pass
    def flush(self): pass


class PreparationBudgetTests(unittest.TestCase):
    def stream(self, read, eof, clock, *, timeout, idle_timeout=None, returncode=0):
        config = lambda: SimpleNamespace(silero_vad=SimpleNamespace())
        module = SimpleNamespace(VadModelConfig=config, VoiceActivityDetector=FakeVad)
        transcriber = QwenTranscriber()
        def sleep(_): clock[0] += 100
        with patch.dict('sys.modules', {'sherpa_onnx': module}), \
                patch('src.ai.qwen_transcriber.time.monotonic', side_effect=lambda: clock[0]), \
                patch('src.ai.qwen_transcriber.time.sleep', side_effect=sleep):
            transcriber.prepare_pcm_stream(read, eof, lambda: b'Duration: 00:00:01.00',
                lambda: returncode, audio_path='local synthetic PCM', timeout=timeout,
                idle_timeout=idle_timeout)
        return transcriber

    def progressing_reader(self, frames, clock):
        frame = np.zeros(512, dtype=np.float32).tobytes()
        remaining = [frames]
        def read(_):
            if remaining[0]:
                remaining[0] -= 1; clock[0] += 500
                return frame
            return b''
        return read, lambda: remaining[0] == 0

    def test_progressing_stream_past_old_forty_minutes_finishes_with_same_samples(self):
        clock = [0]; read, eof = self.progressing_reader(6, clock)
        with self.assertRaisesRegex(TimeoutError, 'deadline exceeded'):
            self.stream(read, eof, clock, timeout=2400)
        clock = [0]; read, eof = self.progressing_reader(6, clock)
        t = self.stream(read, eof, clock, timeout=pipeline.PREPARE_STREAM_TIMEOUT,
                        idle_timeout=pipeline.PREPARE_IDLE_TIMEOUT)
        self.assertEqual(t.last_audio_duration, 6 * 512 / RATE)
        self.assertEqual(t.last_prepare_stats['elapsed_seconds'], 3000)
        self.assertTrue(t.last_prepare_stats['stream_eof'])

    def test_stalled_input_stops_early_without_spending_the_long_total_budget(self):
        clock = [0]
        with self.assertRaisesRegex(TimeoutError, 'Audio preparation stalled') as failure:
            self.stream(lambda _: b'', lambda: False, clock,
                timeout=pipeline.PREPARE_STREAM_TIMEOUT, idle_timeout=pipeline.PREPARE_IDLE_TIMEOUT)
        self.assertLess(clock[0], pipeline.PREPARE_STREAM_TIMEOUT)
        self.assertEqual(pipeline.failure_code(failure.exception), 'preparation_stalled')

    def test_continuous_progress_still_cannot_exceed_the_absolute_deadline(self):
        clock = [0]; read, eof = self.progressing_reader(20, clock)
        with self.assertRaisesRegex(TimeoutError, 'deadline exceeded') as failure:
            self.stream(read, eof, clock, timeout=pipeline.PREPARE_STREAM_TIMEOUT,
                        idle_timeout=pipeline.PREPARE_IDLE_TIMEOUT)
        self.assertEqual(pipeline.failure_code(failure.exception), 'preparation_deadline')
        self.assertLess(clock[0], 20 * 500)

    def test_process_exit_race_still_drains_final_pcm_and_counts_real_tail_only(self):
        frame = np.zeros(512, dtype=np.float32).tobytes()
        tail = np.zeros(17, dtype=np.float32).tobytes()
        source = iter([frame, b'', tail, b'', b''])
        t = self.stream(lambda _: next(source), lambda: True, [0], timeout=100, idle_timeout=5)
        self.assertEqual(t.last_audio_duration, 529 / RATE)
        self.assertTrue(t.last_prepare_stats['stream_eof'])
        with self.assertRaisesRegex(RuntimeError, 'Audio download failed'):
            self.stream(lambda _: b'', lambda: True, [0], timeout=100, returncode=1)

    def test_formal_prepare_passes_budgets_and_keeps_timing_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'source.raw'; path.write_bytes(b'original PCM')
            handle = SimpleNamespace(path=path, process=MagicMock(), stderr_chunks=[])
            t = MagicMock(); failure = TimeoutError('Audio preparation stalled')
            t.prepare_pcm_stream.side_effect = failure
            t.last_prepare_stats = {'elapsed_seconds':301, 'stream_eof':False}
            spec = {}
            with self.assertRaises(TimeoutError) as caught:
                pipeline.prepare_audio_stream(t, handle, spec)
            self.assertIs(caught.exception, failure)
            self.assertEqual(path.read_bytes(), b'original PCM')
            kwargs = t.prepare_pcm_stream.call_args.kwargs
            self.assertEqual(kwargs['timeout'], 5400)
            self.assertEqual(kwargs['idle_timeout'], 300)
            self.assertEqual(spec['preparation_timing']['elapsed_seconds'], 301)

    def test_safe_failure_audit_includes_budget_and_does_not_publish_provider_text(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'RUNNER_TEMP':tmp}):
            spec = {'prepare_phase':'vad', 'preparation_timing':{
                'elapsed_seconds':5401, 'idle_seconds':1, 'timeout_seconds':5400,
                'idle_timeout_seconds':300, 'stream_eof':False, 'private_url':'secret'}}
            pipeline.preparation_failure_audit(spec, {}, TimeoutError('Audio preparation deadline exceeded'))
            audit = json.loads(pipeline.out('prepare-failure.json').read_text())
            self.assertEqual(audit['error_code'], 'preparation_deadline')
            self.assertEqual(audit['preparation_timeout_seconds'], 5400)
            self.assertFalse(audit['preparation_stream_eof'])
            self.assertNotIn('private', json.dumps(audit)); self.assertNotIn('secret', json.dumps(audit))

    def test_workflow_allows_environment_retention_and_upload_after_the_stream_deadline(self):
        root = Path(__file__).resolve().parents[1]
        jobs = yaml.load((root/'.github/workflows/qwen_production_lecture.yml').read_text(),
                         Loader=yaml.BaseLoader)['jobs']
        prepare = jobs['prepare']
        self.assertGreaterEqual(int(prepare['timeout-minutes'])*60-pipeline.PREPARE_STREAM_TIMEOUT, 1800)
        upload = next(s for s in prepare['steps'] if
            s.get('with', {}).get('name') == 'qwen-production-prepare-${{ inputs.task_slot }}')
        self.assertIn('always()', upload['if'])
        self.assertEqual(jobs['asr']['if'], "${{ needs.prepare.result == 'success' }}")


if __name__ == '__main__': unittest.main()
