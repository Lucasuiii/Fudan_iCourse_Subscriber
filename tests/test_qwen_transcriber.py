from contextlib import nullcontext
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from src.ai.qwen_transcriber import QwenTranscriber, MODEL, REVISION


class QwenRuntimeTests(unittest.TestCase):
    def consume(self, transcriber, data, read_fn=None, chunks=None):
        """Exercise the PCM loop without loading model weights or native VAD."""
        vad = SimpleNamespace(empty=lambda: True, accept_waveform=lambda _: None,
                              flush=lambda: None)
        sherpa = SimpleNamespace(
            VadModelConfig=lambda: SimpleNamespace(silero_vad=SimpleNamespace()),
            VoiceActivityDetector=lambda *args, **kwargs: vad,
        )
        import tempfile
        with tempfile.NamedTemporaryFile() as audio:
            audio.write(data); audio.flush()
            with patch.dict(sys.modules, {'sherpa_onnx': sherpa}), \
                 patch('src.ai.qwen_transcriber.plan_long_chunks', return_value=chunks or []):
                return transcriber._consume_pcm_stream(
                    read_fn or io.BytesIO(data).read, lambda: True,
                    lambda: b'', lambda: 0, audio_path=audio.name,
                )

    def test_drains_final_pcm_written_between_empty_read_and_process_exit(self):
        import numpy as np
        block = np.zeros(512, dtype=np.float32).tobytes()
        tail = np.zeros(10, dtype=np.float32).tobytes()
        reads = iter([block, b'', tail, b''])
        transcriber = QwenTranscriber()
        self.consume(transcriber, block + tail, lambda _: next(reads, b''))
        self.assertEqual(transcriber.last_audio_duration, 522 / 16000)

    def test_timed_segments_keep_boundary_dedupe_and_raw_alignment_text(self):
        import numpy as np
        transcriber = QwenTranscriber()
        transcriber._init = MagicMock()
        transcriber._recognize = MagicMock(side_effect=[
            {'text': '前面内容边界重复这句话'},
            {'text': '边界重复这句话后续内容'},
        ])
        text, segments = self.consume(
            transcriber, np.zeros(1024, dtype=np.float32).tobytes(),
            chunks=[(0, 0.02), (0.015, 0.064)],
        )
        self.assertEqual(text, '\n'.join(s['text'] for s in segments))
        self.assertEqual(segments[1]['text'], '后续内容')
        self.assertEqual(transcriber.last_chunks[1]['text'], '边界重复这句话后续内容')

    def make(self,texts,tokens=3):
        t=QwenTranscriber()
        t._model=MagicMock()
        t._model.transcribe.side_effect=[[SimpleNamespace(text=s)] for s in texts]
        t._model.processor.tokenizer.encode.return_value=list(range(tokens))
        t._model.max_new_tokens=2048
        return t

    def run_decode(self,t):
        with patch.dict(sys.modules,{'torch':SimpleNamespace(inference_mode=nullcontext),
                                    'transformers':SimpleNamespace(StoppingCriteriaList=list)}):
            return t._recognize([])

    def test_qwen_only_and_pinned_model(self):
        self.assertEqual(MODEL,'Qwen/Qwen3-ASR-1.7B')
        self.assertEqual(len(REVISION),40)
        with self.assertRaises(ValueError): QwenTranscriber(backend='sensevoice')

    def test_filter_fillers_not_short_math(self):
        for text in ('嗯。','啊！','呵呵。'):
            self.assertEqual(self.run_decode(self.make([text]))['text'],'')
        self.assertEqual(self.run_decode(self.make(['矩阵']))['text'],'矩阵')
        self.assertEqual(self.run_decode(self.make(['A']))['text'],'A')

    def test_prompt_echo_retries_without_hints_and_restores_limit(self):
        t=self.make(['术语：矩阵、条件数、扰动、范数、逆矩阵','条件数用于描述敏感性。'])
        t.set_terms(['矩阵','条件数','扰动','范数','逆矩阵'])
        self.assertEqual(self.run_decode(t)['text'],'条件数用于描述敏感性。')
        self.assertEqual(t._model.transcribe.call_args.kwargs['context'],'')
        self.assertEqual(t._model.max_new_tokens,2048)

    def test_truncation_fails_instead_of_persisting_partial(self):
        with self.assertRaises(RuntimeError):
            self.run_decode(self.make(['长内容'],tokens=2040))

    def test_state_not_reused_for_cached_next_lecture(self):
        t=QwenTranscriber();t.last_chunks=[{'text':'old'}]
        t.last_vad_windows=[(1,2)];t._last_duration=300
        t.reset_lecture_state()
        self.assertEqual(t.last_chunks,[])
        self.assertEqual(t.last_audio_duration,0)
        t.release_model()
        self.assertIsNone(t._model)


if __name__=='__main__': unittest.main()
