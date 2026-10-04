from contextlib import nullcontext
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from src.ai.qwen_transcriber import QwenTranscriber, MODEL, REVISION


class QwenRuntimeTests(unittest.TestCase):
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
