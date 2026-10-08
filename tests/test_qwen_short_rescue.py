"""Precise short-clip rescue, cumulative budgets and public numeric evidence."""
from contextlib import nullcontext
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock,patch
import numpy as np
from scripts.qwen_quality import generation_audit
from src.ai.qwen_transcriber import QwenTranscriber,QwenTokenBudgetError,RATE,MAX_BLOCK_ATTEMPTS
from scripts.qwen_sharding import validate_block_row
from test_qwen_block_recovery import recognizer,MODULES


class ShortRescueTests(unittest.TestCase):
    def test_historical_short_spans_recover_all_samples_in_order(self):
        for start,end in ((122.107,135.144),(2965.331,2980.331),(10.00001,23.03703)):
            t=recognizer();samples=np.arange(round(end*RATE)-round(start*RATE),dtype=np.float32)
            good=[];calls=[]
            def decode(part,**kw):
                calls.append(len(part))
                if len(part)>8*RATE:return {'text':'TRUNCATED','quality_state':'retry_timeout'}
                good.append(part.copy());return {'text':str(int(part[0])),'quality_state':'recognized'}
            t._recognize=decode
            with patch.dict(sys.modules,MODULES):
                row=t._recognize_resilient(samples,{'chunk_id':2,'start':start,'end':end},1e30)
            np.testing.assert_array_equal(np.concatenate(good),samples)
            self.assertEqual(len(calls),4);self.assertEqual(row['quality_state'],'split_retry')
            self.assertEqual(row['missing_intervals'],[]);self.assertNotIn('TRUNCATED',row['text'])
            self.assertEqual([a['attempt_kind'] for a in row['recognition_attempts']],
                ['original','unhinted_full_retry','split_short','split_short'])
            self.assertTrue(all('elapsed_seconds' in a and 'budget_seconds' in a for a in row['recognition_attempts']))

    def test_large_block_failed_15s_leaf_is_bisected_once(self):
        t=recognizer();samples=np.arange(122*RATE,dtype=np.float32);good=[];calls=[]
        def decode(part,**kw):
            calls.append(len(part))
            if len(part)>8*RATE:raise QwenTokenBudgetError('normal',2048,2040)
            good.append(part.copy());return {'text':'complete'}
        t._recognize=decode
        block={'chunk_id':49,'start':5508.555,'end':5630.555}
        with patch.dict(sys.modules,MODULES):row=t._recognize_resilient(samples,block,1e30)
        np.testing.assert_array_equal(np.concatenate(good),samples)
        self.assertEqual(len(calls),31);self.assertLessEqual(len(calls),MAX_BLOCK_ATTEMPTS)
        self.assertEqual(row['missing_intervals'],[]);self.assertEqual(row['quality_state'],'split_retry')
        validate_block_row(block,dict(row,**block))

    def test_no_unbounded_recovery_even_for_unusually_large_block(self):
        t=recognizer();t._recognize=MagicMock(side_effect=QwenTokenBudgetError('normal',2048,2040))
        block={'chunk_id':0,'start':0,'end':300}
        with patch.dict(sys.modules,MODULES):row=t._recognize_resilient(np.zeros(300*RATE),block,1e30)
        self.assertEqual(t._recognize.call_count,MAX_BLOCK_ATTEMPTS)
        self.assertEqual(sum(g['end']-g['start'] for g in row['missing_intervals']),300)
        self.assertTrue(any(g.get('stop_reason')=='attempt_limit' for g in row['missing_intervals']))
        validate_block_row(block,dict(row,**block))

    def test_cumulative_block_deadline_discards_late_text_and_later_blocks_continue(self):
        now=[0.0];t=recognizer();calls=[]
        def decode(part,**kw):
            calls.append(len(part));now[0]+=310
            return {'text':'late text','quality_state':'retry_timeout'}
        t._recognize=decode;block={'chunk_id':0,'start':0,'end':15}
        with patch.dict(sys.modules,MODULES),patch('src.ai.qwen_transcriber.time.monotonic',side_effect=lambda:now[0]):
            row=t._recognize_resilient(np.zeros(15*RATE),block,10000)
            self.assertLessEqual(len(calls),3);self.assertEqual(row['text'],'')
            self.assertTrue(any(g.get('stop_reason')=='block_deadline' for g in row['missing_intervals']))
            t._recognize=lambda *a,**kw:{'text':'next block'}
            next_row=t._recognize_resilient(np.zeros(RATE),{'chunk_id':1,'start':15,'end':16},10000)
        self.assertEqual(next_row['text'],'next block');validate_block_row(block,dict(row,**block))

    def test_remaining_worker_deadline_overrules_new_block_budget(self):
        now=[0.0];t=recognizer();calls=[]
        def decode(part,**kw):
            calls.append(1);now[0]=11;return {'text':'late partial'}
        t._recognize=decode
        with patch.dict(sys.modules,MODULES),patch('src.ai.qwen_transcriber.time.monotonic',side_effect=lambda:now[0]):
            row=t._recognize_resilient(np.zeros(15*RATE),{'start':0,'end':15},10)
        self.assertEqual(len(calls),1);self.assertEqual(row['text'],'')
        self.assertEqual(row['missing_intervals'][0]['error_code'],'worker_deadline')
        self.assertEqual(row['recognition_attempts'][0]['budget_seconds'],10)

    def test_actual_generation_eos_and_cap_keep_timing_without_content(self):
        now=[0.0]
        class Backend:
            generation_config=SimpleNamespace(eos_token_id=9)
            def generate(self,**kw):
                now[0]+=2.5
                return np.array([[1,2,3,8,9]])
        model=SimpleNamespace(model=Backend(),max_new_tokens=2048)
        with generation_audit(model,clock=lambda:now[0]) as records:
            model.model.generate(input_ids=np.zeros((1,3)))
        self.assertEqual(records,[{'generated_tokens':2,'token_limit':2048,'elapsed_seconds':2.5,
                                 'stop_reason':'eos','eos_observed':True}])
        self.assertNotIn('generate',model.model.__dict__)
        with generation_audit(model,clock=lambda:now[0]) as records:
            model.model.generate(input_ids=np.zeros((1,3)),max_new_tokens=2)
        self.assertEqual(records[0]['stop_reason'],'token_limit')

    def test_token_failure_keeps_actual_generation_evidence(self):
        t=recognizer();t._model.processor.tokenizer.encode.return_value=[1]
        t._model.model.generate.side_effect=lambda **kw:np.zeros((1,2051))
        def transcribe(**kw):
            t._model.model.generate(input_ids=np.zeros((1,3)),max_new_tokens=2048)
            return [SimpleNamespace(text='private text never audited')]
        t._model.transcribe.side_effect=transcribe
        for unhinted in (False,True):
            with patch.dict(sys.modules,MODULES),self.assertRaises(QwenTokenBudgetError) as caught:
                t._recognize(np.zeros(RATE),unhinted=unhinted)
            diag=caught.exception.diagnostic
            self.assertEqual(diag['generation_diagnostics'][0]['generated_tokens'],2048)
            self.assertEqual(diag['generation_phase'],'unhinted' if unhinted else 'normal')
            self.assertEqual(diag['generation_diagnostics'][0]['generation_phase'],diag['generation_phase'])
            self.assertNotIn('private',json.dumps(diag))
