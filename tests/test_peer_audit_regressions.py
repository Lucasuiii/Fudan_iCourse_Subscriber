"""Reproduce external audit findings without campus login or Actions dispatch."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import MagicMock, patch

from scripts import production_pool as pool
from scripts import production_resource_fetch as resource
from src.runtime.audio_preparation import collect_decode_diagnostics, validate_prepared_audio
from test_production_pool import journal, Simulation
from test_shared_asr_queue import MemoryStore


class DispatchTests(unittest.TestCase):
    def test_explicit_rejection_releases_reservation_and_allows_recovery(self):
        for code in (400, 401, 403, 404, 405, 410, 422):
            with self.subTest(code=code), patch.dict(os.environ, {'GITHUB_REPOSITORY':'owner/repo'}):
                store = MemoryStore(journal(1)); simulation = Simulation(store)
                simulation.dispatch = lambda ticket, ref: pool.Actions(store.state).dispatch(ticket, ref)
                with patch.object(pool.subprocess, 'run', return_value=MagicMock(
                        returncode=1, stdout='', stderr=f'gh: private response (HTTP {code})')) as request:
                    with self.assertRaises(pool.CoordinationError): simulation.run()
                ticket = store.state['tickets'][0]
                self.assertEqual(ticket['status'], 'completed')
                self.assertEqual(ticket['conclusion'], 'failure')
                self.assertEqual(ticket['dispatch_rejected_http'], code)
                self.assertIsNone(ticket['run']); request.assert_called_once()
                pool.recover(store.state, 2)
                self.assertEqual(store.state['courses']['0']['phase'], 'new')
                owner = MemoryStore({'schema':1, 'run_id':'99'})
                pool.claim_owner(journal(1, '100'), owner, open_pool=lambda _:store,
                    inspect_parent=lambda _: {'status':'completed'}, poll=lambda _:None)
                self.assertEqual(owner.state['run_id'], '100')

    def test_unknown_write_outcomes_remain_charged_without_replay(self):
        for outcome in (subprocess.TimeoutExpired('gh',120),
                        MagicMock(returncode=1, stdout='', stderr='gh: failed (HTTP 408)'),
                        MagicMock(returncode=1, stdout='', stderr='gh: failed (HTTP 429)'),
                        MagicMock(returncode=1, stdout='', stderr='gh: failed (HTTP 503)'),
                        MagicMock(returncode=0, stdout='{broken', stderr='')):
            with self.subTest(outcome=type(outcome).__name__), patch.dict(os.environ, {'GITHUB_REPOSITORY':'owner/repo'}):
                store=MemoryStore(journal(1)); simulation=Simulation(store)
                simulation.dispatch=lambda ticket,ref:pool.Actions(store.state).dispatch(ticket,ref)
                args={'side_effect':outcome} if isinstance(outcome,Exception) else {'return_value':outcome}
                with patch.object(pool.subprocess,'run',**args) as request:
                    with self.assertRaises(pool.CoordinationError):simulation.run()
                self.assertEqual(store.state['tickets'][0]['status'],'reserved')
                with self.assertRaises(ValueError):pool.recover(store.state,2)
                request.assert_called_once()

    def test_unchanged_polling_does_not_write_journal(self):
        store=MemoryStore(journal(1)); simulation=Simulation(store)
        simulation.dispatch=lambda *args:None
        with self.assertRaises(TimeoutError):
            pool.controller(store,simulation,attempt=1,clock=lambda:simulation.tick*30,
                sleep=simulation.sleep,timeout=180,get_work=simulation.work,acquire=lambda *args:None)
        self.assertEqual(store.version,1)  # Only the pre-dispatch reservation.
        self.assertEqual(store.state['tickets'][0]['status'],'reserved')

    def test_rejected_ticket_cannot_become_an_authorized_child(self):
        state=journal(1); ticket=pool.reserve(state,0,'prepare',1)
        ticket.update(status='completed',conclusion='failure',dispatch_rejected_http=422)
        run={'id':123,'display_title':f'icourse-stage-99-{ticket["nonce"]}',
             'head_sha':state['sha'],'path':'.github/workflows/'+pool.WORKFLOW,'status':'in_progress'}
        with patch.dict(os.environ,{'GITHUB_REPOSITORY':'owner/repo'}), \
             patch.object(pool,'read_json',return_value=[{'workflow_runs':[run]}]):
            with self.assertRaises(ValueError):pool.Actions(state).poll(state)
        self.assertIsNone(ticket['run'])

    def test_failure_audit_reports_ended_only_for_confirmed_terminal_tickets(self):
        from scripts import production_qwen as pipeline
        for rejected in (False,True):
            state=journal(1);ticket=pool.reserve(state,0,'prepare',1)
            if rejected:ticket.update(status='completed',conclusion='failure',dispatch_rejected_http=403)
            store=MemoryStore(state)
            with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'GITHUB_RUN_ID':'99','GITHUB_REPOSITORY':'owner/repo'}), \
                 patch.object(pool,'store_for',return_value=store),patch.object(pool,'PoolProgress',return_value=None), \
                 patch.object(pool,'controller',side_effect=pool.CoordinationError('github_dispatch','authorization',1)), \
                 patch.object(pipeline,'out',side_effect=lambda name:Path(tmp)/name), \
                 patch.object(pipeline,'write_outputs') as output:
                with self.assertRaises(pool.CoordinationError):pool.main()
                audit=json.loads((Path(tmp)/'pool-audit.json').read_bytes())
            self.assertEqual(audit['all_ended'],rejected)
            self.assertEqual(output.call_args.kwargs['all_ended'],rejected)


class PrivacyTests(unittest.TestCase):
    def test_selection_audit_has_only_anonymous_slots_and_status(self):
        row={'task_slot':0,'course_id':'private-id','status':'selected',
             'task':['private-id','private-title',{'sub_id':'private-sub',
               '_validation':{'date':'2026-09-29','rank':1,'teacher':'private-teacher'}}]}
        self.assertEqual(resource.selection_audit([row]), {'resource_only':True,'course_count':1,
            'courses':[{'task_slot':0,'status':'selected'}]})

    def test_encrypted_result_preserves_private_audit_public_output_filters_nested_fields(self):
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        from scripts.production_result_export import decrypt
        key=X25519PrivateKey.generate()
        public=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
        private=key.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption())
        row={'selection':{'course_id':'private-id'},'audit':{'task_slot':0,'status':'failed',
            'course_id':'private-id','date':'2026-09-29','seconds':2.5,'error_type':'PrivateCustomException',
            'error_code':'audio_decode_errors','phase':'audio_validation','audio_retained':True,
            'audio_diagnostics':{'source_transport':{'url':'private-url'},'audio_sha256':'private-hash'}}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(resource,'check_request',return_value=public), \
             patch.dict(os.environ,{'GITHUB_RUN_ID':'99'}), \
             patch.object(resource.pipeline,'out',side_effect=lambda name:Path(tmp)/name):
            resource.save_result(row,0)
            self.assertEqual(json.loads(decrypt((Path(tmp)/'resource-result.enc').read_bytes(),private,'99',0)),row)
            audit=json.loads((Path(tmp)/'resource-audit.json').read_bytes())
        self.assertEqual(audit,{'task_slot':0,'status':'failed','seconds':2.5,
            'error_code':'audio_decode_errors','phase':'audio_validation','audio_retained':True})


class UnknownDurationTests(unittest.TestCase):
    def test_production_prepare_preserves_unknown_duration_through_encrypted_checkpoint(self):
        import numpy as np
        from scripts import production_qwen as pipeline
        from test_qwen_production_pipeline import database
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{
                'RUNNER_TEMP':tmp,'GITHUB_RUN_ID':'99','COURSE_SLOT':'0','DB_ENCRYPTION_KEY':'k'*32,
                'QWEN_PRODUCTION_TASK':'true','AUTO_COURSE_TERMS':'false','PUBLISH_RESULTS':'false',
                'SHARD_MODE':'shared','GITHUB_ACTIONS':'false','POOL_ARTIFACTS':'false'}):
            root=pipeline.root();db=database(root/'fixture.db')
            raw=root/'source.raw';raw.write_bytes(np.zeros(4*16000,dtype=np.float32).tobytes())
            process=MagicMock();process.poll.return_value=0;process.returncode=0
            handle=SimpleNamespace(path=str(raw),process=process,timeline_preserved=True,stderr_chunks=[b'Duration: N/A'])
            scheduler=MagicMock();scheduler.audio_downloader.get.return_value=handle
            transcriber=MagicMock();transcriber.prepare_pcm_stream.return_value=[(0,4)]
            transcriber.last_audio_duration=4;transcriber.last_media_duration=None
            transcriber.last_vad_windows=[(0,4)];transcriber.last_prepare_stats={'stream_eof':True}
            lecture={'sub_id':'1','date':'2026-10-04'}
            with patch.object(pipeline,'artifact',return_value=False), \
                 patch.object(pipeline,'task_files',return_value=(db,'10','概率论',lecture)), \
                 patch.object(pipeline,'recover_preparation',return_value=None), \
                 patch.dict('sys.modules',{'main':SimpleNamespace(login_with_retry=lambda:None),
                    'src.pipeline.ppt_pipeline':SimpleNamespace(PPTPipeline=MagicMock()),
                    'src.api.icourse':SimpleNamespace(ICourseClient=MagicMock())}), \
                 patch('src.runtime.scheduler.Scheduler',return_value=scheduler), \
                 patch('src.ai.qwen_transcriber.QwenTranscriber',return_value=transcriber), \
                 patch('scripts.shared_asr_worker.initialize'),patch.object(pipeline,'write_outputs'):
                pipeline.prepare()
            files=pipeline.decode(root/'out'/'prepared.enc','prepared')
            spec=json.loads(files['specification.json'])
            self.assertEqual(spec['mode'],'sharded');self.assertIsNone(spec['media_seconds'])
            self.assertEqual(spec['audio_seconds'],4);self.assertTrue(spec['preparation_timing']['stream_eof'])
            self.assertIn('chunk-0.flac',files);transcriber.recognize_blocks.assert_not_called()

    def test_unknown_duration_with_decoder_eof_is_valid_but_truncation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'audio.raw'; raw.write_bytes(b'\0'*64000)
            handle=MagicMock(path=raw,stderr_chunks=[b'Duration: N/A'],timeline_preserved=True)
            handle.process.returncode=0; handle.stderr_done.wait.return_value=True
            handle.decode_error_counts={}; handle.media_transport=None
            diagnostic=collect_decode_diagnostics(handle)
            self.assertIsNone(diagnostic['media_seconds'])
            spec={'audio_seconds':1,'media_seconds':None,'audio_diagnostics':diagnostic}
            validate_prepared_audio(spec)
            for key,value in [('decode_return_code',1),('decode_interrupted',True),
                    ('decode_error_counts',{'premature_eof':1}),('stderr_complete',False),
                    ('source_transport',{'terminal_error_code':'upstream_premature_eof'})]:
                broken=copy.deepcopy(spec);broken['audio_diagnostics'][key]=value
                with self.assertRaises(ValueError):validate_prepared_audio(broken)
            spec['media_seconds']=0
            with self.assertRaises(ValueError):validate_prepared_audio(spec)
            spec['media_seconds']=None;spec['preparation_timing']={'stream_eof':False}
            with self.assertRaises(ValueError):validate_prepared_audio(spec)


class ConcurrencyAndCostTests(unittest.TestCase):
    def test_four_parallel_auth_flows_use_distinct_sessions(self):
        import requests
        from src.api.auth_recovery import authenticated_session
        barrier=threading.Barrier(4);sessions=[];lock=threading.Lock()
        class VPN:
            requires_webvpn_login=True
            def __init__(self):
                self.session=requests.Session()
                with lock:
                    self.number=len(sessions);sessions.append(self)
            def probe_login_service(self):barrier.wait(timeout=5)
            def login(self,student,password):self.session.cookies.set('test-token',str(self.number))
            def authenticate_icourse(self,student,password,strict):
                return self.session.cookies.get('test-token')==str(self.number) and strict
        with ThreadPoolExecutor(max_workers=4) as executor:
            results=list(executor.map(lambda n:authenticated_session(factory=VPN,
                student_id='same-synthetic-account',password='synthetic'),range(4)))
        self.assertEqual(len({id(v.session) for v in results}),4)
        self.assertEqual({v.session.cookies.get('test-token') for v in results},{'0','1','2','3'})
        for v in results:v.session.close()

    def test_shared_worker_measures_bundle_bytes_versus_actual_claims(self):
        import hashlib
        import io
        import numpy as np
        import soundfile as sf
        from scripts.shared_asr_worker import run_worker
        from src.pipeline.asr_queue import SharedQueue,initial_queue
        from test_shared_asr_queue import plan_for,decoded
        plan=plan_for(2);buf=io.BytesIO();sf.write(buf,np.zeros(960000,dtype=np.float32),16000,format='FLAC')
        blob=buf.getvalue();files={}
        for block in plan['blocks']:
            block['flac_sha256']=hashlib.sha256(blob).hexdigest();files[f'chunk-{block["chunk_id"]}.flac']=blob
        store=MemoryStore(initial_queue(plan));queue=SharedQueue(plan,store)
        block,token=queue.claim('old',1);queue.finish(block['chunk_id'],token,decoded(block))
        model=MagicMock()
        model.recognize_blocks.side_effect=lambda blocks,load,checkpoint,**kw:checkpoint([decoded(blocks[0])])
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.shared_asr_worker.shards.seal'), \
             patch('scripts.shared_asr_worker.shards.root',return_value=Path(tmp)):
            report=run_worker(plan,files,store,0,2,transcriber=model)
            audit=json.loads((Path(tmp)/'out'/'worker-audit.json').read_bytes())
            idle=run_worker(plan,files,store,0,3,transcriber=model)
        self.assertEqual(report['input_audio_bytes'],2*len(blob))
        self.assertEqual(report['claimed_audio_bytes'],len(blob))
        self.assertEqual(report['unused_input_audio_bytes'],len(blob))
        self.assertEqual(report['claimed_audio_blocks'],1)
        self.assertEqual(audit['input_audio_bytes'],report['input_audio_bytes'])
        self.assertEqual(idle['claimed_audio_bytes'],0)

    def test_stage_download_metrics_contain_only_sizes_and_timing(self):
        from scripts import production_pool_stage as stage
        from scripts import production_qwen as pipeline
        ticket={'stage':'asr','slot':0,'worker':1,'attempt':1}
        with tempfile.TemporaryDirectory() as tmp,patch.object(stage,'context',return_value=(ticket,dict(os.environ))), \
             patch.object(pipeline,'root',return_value=Path(tmp)), \
             patch.object(pipeline,'out',side_effect=lambda name:Path(tmp)/name), \
             patch.object(pipeline,'artifact') as artifact, \
             patch.object(stage.subprocess,'run',return_value=SimpleNamespace(returncode=0)):
            inbox=Path(tmp)/'inbox';inbox.mkdir();(inbox/'prepared.enc').write_bytes(b'x'*123)
            stage.execute();audit=json.loads((Path(tmp)/'stage-audit.json').read_bytes())
        self.assertEqual(audit['input_bundle_bytes'],123);self.assertEqual(audit['input_file_count'],1)
        self.assertGreaterEqual(audit['input_download_seconds'],0)
        self.assertNotIn('prepared.enc',json.dumps(audit));artifact.assert_called_once()
