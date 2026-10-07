"""Resource-only selection, integrity failures, bounded waits and workflow isolation."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml
from scripts import production_resource_fetch as resource


class ResourceFetchTests(unittest.TestCase):
    def test_plan_freezes_four_selections_with_recipient_export_and_private_names(self):
        import base64
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        from scripts.production_result_export import decrypt
        key=X25519PrivateKey.generate()
        public=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
        private=key.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption())
        rows=[{'task_slot':i,'course_id':str(i+1),'status':'selected',
               'task':[str(i+1),'private course',{'sub_id':str(i+11),'_validation':{'date':'2026-10-01'}}]} for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{
                'COURSE_IDS':'1,2,3,4','GITHUB_RUN_ATTEMPT':'1','GITHUB_RUN_ID':'123',
                'RECIPIENT_PUBLIC_KEY':base64.b64encode(public).decode()}), \
             patch.object(resource.pipeline,'root',return_value=Path(tmp)), \
             patch.object(resource.pipeline,'out',side_effect=lambda name:Path(tmp)/name), \
             patch.object(resource,'load_remote') as load,patch.object(resource,'Database') as db, \
             patch.object(resource,'authenticated_session') as login,patch.object(resource,'select_latest',return_value=rows), \
             patch.object(resource.shards,'seal') as seal,patch.object(resource.pipeline,'write_outputs') as outputs:
            resource.plan()
            self.assertEqual(seal.call_args.args[1],'resource-selection')
            self.assertEqual(outputs.call_args.kwargs['tasks']['include'],[{'task_slot':i} for i in range(4)])
            result=json.loads(decrypt((Path(tmp)/'selection-export.enc').read_bytes(),private,'123',255))
            self.assertEqual(result['selections'],rows)
            self.assertNotIn('private course',(Path(tmp)/'selection-audit.json').read_text())
            load.assert_called_once();db.return_value.conn.close.assert_called_once()
            login.return_value.session.close.assert_called_once()

    def test_one_unavailable_course_does_not_hide_other_selections(self):
        def latest(client, db, course):
            if course == '2': raise ValueError('private provider message')
            return (course, 'private title', {'sub_id': course+'1', '_validation': {'date':'2026-10-01'}}), 'private teacher'
        with patch.object(resource.pipeline, 'latest_validation_task', side_effect=latest) as select:
            rows=resource.select_latest(object(), object(), ['1','2','3','4'])
        self.assertEqual(select.call_count,4)
        self.assertEqual([r['status'] for r in rows],['selected','selection_failed','selected','selected'])
        public=json.dumps(resource.selection_audit(rows))
        self.assertNotIn('private',public)
        self.assertNotIn('task"',public)

    def test_rerun_refused_before_authentication_or_audio(self):
        with patch.dict(os.environ, {'GITHUB_RUN_ATTEMPT':'2'}), patch.object(resource, 'authenticated_session') as login:
            with self.assertRaises(ValueError): resource.plan()
            with self.assertRaises(ValueError): resource.fetch()
            login.assert_not_called()

    def test_idle_and_total_deadlines_stop_without_restarting_download(self):
        for elapsed, reason in ((resource.PREPARE_IDLE_TIMEOUT+1,'stalled'),
                                (resource.PREPARE_STREAM_TIMEOUT+1,'deadline')):
            with tempfile.TemporaryDirectory() as tmp:
                handle=MagicMock(path=str(Path(tmp)/'missing.raw'));handle.process.poll.return_value=None
                with self.assertRaisesRegex(TimeoutError, reason):
                    resource.wait_audio(handle,clock=MagicMock(side_effect=[0,elapsed]),sleep=MagicMock())

    def test_resource_fetch_preserves_failed_audio_and_never_marks_it_complete(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                selected={'task_slot':0,'course_id':'1','status':'selected',
                          'task':['1','private course',{'sub_id':'11','_validation':{'date':'2026-10-01'}}]}
                rows={'selections.json':json.dumps([selected]).encode()}
                downloader=MagicMock();handle=MagicMock();downloader.get.return_value=handle
                def retain(h,spec,files):
                    spec.update(audio_seconds=1 if invalid else 100,media_seconds=1000 if invalid else 100,
                                audio_diagnostics={'stderr_complete':True,'decode_return_code':0,
                                  'pcm_sample_aligned':True,'decode_error_counts':{}})
                    files['lecture.flac']=b'actual retained audio'
                with patch.object(resource,'check_request'), patch.dict(os.environ,{'COURSE_SLOT':'0'}), \
                     patch.object(resource.pipeline,'artifact'),patch.object(resource.shards,'unseal',return_value=rows), \
                     patch.object(resource.pipeline,'root',return_value=Path(tmp)), \
                     patch.object(resource.pipeline,'out',side_effect=lambda name:Path(tmp)/name), \
                     patch.object(resource,'authenticated_session') as login, \
                     patch.object(resource,'AudioDownloader',return_value=downloader), \
                     patch.object(resource,'wait_audio'), \
                     patch.object(resource.pipeline,'retain_prepared_audio',side_effect=retain), \
                     patch.object(resource.shards,'seal') as seal,patch.object(resource,'save_result') as save:
                    self.assertEqual(resource.fetch(),not invalid)
                downloader.schedule.assert_called_once()
                self.assertTrue(downloader.schedule.call_args.kwargs['preserve_timestamps'])
                self.assertEqual(downloader.schedule.call_args.args[1:],('1','11'))
                downloader.shutdown.assert_called_once();login.return_value.session.close.assert_called_once()
                self.assertEqual(seal.call_args.args[1],'prepared')
                self.assertEqual(seal.call_args.kwargs['slot'],0)
                self.assertEqual(seal.call_args.args[0]['lecture.flac'],b'actual retained audio')
                audit=save.call_args.args[0]['audit']
                self.assertEqual(audit['status'],'failed' if invalid else 'complete')
                self.assertTrue(audit['audio_retained'])
                self.assertEqual(audit['model_calls'],0)
                self.assertFalse(audit['publication']);self.assertFalse(audit['emailed'])

    def test_unavailable_selection_is_reported_without_authentication(self):
        row={'task_slot':0,'course_id':'1','status':'selection_failed','error_type':'ValueError'}
        with patch.object(resource,'check_request'),patch.dict(os.environ,{'COURSE_SLOT':'0'}), \
             patch.object(resource.pipeline,'root',return_value=Path('/tmp/resource-test')), \
             patch.object(resource.pipeline,'artifact'),patch.object(resource.shards,'unseal',return_value={'selections.json':json.dumps([row]).encode()}), \
             patch.object(resource,'save_result') as save,patch.object(resource,'authenticated_session') as login:
            self.assertFalse(resource.fetch())
            login.assert_not_called();save.assert_called_once()

    def test_manual_workflow_has_one_acquisition_slot_and_no_model_or_publish_secrets(self):
        path=Path(__file__).resolve().parents[1]/'.github/workflows/qwen_production_resources.yml'
        workflow=yaml.safe_load(path.read_text())
        self.assertEqual(workflow['jobs']['fetch']['strategy']['max-parallel'],2)
        self.assertFalse(workflow['jobs']['fetch']['strategy']['fail-fast'])
        self.assertEqual(workflow['permissions']['contents'],'read')
        text=path.read_text()
        for secret in ('DASHSCOPE_API_KEY','DOUBAO_ASR_API_KEY','DEEPSEEK_API_KEY','SMTP_PASSWORD'):
            self.assertNotIn(secret,text)
        for key in ('plan','fetch'):
            commands=[s.get('run','') for s in workflow['jobs'][key]['steps']]
            self.assertFalse(any(' worker' in c or ' gather' in c or ' publish' in c for c in commands))


class FrozenResourceTests(unittest.TestCase):
    def test_slot_subset_is_explicit_and_unique(self):
        for value in ('0,0','4','-1','0, 3',''):
            with patch.dict(os.environ,{'RESOURCE_SLOTS':value}):
                if value:self.assertRaises(ValueError,resource.requested_slots)
                else:self.assertEqual(resource.requested_slots(),[0,1,2,3])
        with patch.dict(os.environ,{'RESOURCE_SLOTS':'0,3'}):self.assertEqual(resource.requested_slots(),[0,3])

    def test_frozen_selection_rebinds_without_login_or_latest_scan(self):
        rows=[{'task_slot':i,'course_id':str(i+1),'status':'selected',
               'task':[str(i+1),'private',{'sub_id':str(i+11),'_validation':{'date':'2026-09-29'}}]} for i in range(4)]
        env={'GITHUB_RUN_ID':'200','GITHUB_REPOSITORY':'owner/repo'}
        info={'status':'completed','path':'.github/workflows/qwen_production_resources.yml'}
        def unseal(*args,**kwargs):
            self.assertEqual(os.environ['GITHUB_RUN_ID'],'100')
            return {'selections.json':json.dumps(rows).encode()}
        with patch.dict(os.environ,env),patch.object(resource.pipeline,'root',return_value=Path('/tmp/frozen-resource-test')),patch.object(resource.subprocess,'check_output',return_value=json.dumps(info).encode()), \
             patch.object(resource.pipeline,'artifact') as artifact,patch.object(resource.shards,'unseal',side_effect=unseal), \
             patch.object(resource,'authenticated_session') as login,patch.object(resource,'select_latest') as select:
            frozen=resource.frozen_resource_selection('100')
            self.assertEqual(os.environ['GITHUB_RUN_ID'],'200')
            self.assertEqual([r['task'][2]['_validation']['selection_source_run_id'] for r in frozen],['100']*4)
            self.assertEqual(artifact.call_args.kwargs['run'],'100')
            login.assert_not_called();select.assert_not_called()

    def test_active_or_other_workflow_source_is_rejected_before_download(self):
        for status,path in [('in_progress','.github/workflows/qwen_production_resources.yml'),('completed','other.yml')]:
            with patch.dict(os.environ,{'GITHUB_RUN_ID':'200','GITHUB_REPOSITORY':'owner/repo'}), \
                 patch.object(resource.subprocess,'check_output',return_value=json.dumps({'status':status,'path':path}).encode()), \
                 patch.object(resource.pipeline,'artifact') as artifact:
                with self.assertRaises(ValueError):resource.frozen_resource_selection('100')
                artifact.assert_not_called()
