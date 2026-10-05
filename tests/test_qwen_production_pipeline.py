"""Real SQLite/encryption/Git integration, with no course, model or SMTP calls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import yaml
from scripts.qwen_sharding import build_audio_plan, fingerprint, validate_plan
from scripts.production_db import snapshot, lecture_snapshot, merge_lecture
from scripts import production_qwen as pipeline
from src.data.database import Database
from src.pipeline.prepared_lecture import assemble_material, cached_material
from test_lecture_quality_gate import _load_runner_class

ROOT = Path(__file__).resolve().parents[1]


def fixture(slot=0, run='99'):
    plan = build_audio_plan({'selection': {'course_id':'10','sub_id':'1'}, 'audio_seconds':600,
        'full_chunks':[{'start':0,'end':120},{'start':119,'end':239}],
        'recognition_terms':['条件期望'], 'vad_windows':[[0,120],[119,239]]},
        reference={'pipeline':'production'}, course_slot=slot, run_id=run,
        audio_sha256='a'*64, production=True)
    results=[]
    for shard in plan['shards']:
        rows=[dict(plan['blocks'][i], text=('定义随机变量的分布与期望。'*70 if i==0 else '计算方差并使用条件概率公式。'*70))
              for i in shard['chunk_ids']]
        results.append({'plan_hash':fingerprint(plan),'shard_id':shard['shard_id'],
                        'complete':True,'chunks':rows,'attempts':[]})
    return plan, results


def database(path, *, summary=None):
    db=Database(str(path));db.upsert_course('10','概率论','教师')
    db.insert_lecture('1','10','课次','2026-10-04')
    if summary: db.update_summary('1',summary,'test');db.mark_processed('1')
    return db


class PreparedLectureTests(unittest.TestCase):
    def test_complete_original_blocks_pass_normal_runner_and_persist(self):
        Runner=_load_runner_class();plan,results=fixture()
        material=assemble_material(plan,results,media_seconds=600)
        with tempfile.TemporaryDirectory() as tmp:
            db=database(Path(tmp)/'course.db')
            summarizer=MagicMock();summarizer.summarize.return_value=('完整摘要','test')
            transcriber=MagicMock();scheduler=MagicMock()
            runner=Runner(None,db,scheduler,transcriber,summarizer,MagicMock())
            with patch('src.runtime.config.DOUBAO_ASR_API_KEY',''):
                result=runner.run('10','概率论',{'sub_id':'1'},prepared_asr=material,review_state={},checkpoint=lambda:None)
            row=db.get_lecture('1')
            self.assertEqual(result,'完整摘要');self.assertEqual(row['transcript'],material['transcript'])
            self.assertEqual(row['summary'],'完整摘要');self.assertIsNotNone(row['processed_at'])
            self.assertIsNone(row['emailed_at']);transcriber.transcribe_tail.assert_not_called()
            scheduler.prefetch_lecture.assert_not_called();db.conn.close()

    def test_missing_or_duplicate_shard_cannot_bypass_quality_boundary(self):
        plan,results=fixture()
        for invalid in (results[:1], [results[0],results[0]], [dict(results[0],complete=False),results[1]]):
            with self.assertRaises(ValueError): assemble_material(plan,invalid)

    def test_summary_failure_then_retry_reuses_original_timestamps_and_asr(self):
        Runner=Runner=_load_runner_class();plan,results=fixture();material=assemble_material(plan,results,media_seconds=600)
        with tempfile.TemporaryDirectory() as tmp:
            db=database(Path(tmp)/'course.db');summarizer=MagicMock()
            summarizer.summarize.side_effect=[RuntimeError('temporary'),('恢复摘要','test')]
            transcriber=MagicMock();runner=Runner(None,db,MagicMock(),transcriber,summarizer,MagicMock())
            with patch('src.runtime.config.DOUBAO_ASR_API_KEY',''):
                with self.assertRaises(RuntimeError):
                    runner.run('10','概率论',{'sub_id':'1'},prepared_asr=material,review_state={},checkpoint=lambda:None)
                row=db.get_lecture('1');self.assertIsNotNone(row['transcript']);self.assertIsNone(row['processed_at'])
                result=runner.run('10','概率论',{'sub_id':'1'},prepared_asr=material,review_state={},checkpoint=lambda:None)
            self.assertEqual(result,'恢复摘要');self.assertEqual(db.get_lecture('1')['error_count'],0)
            transcriber.transcribe_tail.assert_not_called();self.assertEqual(transcriber.last_chunks,material['full_chunks'])
            db.conn.close()

    def test_existing_summary_is_preserved_and_not_regenerated(self):
        Runner=_load_runner_class();plan,results=fixture();material=assemble_material(plan,results)
        with tempfile.TemporaryDirectory() as tmp:
            db=database(Path(tmp)/'course.db',summary='历史摘要');db.mark_emailed('1')
            llm=MagicMock();runner=Runner(None,db,MagicMock(),MagicMock(),llm,MagicMock())
            self.assertIsNone(runner.run('10','概率论',{'sub_id':'1'},prepared_asr=material))
            llm.summarize.assert_not_called();self.assertEqual(db.get_lecture('1')['summary'],'历史摘要');db.conn.close()

    def test_silent_complete_audio_uses_existing_no_content_gate(self):
        Runner=_load_runner_class();plan,results=fixture()
        plan['audio_seconds']=1800
        for result in results:
            result['plan_hash']=fingerprint(plan)
            for row in result['chunks']:row['text']=''
        material=assemble_material(plan,results,media_seconds=1800)
        with tempfile.TemporaryDirectory() as tmp:
            db=database(Path(tmp)/'course.db');llm=MagicMock()
            runner=Runner(None,db,MagicMock(),MagicMock(),llm,MagicMock())
            self.assertIsNone(runner.run('10','概率论',{'sub_id':'1'},prepared_asr=material))
            self.assertIsNotNone(db.get_lecture('1')['processed_at']);self.assertIsNone(db.get_lecture('1')['summary'])
            llm.summarize.assert_not_called();db.conn.close()

    def test_production_queue_accepts_more_than_five_total_tasks_and_uncapped_audio(self):
        plan,_=fixture(slot=255);validate_plan(plan)
        plan['course_slot']=256
        with self.assertRaises(ValueError): validate_plan(plan)
        plan,_=fixture();plan['audio_seconds']=10900
        validate_plan(plan)


class ClassroomSelectionTests(unittest.TestCase):
    def test_latest_real_playback_skips_holidays_future_and_deleted_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = database(Path(tmp)/'history.db', summary='历史摘要')
            db.insert_lecture('9', '10', '删除课次', '2026-10-05')
            with db.conn: db.conn.execute("UPDATE lectures SET deleted_at='removed' WHERE sub_id='9'")
            client = MagicMock()
            client.get_course_detail.return_value = {'title':'概率论', 'teacher':'教师', 'lectures':[
                {'sub_id':'20','date':'2026-10-06'}, {'sub_id':'9','date':'2026-10-05'},
                {'sub_id':'8','date':'2026-10-05','sub_title':'第6节','has_playback':True},
                {'sub_id':'1','date':'2026-10-04','has_playback':False},
                {'sub_id':'4','date':'2026-02-30'}, {'sub_id':'bad','date':'2026-10-05'}]}
            client.get_video_url.side_effect = [None, 'https://private.example/recording']
            task, _ = pipeline.latest_validation_task(client, db, '10', today='2026-10-05')
            self.assertEqual(task[2]['sub_id'], '1')
            self.assertEqual(task[2]['_validation']['skipped_unavailable'], 1)
            self.assertEqual([c.args for c in client.get_video_url.call_args_list], [('10','8'),('10','1')])
            self.assertNotIn('https:', json.dumps(task))
            self.assertEqual(db.get_lecture('1')['summary'], '历史摘要')
            db.conn.close()

    def test_no_recording_is_not_an_authorization_to_process_other_courses(self):
        client = MagicMock(); client.get_course_detail.return_value = {'title':'高代', 'lectures':[
            {'sub_id':'1', 'date':'2026-10-01'}]}
        client.get_video_url.return_value = None
        db = MagicMock(); db.get_lecture.return_value = None
        with self.assertRaises(ValueError):
            pipeline.latest_validation_task(client, db, '38404', today='2026-10-05')
        client.get_course_detail.assert_called_once_with('38404')

    def test_validation_requires_both_side_effect_flags_disabled(self):
        for publish, email in [('true','false'), ('false','true'), ('','false')]:
            with patch.dict(os.environ, {'VALIDATION_COURSE_ID':'38404','COURSE_IDS':'38404,40329',
                                         'PUBLISH_RESULTS':publish,'SEND_EMAIL':email}):
                with self.assertRaises(ValueError): pipeline.validation_course()
        with patch.dict(os.environ, {'VALIDATION_COURSE_ID':'38404','COURSE_IDS':'38404,40329',
                                     'PUBLISH_RESULTS':'false','SEND_EMAIL':'false'}):
            self.assertEqual(pipeline.validation_course(), '38404')

    def test_workflow_registration_never_processes_classrooms_on_push(self):
        workflow=yaml.safe_load((ROOT/'.github/workflows/parallel_pilot.yml').read_text())
        self.assertIn("github.event_name != 'push'",workflow['jobs']['plan']['if'])
        self.assertIn("needs.plan.result == 'success'",workflow['jobs']['lecture']['if'])
        self.assertEqual(workflow['jobs']['register']['permissions'],{})
        self.assertEqual(workflow['on']['push']['paths'],['.github/workflows/parallel_pilot.yml'])

    def test_validation_queue_forces_fresh_asr_and_retains_original_history(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'RUNNER_TEMP':tmp,'GITHUB_RUN_ID':'99','COURSE_SLOT':'0','DB_ENCRYPTION_KEY':'k'*32,
            'QWEN_PRODUCTION_TASK':'true','GITHUB_RUN_ATTEMPT':'1','VALIDATION_COURSE_ID':'10',
            'COURSE_IDS':'10','PUBLISH_RESULTS':'false','SEND_EMAIL':'false'}):
            root=pipeline.root(); db=database(root/'original.db', summary='历史摘要')
            original=snapshot(db,root/'original-snapshot.db'); db.conn.close()
            def load(path): path.write_bytes(original)
            client=MagicMock();client.get_course_detail.return_value={'title':'概率论','teacher':'教师',
                'lectures':[{'sub_id':'1','date':'2026-10-04'}]}
            client.get_video_url.return_value='https://private.example/recording'
            fake_main=SimpleNamespace(login_with_retry=lambda:None, _enumerate_lectures=MagicMock(),
                                      _crawl_semester_catalog=MagicMock())
            with patch.dict('sys.modules', {'main':fake_main}), \
                 patch.object(pipeline,'artifact',return_value=False), patch.object(pipeline,'load_remote',side_effect=load), \
                 patch('src.api.icourse.ICourseClient',return_value=client), patch.object(pipeline,'write_outputs') as outputs:
                pipeline.plan()
            saved=pipeline.decode(root/'out'/'queue.enc','queue')
            tasks=json.loads(saved['queue.json']);self.assertEqual(len(tasks),1)
            self.assertEqual(tasks[0][2]['sub_id'],'1')
            (root/'fresh.db').write_bytes(saved['database.db']);db=Database(str(root/'fresh.db'))
            self.assertIsNone(db.get_lecture('1')['summary']);db.conn.close()
            (root/'preserved.db').write_bytes(saved['history.db']);db=Database(str(root/'preserved.db'))
            self.assertEqual(db.get_lecture('1')['summary'],'历史摘要');db.conn.close()
            fake_main._enumerate_lectures.assert_not_called();fake_main._crawl_semester_catalog.assert_not_called()
            outputs.assert_called_once_with(tasks={'include':[{'task_slot':0}]},count=1)
            audit=(root/'out'/'validation-selection.json').read_text()
            self.assertNotIn('sub_id',audit);self.assertNotIn('https:',audit)


class PrivateSummaryExportTests(unittest.TestCase):
    def test_summary_export_is_bound_to_recipient_and_source(self):
        from scripts.production_result_export import encrypt,decrypt
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
        private=X25519PrivateKey.generate()
        raw=private.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption())
        public=private.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
        payload='私有摘要'.encode();blob=encrypt(payload,public,'99',0)
        self.assertNotIn(payload,blob);self.assertEqual(decrypt(blob,raw,'99',0),payload)
        for invalid,run,slot in [(blob,'100',0),(blob,'99',1),(blob[:-1]+bytes([blob[-1]^1]),'99',0)]:
            with self.assertRaises(Exception):decrypt(invalid,raw,run,slot)

    def test_export_reads_exact_completed_summary_without_transcript(self):
        from scripts.production_result_export import summary_payload
        with tempfile.TemporaryDirectory() as tmp:
            db=database(Path(tmp)/'source.db',summary='历史摘要')
            db.update_transcript('1','原课堂全文')
            files={'database.db':snapshot(db,Path(tmp)/'snapshot.db'),
                   'specification.json':pipeline.shards.encoded({'course_id':'10','course_title':'概率论',
                    'lecture':{'sub_id':'1','date':'2026-10-04'},'plan':{'recognition_terms':['条件期望']}}),
                   'review.json':b'{}'}
            db.conn.close();payload=summary_payload(files)
            self.assertEqual(payload['summary'],'历史摘要')
            self.assertEqual(payload['recognition_terms'],['条件期望'])
            self.assertNotIn('原课堂全文',json.dumps(payload,ensure_ascii=False))
            spec=json.loads(files['specification.json']);spec['lecture']['sub_id']='2'
            files['specification.json']=pipeline.shards.encoded(spec)
            with self.assertRaises(ValueError):summary_payload(files)

    def test_export_workflow_has_no_credentials_for_models_or_mail(self):
        workflow=yaml.safe_load((ROOT/'.github/workflows/qwen_production_validation.yml').read_text())
        job=workflow['jobs']['export']
        self.assertEqual(job['permissions'],{'contents':'read','actions':'read'})
        self.assertEqual([k for k in job['env'] if 'secrets.' in job['env'][k]],['DB_ENCRYPTION_KEY'])
        self.assertIn("github.event_name != 'workflow_dispatch'",workflow['jobs']['validate']['if'])


class CloudLedgerTests(unittest.TestCase):
    def test_reserved_unknown_call_is_not_repeated_on_resume(self):
        from src.ai.qwen_review_ledger import review_prepared,validate_ledger
        intervals=[dict(start_ms=i*10000,end_ms=i*10000+1000,quote_start_ms=i*10000,
                        quote_end_ms=i*10000+1000,text='疑点',chunk_id=i) for i in range(2)]
        material={'full_chunks':[],'vad_windows':[],'audio_path':'unused','recognition_terms':[]}
        ledger={'intervals':intervals};checkpoints=[]
        save=lambda:checkpoints.append(copy.deepcopy(ledger))
        with patch('src.ai.qwen_review_ledger.config.DOUBAO_ASR_API_KEY','fake'), \
             patch('src.ai.doubao_asr.rescue_intervals_pcm',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): review_prepared(material,[],MagicMock(),ledger,save)
        self.assertEqual(checkpoints[-1]['attempts'][0]['status'],'reserved')
        self.assertEqual(ledger['seconds'],1)
        def rescue(path,key,windows,**kwargs):
            self.assertEqual(windows,[intervals[1]])
            return [(windows[0],[{'start_ms':10000,'end_ms':11000,'text':'复核版本'}])],1,False
        with patch('src.ai.qwen_review_ledger.config.DOUBAO_ASR_API_KEY','fake'), \
             patch('src.ai.doubao_asr.rescue_intervals_pcm',side_effect=rescue) as transport:
            result=review_prepared(material,[],MagicMock(),ledger,save)
            review_prepared(material,[],MagicMock(),ledger,save)
        self.assertEqual(transport.call_count,1);self.assertEqual(ledger['seconds'],2)
        self.assertEqual(result['uncertain_calls'],1);validate_ledger(ledger)

    def test_tampered_or_excessive_quota_is_rejected(self):
        from src.ai.qwen_review_ledger import validate_ledger
        for ledger in ({'seconds':1},{'attempts':[{'seconds':-1,'interval':{'start_ms':0,'end_ms':1000},'status':'reserved'}], 'seconds':-1}):
            with self.assertRaises(ValueError):validate_ledger(ledger)


class ScopedPublicationTests(unittest.TestCase):
    def test_snapshot_includes_wal_and_only_one_lecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);db=database(tmp/'source.db');db.insert_lecture('2','10','另一节','2026-10-04')
            db.write_meta('qwen_pipeline:1','{}');db.write_meta('qwen_pipeline:2','{}')
            data=lecture_snapshot(db,tmp/'delta.db','10','1')
            self.assertTrue(data.startswith(b'SQLite format 3'))
            with sqlite3.connect(tmp/'delta.db') as conn:
                self.assertEqual(conn.execute('SELECT sub_id FROM lectures').fetchall(),[('1',)])
                self.assertEqual(conn.execute('SELECT key FROM meta').fetchall(),[('qwen_pipeline:1',)])
            db.conn.close()

    def test_newer_history_and_tombstones_win_but_matching_mail_receipt_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);local=database(tmp/'local.db',summary='过时摘要');remote=database(tmp/'remote.db',summary='新的摘要')
            local.conn.close();remote.conn.close()
            merge_lecture(tmp/'local.db',tmp/'remote.db','10','1')
            db=Database(str(tmp/'remote.db'));self.assertEqual(db.get_lecture('1')['summary'],'新的摘要');db.conn.close()
            db=Database(str(tmp/'local.db'));db.mark_emailed('1');db.conn.close()
            merge_lecture(tmp/'local.db',tmp/'remote.db','10','1')
            db=Database(str(tmp/'remote.db'));self.assertIsNotNone(db.get_lecture('1')['emailed_at'])
            db.suppress_lectures('10',['1']);db.conn.close()
            merge_lecture(tmp/'local.db',tmp/'remote.db','10','1')
            db=Database(str(tmp/'remote.db'));self.assertIsNone(db.get_lecture('1')['summary']);self.assertIsNotNone(db.get_lecture('1')['deleted_at']);db.conn.close()

    def test_cross_scope_delta_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);local=database(tmp/'local.db');remote=database(tmp/'remote.db')
            local.insert_lecture('2','10','另一个课次','2026-10-04');local.conn.close();remote.conn.close()
            with self.assertRaises(ValueError):merge_lecture(tmp/'local.db',tmp/'remote.db','10','1')

    def test_selected_pending_ppt_and_recovery_metadata_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);local=database(tmp/'local.db');remote=database(tmp/'remote.db')
            for db in (local,remote):db.insert_ppt_pages_pending('1',[{'page_num':1,'created_sec':1}])
            local.update_ppt_page('1',1,'有效OCR','done');local.write_meta('qwen_pipeline:1','{"review":{"seconds":0}}')
            local.conn.close();remote.conn.close();merge_lecture(tmp/'local.db',tmp/'remote.db','10','1')
            db=Database(str(tmp/'remote.db'));self.assertEqual(db.get_done_ppt_pages('1')[0]['text'],'有效OCR')
            self.assertIsNotNone(db.read_meta('qwen_pipeline:1'));db.conn.close()

    def test_real_encrypted_git_publish_conflict_refresh_preserves_both_lessons(self):
        from scripts import production_db as storage
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);bare=tmp/'remote.git'
            subprocess.run(['git','init','--bare','-q',str(bare)],check=True)
            a=database(tmp/'a.db');a.update_summary('1','第一堂摘要','test');a.mark_processed('1');a.conn.close()
            b=Database(str(tmp/'b.db'));b.upsert_course('20','高代','教师');b.insert_lecture('2','20','课次','2026-10-04')
            b.update_summary('2','第二堂摘要','test');b.mark_processed('2');b.conn.close()
            actual=storage.command;injected=False
            def git(args,**kwargs):
                nonlocal injected
                args=[str(bare) if x=='https://github.com/test/repo.git' else x for x in args]
                if args[:2]==['git','push'] and not injected:
                    injected=True;storage.publish(tmp/'b.db','20','2')
                return actual(args,**kwargs)
            with patch.dict(os.environ,{'GITHUB_REPOSITORY':'test/repo','DB_ENCRYPTION_KEY':'k'*32,'COURSE_IDS':'10,20'}), \
                 patch.object(storage,'command',side_effect=git):
                storage.publish(tmp/'a.db','10','1');storage.load_remote(tmp/'verified.db')
            db=Database(str(tmp/'verified.db'))
            self.assertEqual(db.get_lecture('1')['summary'],'第一堂摘要');self.assertEqual(db.get_lecture('2')['summary'],'第二堂摘要')
            db.conn.close();self.assertTrue(injected)


class FormalWorkflowTests(unittest.TestCase):
    def test_scheduled_entry_uses_same_bounded_pipeline_and_retains_cron(self):
        caller=yaml.load((ROOT/'.github/workflows/check.yml').read_text(),Loader=yaml.BaseLoader)
        formal=yaml.load((ROOT/'.github/workflows/parallel_pilot.yml').read_text(),Loader=yaml.BaseLoader)
        child=yaml.load((ROOT/'.github/workflows/qwen_production_lecture.yml').read_text(),Loader=yaml.BaseLoader)
        self.assertEqual([v['cron'] for v in caller['on']['schedule']],['7 9 * * *','7 12 * * *'])
        self.assertEqual(caller['jobs']['check']['uses'],'./.github/workflows/parallel_pilot.yml')
        self.assertIn('concurrency',caller);self.assertIn('concurrency',formal);self.assertNotIn('concurrency',child)
        self.assertEqual(caller['jobs']['check']['with']['caller_holds_lock'],'true')
        self.assertIn('icourse-subrun-',formal['concurrency']['group'])
        self.assertEqual(int(formal['jobs']['lecture']['strategy']['max-parallel'])*int(child['jobs']['asr']['strategy']['max-parallel']),15)
        self.assertEqual(child['jobs']['gather']['needs'],['prepare','asr'])
        self.assertEqual(child['jobs']['publish']['needs'],'gather')
        self.assertEqual(formal['on']['workflow_dispatch']['inputs']['send_email']['default'],'false')
        self.assertEqual(formal['on']['workflow_dispatch']['inputs']['publish_results']['default'],'false')
        self.assertNotIn('asr',str(child['jobs']['publish']['steps']))
        self.assertNotIn('toJSON(secrets)',str(formal))
        self.assertNotIn('SECRETS_CONTEXT',str(child))
        self.assertNotIn('SMTP_PASSWORD',child['jobs']['gather']['env'])
        self.assertEqual(child['jobs']['gather']['env']['DEEPSEEK_API_KEY'],'${{ secrets.DEEPSEEK_API_KEY }}')



class EncryptedStageTests(unittest.TestCase):
    def test_encrypted_finalization_retries_summary_without_decoding_or_resetting_quota(self):
        Runner=_load_runner_class();plan,results=fixture();material=assemble_material(plan,results,media_seconds=600)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{
            'RUNNER_TEMP':tmp,'GITHUB_RUN_ID':'99','COURSE_SLOT':'0','DB_ENCRYPTION_KEY':'k'*32,
            'QWEN_PRODUCTION_TASK':'true','GITHUB_ACTIONS':'false','AUTO_COURSE_TERMS':'false'}):
            root=pipeline.root();(root/'inbox').mkdir()
            db=database(root/'fixture.db');payload=snapshot(db,root/'snapshot.db');db.conn.close()
            interval={'start_ms':0,'end_ms':1000,'quote_start_ms':0,'quote_end_ms':1000,'text':'疑点'}
            review={'complete':True,'seconds':1,'attempts':[{'interval':interval,'seconds':1,'status':'reserved'}],
                    'material':{'variants':[],'uncertain_calls':1}}
            spec={'course_id':'10','course_title':'概率论','lecture':{'sub_id':'1',
                  '_validation':{'date':'2026-10-04','skipped_unavailable':2}},'mode':'cached',
                  'material':material,'review':review}
            pipeline.encode({'specification.json':pipeline.shards.encoded(spec),'database.db':payload},
                            'prepared',root/'inbox'/'prepared.enc')
            llm=MagicMock();llm.summarize.side_effect=[RuntimeError('temporary'),('恢复后的摘要','test')]
            with patch.dict('sys.modules', {'src.pipeline.lecture_runner':SimpleNamespace(LectureRunner=Runner)}), \
                 patch.object(pipeline,'artifact',return_value=False), \
                 patch('src.ai.summarizer.Summarizer',return_value=llm), \
                 patch('src.runtime.config.DOUBAO_ASR_API_KEY',''), \
                 patch.object(pipeline.shards,'command') as commands:
                with self.assertRaises(RuntimeError):pipeline.gather()
            saved=pipeline.decode(root/'out'/'state.enc','state')
            self.assertEqual(json.loads(saved['review.json'])['seconds'],1)
            recovery=root/'after-failure.db';recovery.write_bytes(saved['database.db'])
            db=Database(str(recovery));self.assertIsNotNone(db.get_lecture('1')['transcript'])
            self.assertIsNone(db.get_lecture('1')['summary']);db.conn.close();commands.assert_not_called()
            def restore(name,target,**kwargs):
                target.mkdir(exist_ok=True)
                (target/'state.enc').write_bytes((root/'out'/'state.enc').read_bytes())
                return True
            with patch.dict('sys.modules', {'src.pipeline.lecture_runner':SimpleNamespace(LectureRunner=Runner)}), \
                 patch.object(pipeline,'artifact',side_effect=restore), \
                 patch('src.ai.summarizer.Summarizer',return_value=llm), \
                 patch('src.runtime.config.DOUBAO_ASR_API_KEY',''), \
                 patch.object(pipeline.shards,'command') as commands:
                pipeline.gather()
            saved=pipeline.decode(root/'out'/'state.enc','state');commands.assert_not_called()
            self.assertEqual(json.loads(saved['review.json'])['seconds'],1)
            recovery.write_bytes(saved['database.db']);db=Database(str(recovery))
            self.assertEqual(db.get_lecture('1')['summary'],'恢复后的摘要')
            self.assertIsNotNone(db.get_lecture('1')['processed_at'])
            self.assertIsNone(db.get_lecture('1')['emailed_at'])
            audit=json.loads(db.read_meta('qwen_pipeline:1'));self.assertTrue(audit['complete'])
            self.assertNotIn('material',audit);db.conn.close()
            audit=json.loads((root/'out'/'validation-result.json').read_bytes())
            self.assertTrue(audit['processed']);self.assertTrue(audit['asr_complete'])
            self.assertFalse(audit['emailed']);self.assertEqual(audit['review_seconds'],1)
            self.assertEqual(audit['summary_chars'],len('恢复后的摘要'))
            self.assertNotIn('恢复后的摘要',json.dumps(audit,ensure_ascii=False))

    def test_missing_asr_artifact_saves_retry_state_without_constructing_summarizer(self):
        _load_runner_class();plan,_=fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{
            'RUNNER_TEMP':tmp,'GITHUB_RUN_ID':'99','COURSE_SLOT':'0','DB_ENCRYPTION_KEY':'k'*32,
            'QWEN_PRODUCTION_TASK':'true','GITHUB_ACTIONS':'false'}):
            root=pipeline.root();(root/'inbox').mkdir()
            db=database(root/'fixture.db');payload=snapshot(db,root/'snapshot.db');db.conn.close()
            spec={'course_id':'10','course_title':'概率论','lecture':{'sub_id':'1'},'mode':'sharded',
                  'plan':plan,'media_seconds':600}
            pipeline.encode({'specification.json':pipeline.shards.encoded(spec),'database.db':payload},
                            'prepared',root/'inbox'/'prepared.enc')
            with patch.object(pipeline,'artifact',return_value=False),patch('src.ai.summarizer.Summarizer') as llm:
                with self.assertRaises(FileNotFoundError):pipeline.gather()
            llm.assert_not_called();saved=pipeline.decode(root/'out'/'state.enc','state')
            path=root/'verified.db';path.write_bytes(saved['database.db']);db=Database(str(path))
            self.assertIsNone(db.get_lecture('1')['transcript']);self.assertIsNone(db.get_lecture('1')['summary'])
            self.assertEqual(db.get_lecture('1')['error_stage'],'sharded_finalize')
            self.assertEqual(json.loads(db.read_meta('qwen_pipeline:1'))['recovery']['plan_hash'],fingerprint(plan))
            db.conn.close()

    def test_cross_run_recovery_rebinds_validated_completed_blocks(self):
        plan,results=fixture(slot=4,run='88');results[0]['complete']=False
        results[0]['chunks']=[]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{
            'RUNNER_TEMP':tmp,'GITHUB_RUN_ID':'99','COURSE_SLOT':'3','DB_ENCRYPTION_KEY':'k'*32,
            'QWEN_PRODUCTION_TASK':'true','GITHUB_ACTIONS':'false'}):
            root=pipeline.root();db=database(root/'fixture.db')
            db.write_meta('qwen_pipeline:1',json.dumps({'review':{'seconds':0},
              'recovery':{'run_id':'88','task_slot':4,'plan_hash':fingerprint(plan)}}))
            spec={'course_id':'10','course_title':'概率论','lecture':{'sub_id':'1'},'mode':'sharded','plan':plan}
            def artifacts(name,target,**kwargs):
                if '-state-' in name:return False
                target.mkdir(parents=True,exist_ok=True)
                with pipeline.shards.environment({'GITHUB_RUN_ID':'88','COURSE_SLOT':'4'}):
                    if 'prepare' in name:
                        pipeline.encode({'specification.json':pipeline.shards.encoded(spec)},'prepared',target/'prepared.enc')
                    else:
                        sid=int(name.rsplit('-',1)[1])
                        pipeline.encode({'result.json':pipeline.shards.encoded(results[sid])},f'result-{sid}',target/'worker-result.enc')
                return True
            with patch.object(pipeline,'artifact',side_effect=artifacts):
                files=pipeline.recover_preparation(db,'10','1')
            updated=json.loads(files['specification.json'])['plan']
            self.assertEqual(updated['run_id'],'99');self.assertEqual(updated['course_slot'],3)
            self.assertEqual(updated['blocks'],plan['blocks']);self.assertEqual(updated['audio_sha256'],plan['audio_sha256'])
            for shard in updated['shards']:
                saved=json.loads(files[f'completed-{shard["shard_id"]}.json'])
                self.assertEqual(saved['plan_hash'],fingerprint(updated))
                self.assertEqual(saved['chunks'],results[shard['shard_id']]['chunks'])
            db.conn.close()

    def test_queue_encryption_uses_fixed_queue_slot_but_task_results_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{
            'RUNNER_TEMP':tmp,'GITHUB_RUN_ID':'99','COURSE_SLOT':'0','DB_ENCRYPTION_KEY':'k'*32,
            'QWEN_PRODUCTION_TASK':'true','GITHUB_ACTIONS':'false'}):
            root=pipeline.root();pipeline.encode({'queue.json':b'[]'},'queue',root/'queue.enc')
            pipeline.encode({'private':b'classroom'},'prepared',root/'prepared.enc')
            os.environ['COURSE_SLOT']='23'
            self.assertEqual(pipeline.decode(root/'queue.enc','queue')['queue.json'],b'[]')
            with self.assertRaises(Exception):pipeline.decode(root/'prepared.enc','prepared')


class RecoveryBoundaryTests(unittest.TestCase):
    def test_later_lost_finalization_cannot_refund_an_older_artifact(self):
        with patch.object(pipeline,'last_finalization_attempt',return_value=2):
            with self.assertRaises(ValueError):pipeline.validate_checkpoint_age({'attempt.json':b'1'},'99',0)
            pipeline.validate_checkpoint_age({'attempt.json':b'2'},'99',0)

    def test_weak_speech_rescue_keeps_whole_lecture_quota(self):
        from src.ai.qwen_review_ledger import review_prepared
        material={'full_chunks':[],'vad_windows':[],'audio_path':'unused','recognition_terms':[],
                  'audio_seconds':60,'transcript':'','weak_windows':[{'start_ms':0,'end_ms':20000,'text':''}]}
        state={};saved=[]
        def recognize(path,key,windows,**kw):
            self.assertEqual(saved[-1]['seconds'],20)
            self.assertEqual(saved[-1]['attempts'][0]['status'],'reserved')
            return [(windows[0],[{'start_ms':0,'end_ms':20000,'text':'恢复实际讲话'}])],20,False
        with patch('src.ai.qwen_review_ledger.config.DOUBAO_ASR_API_KEY','fake'), \
             patch('src.ai.doubao_asr.rescue_intervals_pcm',side_effect=recognize):
            result=review_prepared(material,[],MagicMock(),state,lambda:saved.append(copy.deepcopy(state)))
        self.assertEqual(state['seconds'],20);self.assertEqual(len(result['weak_rescues']),1)
        self.assertEqual(result['variants'],[]);self.assertTrue(state['complete'])

    def test_cloud_caps_apply_across_all_candidates(self):
        from src.ai.qwen_review_ledger import review_prepared
        intervals=[dict(start_ms=i*60000,end_ms=(i+1)*60000,quote_start_ms=i*60000,
                        quote_end_ms=(i+1)*60000,text='疑点') for i in range(18)]
        state={'intervals':intervals};material={'full_chunks':[],'vad_windows':[],
                'audio_path':'unused','recognition_terms':[]}
        with patch('src.ai.qwen_review_ledger.config.DOUBAO_ASR_API_KEY','fake'), \
             patch('src.ai.doubao_asr.rescue_intervals_pcm',return_value=([],60,False)) as recognize:
            review_prepared(material,[],MagicMock(),state,lambda:None)
        self.assertEqual(recognize.call_count,10);self.assertEqual(state['seconds'],600)
        self.assertLessEqual(len(state['attempts']),12)

    def test_saved_review_variants_survive_missing_api_key(self):
        Runner=_load_runner_class();plan,results=fixture();material=assemble_material(plan,results)
        with tempfile.TemporaryDirectory() as tmp:
            db=database(Path(tmp)/'course.db');llm=MagicMock();llm.summarize.return_value=('摘要','test')
            runner=Runner(None,db,MagicMock(),MagicMock(),llm,MagicMock())
            review={'seconds':0,'complete':True,'material':{'variants':[{'original_quote':'疑点','cloud_text':'已有复核证据'}]}}
            with patch('src.runtime.config.DOUBAO_ASR_API_KEY',''):
                runner.run('10','概率论',{'sub_id':'1'},prepared_asr=material,review_state=review,checkpoint=lambda:None)
            self.assertIn('已有复核证据',llm.summarize.call_args.args[1]);db.conn.close()


class CompletedSummaryRecoveryTests(unittest.TestCase):
    def test_saved_summary_before_processed_marker_recovers_without_llm(self):
        Runner=_load_runner_class()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{
            'RUNNER_TEMP':tmp,'GITHUB_RUN_ID':'99','COURSE_SLOT':'0','DB_ENCRYPTION_KEY':'k'*32,
            'QWEN_PRODUCTION_TASK':'true','GITHUB_ACTIONS':'false','AUTO_COURSE_TERMS':'false'}):
            root=pipeline.root();(root/'inbox').mkdir()
            db=database(root/'fixture.db');db.update_summary('1','已保存的摘要','test')
            self.assertIsNone(db.get_lecture('1')['processed_at'])
            payload=snapshot(db,root/'snapshot.db');db.conn.close()
            spec={'course_id':'10','course_title':'概率论','lecture':{'sub_id':'1'},'mode':'finished'}
            pipeline.encode({'specification.json':pipeline.shards.encoded(spec),'database.db':payload},
                            'prepared',root/'inbox'/'prepared.enc')
            with patch.dict('sys.modules', {'src.pipeline.lecture_runner':SimpleNamespace(LectureRunner=Runner)}), \
                 patch.object(pipeline,'artifact',return_value=False),patch('src.ai.summarizer.Summarizer') as llm:
                pipeline.gather()
            llm.assert_not_called();saved=pipeline.decode(root/'out'/'state.enc','state')
            path=root/'verified.db';path.write_bytes(saved['database.db']);db=Database(str(path))
            self.assertIsNotNone(db.get_lecture('1')['processed_at']);self.assertEqual(db.get_lecture('1')['summary'],'已保存的摘要')
            self.assertIsNone(db.get_lecture('1')['emailed_at']);db.conn.close()

if __name__=='__main__':unittest.main()
