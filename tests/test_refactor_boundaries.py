"""Import and checkpoint boundaries, without models, campus access or SMTP."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest
import sqlite3
from contextlib import closing
import tempfile
import yaml
from unittest.mock import patch
from types import SimpleNamespace

from src.data.checkpoint_database import CheckpointDatabase
from src.data.database import Database

ROOT = Path(__file__).resolve().parents[1]


class CoreImportTests(unittest.TestCase):
    def test_core_plans_and_recognition_import_without_script_entrypoints(self):
        # A core-only consumer must not import CLI modules or construct models.
        code = """
import sys
sys.modules['scripts'] = None
from src.pipeline.asr_queue import SharedQueue
from src.pipeline.prepared_lecture import assemble_material
from src.ai.qwen_transcriber import QwenTranscriber
from src.ai.qwen_missing_fallback import repair_missing
from src.ai.qwen_review_ledger import validate_ledger
from src.pipeline.runner_budget import desired_workers
assert desired_workers(7200, 60, 2) == 4
assert 'torch' not in sys.modules and 'qwen_asr' not in sys.modules
"""
        subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                       check=True, capture_output=True, text=True)

    def test_legacy_helper_imports_share_one_implementation(self):
        # Existing external callers and patch targets retain the same module.
        for old, new in (
            ('scripts.qwen_sharding', 'src.pipeline.qwen_plan'),
            ('scripts.qwen_segmentation', 'src.ai.qwen_segmentation'),
            ('scripts.qwen_quality', 'src.ai.qwen_quality'),
            ('scripts.qwen_audio_alignment', 'src.ai.qwen_audio_alignment'),
        ):
            with self.subTest(old=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))


class CheckpointTests(unittest.TestCase):
    def make_database(self, path, checkpoint=None):
        db = CheckpointDatabase(str(path), checkpoint=checkpoint)
        self.addCleanup(db.conn.close)
        db.upsert_course('10', '课程', '教师')
        db.insert_lecture('1', '10', '录播', '2026-10-09')
        return db

    def test_checkpoint_reads_committed_state_and_metadata_does_not_recurse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'course.db'
            snapshots = []
            def checkpoint():
                with closing(sqlite3.connect(path)) as reader:
                    snapshots.append(reader.execute('SELECT transcript FROM lectures').fetchone()[0])
                db.write_meta('audit', 'saved')  # Must not invoke itself.
            db = self.make_database(path, checkpoint)
            db.update_transcript('1', '正文')
            db.clear_transcript('1')
            self.assertEqual(snapshots, ['正文', None])
            self.assertEqual(db.read_meta('audit'), 'saved')
            db.conn.close()

    def test_checkpoint_failure_preserves_commit_without_changing_other_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = self.make_database(Path(tmp)/'checked.db',
                                    lambda: (_ for _ in ()).throw(RuntimeError('checkpoint failed')))
            plain = Database(str(Path(tmp)/'ordinary.db'))
            try:
                with self.assertRaisesRegex(RuntimeError, 'checkpoint failed'):
                    db.update_summary('1', '已提交摘要', 'test')
                self.assertEqual(db.get_lecture('1')['summary'], '已提交摘要')
                plain.upsert_course('20', '其他课程', '教师')
                plain.insert_lecture('2', '20', '其他录播', '2026-10-09')
                plain.update_summary('2', '普通保存', 'test')
                self.assertEqual(plain.get_lecture('2')['summary'], '普通保存')
            finally:
                plain.conn.close()
                db.conn.close()

    def test_email_delivery_uses_injected_database_and_checkpoints_each_receipt(self):
        from scripts import parallel_courses
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'receipts.db'; saves = []
            db = self.make_database(path, lambda: saves.append(True))
            for _ in range(3):
                db.update_error('1', 'asr', 'failed')
            saves.clear()
            original = Database.mark_emailed_batch
            def factory(_path):
                return db
            def sent(_emailer, database, _reporter, _items):
                database.mark_emailed_batch(['1'])
            def notified(_emailer, database, _reporter):
                database.mark_failure_notified_batch(['1'])
            main = SimpleNamespace(_send_email=sent, _send_failure_notices=notified,
                                   _in_run_scope=lambda *_: True)
            mail = SimpleNamespace(Emailer=lambda: object())
            with patch.dict(sys.modules, {'main': main, 'src.api.emailer': mail}), \
                 patch('src.runtime.config.SMTP_EMAIL', 'sender'), \
                 patch('src.runtime.config.SMTP_PASSWORD', 'password'), \
                 patch('src.runtime.config.RECEIVER_EMAILS', ['receiver']):
                parallel_courses.deliver(database_factory=factory)
            with closing(sqlite3.connect(path)) as reader:
                emailed, notified = reader.execute('SELECT emailed_at, failure_notified_at FROM lectures').fetchone()
            self.assertTrue(emailed and notified)
            self.assertEqual(len(saves), 2)
            self.assertIs(Database.mark_emailed_batch, original)


class OperationalStageTests(unittest.TestCase):
    def test_stage_accepts_explicit_services_and_legacy_adapter_preserves_injection(self):
        from scripts.production import planning
        from scripts import production_qwen as pipeline
        services = SimpleNamespace(validation_rank=lambda: 3)
        with patch.dict(os.environ, {'VALIDATION_LECTURE_RANKS': ''}):
            self.assertEqual(planning.validation_ranks(services), [3])
            with patch.object(pipeline, 'validation_rank', return_value=2):
                self.assertEqual(pipeline.validation_ranks(), [2])

    def test_previous_pool_identity_uses_its_journal_before_legacy_history(self):
        from scripts import production_qwen as pipeline
        with patch.dict(os.environ, {'GITHUB_ACTIONS': 'true', 'POOL_ARTIFACTS': 'false',
                                     'GITHUB_RUN_ID': '99'}), \
             patch.object(pipeline, '_POOL_RUNS', {'77'}), \
             patch('scripts.production_pool.finalization_attempt', return_value=2) as prior:
            self.assertEqual(pipeline.last_finalization_attempt('77', 0), 2)
        prior.assert_called_once_with('77', 0, False, allow_legacy=False)

    def test_original_cli_cleans_only_its_runtime_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            keep = Path(tmp)/'unrelated.txt'; keep.write_text('preserve')
            inbox = Path(tmp)/'qwen-shards'/'inbox'; inbox.mkdir(parents=True)
            (inbox/'synthetic.enc').write_bytes(b'synthetic')
            subprocess.run([sys.executable, '-m', 'scripts.production_qwen', 'clean'],
                           cwd=ROOT, env=dict(os.environ, RUNNER_TEMP=tmp),
                           check=True, capture_output=True, text=True)
            self.assertFalse(inbox.parent.exists())
            self.assertEqual(keep.read_text(), 'preserve')


class WorkflowBoundaryTests(unittest.TestCase):
    def load(self, path):
        return yaml.load((ROOT/path).read_text(), Loader=yaml.BaseLoader)

    def test_stage_authorization_precedes_inference_and_private_credentials(self):
        steps = self.load('.github/workflows/qwen_production_stage.yml')['jobs']['execute']['steps']
        authorize = next(i for i, step in enumerate(steps) if step.get('id') == 'authorize')
        for i, step in enumerate(steps):
            if step.get('uses', '').startswith('./.github/actions/qwen-') or step.get('env'):
                self.assertGreater(i, authorize)
        execution = [step for step in steps if step.get('run') == 'python -m scripts.production_pool_stage execute']
        self.assertEqual(len(execution), 1)
        self.assertEqual(execution[0]['if'], "${{ steps.authorize.outcome == 'success' }}")
        # This evaluated name is also a durable quota-recovery identity.
        self.assertIn("stage == 'gather'", execution[0]['name'])
        self.assertIn('Finalize through LectureRunner with saved quota', execution[0]['name'])
        for value in execution[0]['env'].values():
            self.assertIn("stage != 'asr'", value)

    def test_finished_worker_skips_inference_without_changing_the_existing_cache(self):
        for path, job in [('.github/workflows/qwen_production_stage.yml', 'execute'),
                          ('.github/workflows/qwen_production_lecture.yml', 'asr')]:
            step = next(s for s in self.load(path)['jobs'][job]['steps']
                        if s.get('uses') == './.github/actions/qwen-asr-runtime')
            self.assertEqual(step['with']['inference'], "${{ env.SHARD_ID != '-1' }}")
        worker = self.load('.github/actions/qwen-asr-runtime/action.yml')['runs']['steps']
        install = next(s for s in worker if s.get('uses') == './.github/actions/qwen-cpu-model')
        self.assertIn("inputs.inference == 'true'", install['if'])
        self.assertEqual(install['with']['extras'], 'true')
        cache = next(s for s in worker if s.get('uses') == 'actions/cache@v4')
        self.assertEqual(cache['with']['key'], 'qwen-production-asr-7278e1e-cpu-v1')
        self.assertNotIn('if', cache)  # Cache behavior stays the same even for reused input.


if __name__ == '__main__':
    unittest.main()
