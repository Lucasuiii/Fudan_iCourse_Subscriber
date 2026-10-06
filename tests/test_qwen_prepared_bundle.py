"""Actual authenticated volumes and failure retention, without course/model calls."""
import io
import json
import os
import glob
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import zipfile
import yaml
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from scripts import production_prepared_bundle as bundle
from scripts import production_qwen as pipeline
from scripts import sharded_qwen_pilot as shards


class PreparedBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {'QWEN_PRODUCTION_TASK':'true',
            'DB_ENCRYPTION_KEY':'k'*32, 'GITHUB_RUN_ID':'99',
            'COURSE_SLOT':'0', 'RUNNER_TEMP':self.temp.name})
        self.env.start(); self.addCleanup(self.env.stop)
        self.path = pipeline.out('prepared.enc')

    def test_payload_over_legacy_cap_roundtrips_without_changing_bytes_or_hashes(self):
        files = {'lecture.flac':os.urandom(2500), 'chunk-0.flac':os.urandom(2500),
                 'specification.json':b'{"frozen_terms_sha256":"original"}'}
        with patch.object(shards, 'MAX_BUNDLE', 4096), patch.object(bundle, 'PART_BYTES', 1024):
            shards.seal(files, 'prepared', self.path)
            self.assertEqual(shards.unseal(self.path, 'prepared'), files)
            self.assertGreater(len(list(self.path.parent.glob('prepared-*-part-*.enc'))), 1)
            self.assertEqual(self.path.read_bytes()[:4], b'QSP2')

    def test_legacy_single_prepared_bundle_is_still_readable(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('specification.json', b'{}')
            archive.writestr('lecture.flac', b'original audio')
        nonce = os.urandom(12)
        self.path.write_bytes(b'QSP1'+nonce+AESGCM(shards.key()).encrypt(
            nonce, stream.getvalue(), shards.context('prepared')))
        self.assertEqual(shards.unseal(self.path, 'prepared')['lecture.flac'], b'original audio')

    def test_missing_corrupted_swapped_or_cross_run_volumes_are_rejected(self):
        with patch.object(bundle, 'PART_BYTES', 256):
            shards.seal({'lecture.flac':os.urandom(1300)}, 'prepared', self.path)
            parts = sorted(self.path.parent.glob('prepared-*-part-*.enc'))
            original = parts[0].read_bytes()
            parts[0].unlink()
            with self.assertRaises(FileNotFoundError): shards.unseal(self.path, 'prepared')
            parts[0].write_bytes(original[:-1]+bytes([original[-1]^1]))
            with self.assertRaises(Exception): shards.unseal(self.path, 'prepared')
            parts[0].write_bytes(parts[1].read_bytes())
            with self.assertRaises(Exception): shards.unseal(self.path, 'prepared')
            parts[0].write_bytes(original)
            other = self.path.with_name('other.enc')
            shards.seal({'lecture.flac':os.urandom(1300)}, 'prepared', other)
            other_part = sorted(self.path.parent.glob('other-*-part-*.enc'))[0]
            other_part.write_bytes(original)
            with self.assertRaises(Exception): shards.unseal(other, 'prepared')
            for env in ({'GITHUB_RUN_ID':'100'}, {'COURSE_SLOT':'1'}):
                with patch.dict(os.environ, env), self.assertRaises(Exception):
                    shards.unseal(self.path, 'prepared')
            with self.assertRaises(ValueError): shards.unseal(self.path, 'state')

    def test_failed_new_manifest_write_preserves_previous_checkpoint_and_removes_new_parts(self):
        shards.seal({'lecture.flac':b'old'}, 'prepared', self.path)
        before = set(self.path.parent.iterdir())
        actual = bundle.atomic_write
        def fail_manifest(path, data):
            if path == self.path: raise OSError('disk failure')
            actual(path, data)
        with patch.object(bundle, 'PART_BYTES', 256), patch.object(bundle, 'atomic_write', side_effect=fail_manifest):
            with self.assertRaises(OSError): shards.seal({'lecture.flac':b'new'*300}, 'prepared', self.path)
        self.assertEqual(set(self.path.parent.iterdir()), before)
        self.assertEqual(shards.unseal(self.path, 'prepared')['lecture.flac'], b'old')

    def test_writer_and_reader_reject_oversized_or_unsafe_manifests_before_body_reads(self):
        with patch.object(bundle, 'MAX_BYTES', 1024):
            with self.assertRaisesRegex(ValueError, 'total size limit'):
                shards.seal({'lecture.flac':b'x'*1025}, 'prepared', self.path)
        with patch.object(bundle, 'MAX_FILES', 1):
            with self.assertRaisesRegex(ValueError, 'too many files'):
                shards.seal({'a':b'x','b':b'x'}, 'prepared', self.path)
        with self.assertRaises(ValueError): shards.seal({'../unsafe':b'x'}, 'prepared', self.path)
        # Even authenticated malformed metadata cannot bypass the total-size gate.
        malformed = {'schema':1, 'bundle':'a'*32, 'stem':'prepared',
            'archive_bytes':bundle.MAX_BYTES+1, 'archive_sha256':'a'*64,
            'part_bytes':bundle.PART_BYTES, 'parts':33}
        nonce = os.urandom(12)
        self.path.write_bytes(b'QSP2'+nonce+AESGCM(shards.key()).encrypt(nonce,
            json.dumps(malformed).encode(), shards.context('prepared')+b':multipart-v1:manifest'))
        with self.assertRaisesRegex(ValueError, 'Invalid prepared manifest'):
            shards.unseal(self.path, 'prepared')

    def test_failure_saves_audio_only_when_duplicate_chunks_exceed_limit_and_redacts_secrets(self):
        files = {'lecture.flac':b'x'*2300, 'chunk-0.flac':b'y'*2300}
        spec = {'course_id':'10', 'course_title':'private title', 'lecture':{'sub_id':'1'},
            'mode':'sharded', 'prepare_phase':'checkpoint_write',
            'plan':{'blocks':[{}], 'shards':[{}]},
            'audio_diagnostics':{'audio_seconds':600, 'private_url':'https://secret'}}
        error = ValueError('Prepared bundle exceeds total size limit')
        with patch.object(bundle, 'MAX_BYTES', 4096), patch.object(bundle, 'PART_BYTES', 1024), \
                patch.object(pipeline, 'lecture_snapshot', return_value=b'snapshot'):
            pipeline.preserve_preparation_failure(MagicMock(), '1', spec, files, None, error)
            saved = shards.unseal(self.path, 'prepared')
        saved_spec = json.loads(saved['specification.json'])
        self.assertEqual(saved['lecture.flac'], files['lecture.flac'])
        self.assertNotIn('chunk-0.flac', saved)
        self.assertEqual(saved_spec['mode'], 'failed')
        self.assertTrue(saved_spec['recovery_blocked'])
        audit = pipeline.out('prepare-failure.json').read_text()
        self.assertNotIn('private', audit); self.assertNotIn('https://', audit)
        audit = json.loads(audit)
        self.assertEqual(audit['error_code'], 'bundle_size_limit')
        self.assertTrue(audit['checkpoint_saved']); self.assertTrue(audit['audio_retained'])
        self.assertTrue(audit['fallback_audio_only'])
        self.assertGreater(audit['full_checkpoint_content_bytes'], 4096)
        self.assertLess(audit['content_bytes'], 4096)

    def test_secondary_checkpoint_failure_is_recorded_without_replacing_original_error(self):
        error = TimeoutError('private-url and password must never appear')
        spec = {'course_id':'10', 'lecture':{'sub_id':'1'}, 'prepare_phase':'vad'}
        with patch.object(pipeline, 'lecture_snapshot', return_value=b'snapshot'), \
                patch.object(pipeline, 'encode', side_effect=OSError('private-secret')):
            pipeline.preserve_preparation_failure(MagicMock(), '1', spec, {}, None, error)
        audit = json.loads(pipeline.out('prepare-failure.json').read_text())
        self.assertEqual(audit['error_type'], 'TimeoutError')
        self.assertEqual(audit['secondary_error_types'], ['OSError','OSError'])
        self.assertFalse(audit['checkpoint_saved'])
        self.assertNotIn('private', json.dumps(audit))

    def test_workflow_artifact_upload_transfers_manifest_and_every_volume_for_recovery(self):
        files = {'lecture.flac':os.urandom(2400), 'chunk-0.flac':os.urandom(2400),
                 'specification.json':b'{"original_timestamp":119,"frozen_terms":"unchanged"}'}
        workflow = Path(__file__).resolve().parents[1]/'.github/workflows/qwen_production_lecture.yml'
        steps = yaml.load(workflow.read_text(), Loader=yaml.BaseLoader)['jobs']['prepare']['steps']
        upload = next(s for s in steps if s.get('with', {}).get('name') ==
                      'qwen-production-prepare-${{ inputs.task_slot }}')
        with patch.object(bundle, 'PART_BYTES', 1024):
            shards.seal(files, 'prepared', self.path)
            copied = Path(self.temp.name)/'restored'; copied.mkdir()
            for expression in upload['with']['path'].splitlines():
                for source in glob.glob(expression.replace('${{ runner.temp }}', self.temp.name)):
                    shutil.copyfile(source, copied/Path(source).name)
            self.assertEqual(shards.unseal(copied/'prepared.enc', 'prepared'), files)

    def test_prepare_propagates_original_error_when_both_retention_attempts_fail(self):
        db = MagicMock(); db.get_lecture.return_value = {'summary':'existing summary'}
        original = ValueError('Prepared bundle exceeds total size limit')
        with patch.object(pipeline, 'artifact', return_value=False), \
                patch.object(pipeline, 'task_files', return_value=(db,'10','private title',{'sub_id':'1'})), \
                patch.object(pipeline, 'lecture_snapshot', return_value=b'snapshot'), \
                patch.object(pipeline, 'encode', side_effect=[original, OSError('private'), OSError('private')]), \
                patch.dict('sys.modules', {'main':SimpleNamespace(login_with_retry=MagicMock()),
                    'src.pipeline.ppt_pipeline':SimpleNamespace(PPTPipeline=MagicMock()),
                    'src.api.icourse':SimpleNamespace(ICourseClient=MagicMock())}):
            with self.assertRaises(ValueError) as caught:
                pipeline.prepare()
        self.assertIs(caught.exception, original)
        audit = json.loads(pipeline.out('prepare-failure.json').read_text())
        self.assertEqual(audit['error_code'], 'bundle_size_limit')
        self.assertEqual(audit['phase'], 'checkpoint_write')
        self.assertEqual(audit['secondary_error_types'], ['OSError','OSError'])
        db.conn.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
