"""Assignment reminders, conservative timestamps, private OCR and retry quotas."""
import copy
import unittest
from unittest.mock import MagicMock, patch

from src.ai.homework_review import (assignment_candidates, prioritize_candidates, focus_intervals,
                                    nearby_pages, ensure_homework_notice, homework_prompt)
from src.ai.qwen_review_ledger import review_prepared, validate_ledger


def material():
    return {'full_chunks': [{'start': 0, 'end': 120, 'text': '现在把主要精力放在做矩阵这一章的作业里，具体题号再看一下。'}],
            'vad_windows': [[0, 120]], 'audio_seconds': 180, 'audio_path': 'unused',
            'recognition_terms': ['矩阵'], 'transcript': 'x'*220, 'weak_windows': []}


def aligned(report, selected, *_args, **_kwargs):
    item = selected[0]
    return [dict(start_ms=102000, end_ms=118000, quote_start_ms=105000, quote_end_ms=115000,
                 chunk_id=item['id'], text=item['quote'])], [], [], {}


class AssignmentEvidenceTests(unittest.TestCase):
    def test_late_instruction_survives_many_early_mentions(self):
        chunks = [{'start': i*120, 'end': (i+1)*120, 'text': f'这是第{i}次讨论，上次作业只是举个例子，继续研究矩阵。'} for i in range(8)]
        chunks.append({'start': 960, 'end': 1080, 'text': '今天布置课后作业，完成第七页第二题，下周提交。'})
        selected = prioritize_candidates(assignment_candidates(chunks))
        self.assertEqual(selected[0]['id'], 8)
        self.assertLessEqual(len(selected), 4)
        for candidate in selected:
            self.assertIn(candidate['quote'], chunks[candidate['id']]['text'])

    def test_negation_short_and_repeated_cues_are_not_assignment_facts(self):
        candidates = assignment_candidates([{'start': 0, 'end': 10, 'text': '今天没有作业，下次课再布置。'},
                                            {'start': 10, 'end': 20, 'text': '作业'}])
        self.assertEqual(len(candidates), 2)
        self.assertFalse(candidates[1]['alignable'])
        prompt = homework_prompt({'candidates': candidates})
        self.assertIn('取消要求必须保留', prompt)
        summary = ensure_homework_notice('矩阵知识摘要', {'candidates': candidates})
        self.assertIn('作业与课务提醒', summary)
        self.assertIn('待核实', summary)
        self.assertIn('今天没有作业', summary)
        self.assertEqual(ensure_homework_notice(summary, {'candidates': candidates}), summary)
        self.assertEqual(ensure_homework_notice('摘要', {'candidates': []}), '摘要')

    def test_context_crosses_block_edge_without_changing_original_clock(self):
        raw, *_ = aligned({}, assignment_candidates(material()['full_chunks']))
        intervals = focus_intervals(raw, 180)
        self.assertEqual((intervals[0]['start_ms'], intervals[0]['end_ms']), (90000, 130000))
        self.assertEqual(intervals[0]['quote_start_ms'], 105000)
        self.assertLessEqual(intervals[0]['end_ms']-intervals[0]['start_ms'], 60000)
        self.assertEqual(focus_intervals(raw, 119)[0]['end_ms'], 119000)

    def test_stale_screenshots_and_guessed_keyword_times_are_excluded(self):
        candidate = {'block_start': 100, 'block_end': 120}
        pages = [{'created_sec': 0}, {'created_sec': 105}, {'created_sec': 120}, {'created_sec': 999}]
        self.assertEqual([p['created_sec'] for p in nearby_pages(pages, candidate)], [105, 120])


class AssignmentLedgerTests(unittest.TestCase):
    def test_assignment_priority_ocr_and_shared_quota_resume(self):
        data = material(); data['weak_windows'] = [{'start_ms': 95000, 'end_ms': 105000, 'text': ''}]
        state = {'intervals': [dict(start_ms=100000, end_ms=110000, quote_start_ms=100000,
                                    quote_end_ms=110000, text='普通疑点')]}  # overlaps focused context
        saved = []; ocr = MagicMock(return_value={'status': 'ok', 'frames': [{'seconds': 110, 'text': '第二题'}]})
        def recognize(path, key, windows, **kwargs):
            self.assertEqual(saved[-1]['attempts'][0]['status'], 'reserved')
            self.assertEqual(saved[-1]['seconds'], 40)
            self.assertEqual(windows[0]['kind'], 'homework')
            self.assertEqual(kwargs['hotwords'], ['矩阵'])
            return [(windows[0], [{'start_ms': 105000, 'end_ms': 115000, 'text': '矩阵章节第二题，下周提交。'}])], 40, False
        with patch('src.runtime.config.DOUBAO_ASR_API_KEY', 'fake'), \
             patch('src.ai.qwen_review_ledger.subprocess.run'), \
             patch('scripts.qwen_audio_alignment.align_suspects', side_effect=aligned), \
             patch('src.ai.doubao_asr.rescue_intervals_pcm', side_effect=recognize) as cloud:
            result = review_prepared(data, [], MagicMock(), state, lambda: saved.append(copy.deepcopy(state)), homework_ocr=ocr)
            resumed = review_prepared(data, [], MagicMock(), state, lambda: None, homework_ocr=ocr)
        self.assertEqual(cloud.call_count, 1); self.assertEqual(ocr.call_count, 1)
        self.assertEqual(state['seconds'], 40); self.assertEqual(result, resumed)
        self.assertIn('第二题', result['homework']['cloud'][0]['cloud_text'])
        validate_ledger(state)

    def test_interrupted_assignment_transport_is_not_repeated_or_refunded(self):
        state = {'intervals': []}; saved = []
        with patch('src.runtime.config.DOUBAO_ASR_API_KEY', 'fake'), \
             patch('src.ai.qwen_review_ledger.subprocess.run'), \
             patch('scripts.qwen_audio_alignment.align_suspects', side_effect=aligned), \
             patch('src.ai.doubao_asr.rescue_intervals_pcm', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                review_prepared(material(), [], MagicMock(), state, lambda: saved.append(copy.deepcopy(state)))
        self.assertEqual(saved[-1]['attempts'][0]['status'], 'reserved')
        with patch('src.runtime.config.DOUBAO_ASR_API_KEY', 'fake'), \
             patch('src.ai.doubao_asr.rescue_intervals_pcm') as cloud:
            result = review_prepared(material(), [], MagicMock(), state, lambda: None)
        cloud.assert_not_called(); self.assertEqual(state['seconds'], 40)
        self.assertEqual(result['homework']['cloud'][0]['status'], 'reserved')

    def test_no_key_or_failed_alignment_still_preserves_reminder(self):
        for key, failure in [('', None), ('fake', RuntimeError('do not expose private data'))]:
            state = {'intervals': []}
            with patch('src.runtime.config.DOUBAO_ASR_API_KEY', key), \
                 patch('src.ai.qwen_review_ledger.subprocess.run'), \
                 patch('scripts.qwen_audio_alignment.align_suspects', side_effect=failure), \
                 patch('src.ai.doubao_asr.rescue_intervals_pcm') as cloud:
                result = review_prepared(material(), [], MagicMock(), state, lambda: None)
            self.assertTrue(result['homework']['candidates']); cloud.assert_not_called()
            self.assertNotIn('private data', str(state))

    def test_twenty_short_clips_allowed_but_twenty_first_and_over_time_rejected(self):
        for duration, expected in [(1, 20), (60, 10)]:
            state = {'intervals': [dict(start_ms=i*60000, end_ms=i*60000+duration*1000,
                                        quote_start_ms=i*60000, quote_end_ms=i*60000+duration*1000,
                                        text='疑点') for i in range(25)]}
            data = dict(material(), full_chunks=[], transcript='')
            with patch('src.runtime.config.DOUBAO_ASR_API_KEY', 'fake'), \
                 patch('src.ai.doubao_asr.rescue_intervals_pcm', return_value=([], duration, False)) as cloud:
                review_prepared(data, [], MagicMock(), state, lambda: None)
            self.assertEqual(cloud.call_count, expected)
            self.assertLessEqual(state['seconds'], 600); validate_ledger(state)
            if duration == 1:
                too_many = copy.deepcopy(state)
                too_many['attempts'].append({'interval': {'start_ms': 9999999, 'end_ms': 10000999},
                                            'seconds': 1, 'status': 'reserved'})
                too_many['seconds'] += 1
                with self.assertRaises(ValueError): validate_ledger(too_many)
        extra = copy.deepcopy(state)
        extra['attempts'].append({'interval': {'start_ms': 9999999, 'end_ms': 10000999}, 'seconds': 1, 'status': 'reserved'})
        extra['seconds'] += 1
        with self.assertRaises(ValueError): validate_ledger(extra)

    def test_failed_ocr_preserves_audio_review_and_does_not_expose_transport_details(self):
        state = {'intervals': []}
        with patch('src.runtime.config.DOUBAO_ASR_API_KEY', 'fake'), \
             patch('src.ai.qwen_review_ledger.subprocess.run'), \
             patch('scripts.qwen_audio_alignment.align_suspects', side_effect=aligned), \
             patch('src.ai.doubao_asr.rescue_intervals_pcm', return_value=([], 40, False)) as cloud:
            result = review_prepared(material(), [], MagicMock(), state, lambda: None,
                                     homework_ocr=MagicMock(side_effect=RuntimeError('private-cookie-and-url')))
        cloud.assert_called_once()
        self.assertEqual(result['homework']['visual']['status'], 'failed')
        self.assertNotIn('private-cookie', str(state)); self.assertEqual(state['seconds'], 40)


class AssignmentVisualTests(unittest.TestCase):
    def test_short_snapshot_is_ocrd_without_generic_page_filter(self):
        from src.pipeline.homework_visual import collect_visual_evidence
        client = MagicMock(); client.get_ppt_list.return_value = [{'id': 1, 'created_sec': 110, 'pptimgurl': 'private'}]
        candidates = assignment_candidates(material()['full_chunks']); intervals = focus_intervals(aligned({}, candidates)[0], 180)
        result = collect_visual_evidence(client, '10', '1', candidates, intervals,
                                        screenshot_fetcher=MagicMock(return_value=b'image'), ocr=lambda _: '第2题')
        self.assertEqual(result['frames'][0]['text'], '第2题')
        self.assertNotIn('private', str(result)); client.get_video_url.assert_not_called()

    def test_missing_snapshot_uses_three_real_aligned_frames_and_existing_fallback(self):
        from src.pipeline.homework_visual import collect_visual_evidence
        client = MagicMock(); client.get_ppt_list.return_value = []
        client.get_video_url.return_value = 'signed-private'
        client.get_stream_params.return_value = ('vpn-private', 'cookies-private')
        candidates = assignment_candidates(material()['full_chunks']); intervals = focus_intervals(aligned({}, candidates)[0], 180)
        with patch('src.pipeline.homework_visual.video_frame', return_value=b'image') as frame:
            result = collect_visual_evidence(client, '10', '1', candidates, intervals,
                                            screenshot_fetcher=MagicMock(), ocr=lambda _: '作业第二题')
        self.assertEqual(frame.call_count, 3)
        self.assertEqual([row['seconds'] for row in result['frames']], [90, 110, 129.5])
        self.assertNotIn('private', str(result)); client.get_video_url.assert_called_once_with('10', '1')

    def test_unaligned_quote_never_seeks_guessed_video_position(self):
        from src.pipeline.homework_visual import collect_visual_evidence
        client = MagicMock(); client.get_ppt_list.return_value = []
        result = collect_visual_evidence(client, '10', '1', assignment_candidates(material()['full_chunks']), [],
                                        screenshot_fetcher=MagicMock(), ocr=MagicMock())
        self.assertEqual(result['status'], 'unavailable'); client.get_video_url.assert_not_called()


class AssignmentRunnerTests(unittest.TestCase):
    def test_saved_summary_contains_notice_without_mutating_raw_transcript(self):
        from test_lecture_quality_gate import _load_runner_class
        Runner = _load_runner_class(); db = MagicMock(); db.get_done_ppt_pages.return_value = []
        llm = MagicMock(); llm.summarize.return_value = ('仅包含数学知识的摘要', 'test')
        runner = Runner(None, db, MagicMock(), MagicMock(), llm, MagicMock())
        data = material(); runner._prepared_asr = data
        raw = copy.deepcopy(data)
        summary = runner._summarize('1', '高等代数', data['transcript'], [])
        self.assertIn('作业与课务提醒', summary)
        self.assertIn('题号', llm.summarize.call_args.args[1])
        self.assertEqual(data, raw); self.assertEqual(db.update_summary.call_args.args[1], summary)

    def test_gather_has_scoped_read_credentials_but_no_mail_credentials(self):
        import yaml
        from pathlib import Path
        workflow = yaml.safe_load((Path(__file__).resolve().parents[1]/'.github/workflows/qwen_production_lecture.yml').read_text())
        env = workflow['jobs']['gather']['env']
        self.assertEqual(env['StuId'], '${{ secrets.STUID }}')
        self.assertEqual(env['UISPsw'], '${{ secrets.UISPSW }}')
        self.assertNotIn('SMTP_PASSWORD', env)


class TailValidationTests(unittest.TestCase):
    def test_tail_preserves_original_clock_and_excludes_partial_boundary_block(self):
        from scripts.production_homework_validation import scoped_material
        original = {'audio_seconds': 1000, 'full_chunks': [
            {'chunk_id': 0, 'start': 350, 'end': 472, 'text': '边界前内容'},
            {'chunk_id': 1, 'start': 470, 'end': 592, 'text': '具体作业要求'}]}
        data, scope = scoped_material(original)
        self.assertEqual(scope['start'], 400)
        self.assertEqual(data['full_chunks'][0]['chunk_id'], 1)
        self.assertEqual(data['segments'][0]['start_ms'], 470000)
        self.assertEqual(data['transcript'], '具体作业要求')
        self.assertEqual(data['weak_windows'], [])
        self.assertEqual(len(original['full_chunks']), 2)

    def test_tail_review_keeps_every_original_quota_reservation(self):
        from scripts.production_homework_validation import isolated_ledger
        original = {'complete': True, 'seconds': 30, 'attempts': [
            {'interval': {'start_ms': 0, 'end_ms': 30000, 'kind': 'weak'},
             'seconds': 30, 'status': 'complete', 'segments': []}], 'intervals': [{'old': True}]}
        state = isolated_ledger(original)
        self.assertEqual(state['attempts'], original['attempts'])
        self.assertEqual(state['seconds'], 30)
        self.assertNotIn('complete', state)
        self.assertEqual(state['intervals'], [])
        self.assertTrue(original['complete']); validate_ledger(state)
        for unsafe in [dict(original, complete=False), dict(original, failed=True)]:
            with self.assertRaises(ValueError): isolated_ledger(unsafe)

    def test_tail_entry_is_read_only_and_cannot_run_on_push(self):
        import yaml
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        flow = yaml.safe_load((root/'.github/workflows/qwen_homework_validation.yml').read_text())
        self.assertEqual(flow['permissions'], {'contents': 'read', 'actions': 'read'})
        self.assertIn("github.event_name == 'workflow_dispatch'", flow['jobs']['review']['if'])
        env = flow['jobs']['review']['env']
        self.assertNotIn('SMTP_PASSWORD', env)
        self.assertEqual(env['AUTO_COURSE_TERMS'], 'false')
        self.assertEqual(flow['jobs']['register']['permissions'], {})
