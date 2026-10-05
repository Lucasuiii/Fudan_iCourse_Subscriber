"""Direct board images, late evidence retention and non-repeating transport."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.ai.homework_vision import read_images, validated_frames
from src.ai.homework_visual_evidence import assess_visual
from src.pipeline.homework_visual import collect_visual_evidence, frame_times, visual_window
from test_homework_review import board_png


def response(count=2):
    return {'frames': [{'frame_index': i, 'text': 'P76 1,2,4', 'writing_state': 'stable',
                       'references': [{'raw': 'P76 1,2,4', 'page': 76,
                                       'exercises': [1, 2, 4], 'legible': True}]} for i in range(count)]}


def inputs(count=2):
    return [{'image': board_png(), 'seconds': 100+i*20,
             'source': 'video_frame', 'candidate_id': 'cue'} for i in range(count)]


def client_response(client, data):
    client.with_options.return_value.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=json.dumps(data)))], usage=None)


class DirectVisionTests(unittest.TestCase):
    def test_actual_images_sent_no_ocr_and_later_twelve_frames(self):
        client = MagicMock(); client.get_ppt_list.return_value = []
        client.get_video_url.return_value = 'private'; client.get_stream_params.return_value = ('private', '')
        candidate = {'id': 0, 'quote': '今天作业', 'block_start': 0, 'block_end': 120}
        interval = {'chunk_id': 0, 'text': candidate['quote'], 'quote_start_ms': 10000, 'quote_end_ms': 20000}
        ocr, reader = MagicMock(), MagicMock(side_effect=lambda frames: validated_frames(response(len(frames)), len(frames)))
        with patch('src.pipeline.homework_visual.video_frame', return_value=board_png()) as capture:
            result = collect_visual_evidence(client, '10', '1', [candidate], [interval],
                                            audio_seconds=300, vision_reader=reader, ocr=ocr)
        ocr.assert_not_called(); reader.assert_called_once()
        self.assertEqual(capture.call_count, 12)
        self.assertEqual(result['frames'][-1]['seconds'], 199.9)
        self.assertEqual(result['windows'][0]['range']['end_ms'], 200000)
        self.assertTrue(all(f['reader'] == 'deepseek_vision' for f in result['frames']))
        self.assertEqual(result['reference_status'], 'supported')
        self.assertNotIn('private', str(result))

    def test_video_end_is_clamped_and_no_duplicate_instant(self):
        interval = {'quote_start_ms': 10000, 'quote_end_ms': 20000}
        window = visual_window({'block_end': 120}, interval, 60)
        times = frame_times(interval, window)
        self.assertEqual(len(times), len(set(times)))
        self.assertTrue(all(0 <= t < 60 for t in times))
        self.assertLessEqual(len(times), 12)
        self.assertEqual(visual_window({'block_end': 120}, interval)['end_ms'], 120000)

    def test_batch_transport_precedes_reading_and_failure_uses_local_ocr(self):
        client = MagicMock(); client.get_ppt_list.return_value = []
        client.get_video_url.return_value = 'private'; client.get_stream_params.return_value = ('private', '')
        candidate = {'id': 0, 'quote': '作业', 'block_start': 0, 'block_end': 120}
        interval = {'chunk_id': 0, 'text': '作业', 'quote_start_ms': 10000, 'quote_end_ms': 20000}
        captured = []
        def frame(*args, **kwargs): captured.append(1); return board_png()
        def failed(images):
            self.assertEqual(len(captured), 12)
            raise RuntimeError('signed-url-and-private-cookie')
        with patch('src.pipeline.homework_visual.video_frame', side_effect=frame):
            result = collect_visual_evidence(client, '10', '1', [candidate], [interval], audio_seconds=300,
                                            vision_reader=failed, ocr=lambda _: [])
        self.assertEqual(result['capture_status'], 'complete')
        self.assertEqual(result['reference_status'], 'unverified')
        self.assertTrue(all(f['vision_status'] == 'failed' for f in result['frames']))
        self.assertNotIn('private', str(result))

    def test_json_response_must_cover_exact_frame_identities(self):
        for invalid in [{'frames': []}, {'frames': [response()['frames'][0]]*2},
                        {'frames': [dict(response()['frames'][0], frame_index=9), response()['frames'][1]]}]:
            with self.assertRaises(ValueError): validated_frames(invalid, 2)

    def test_unreadable_invented_and_concatenated_numbers_not_supported(self):
        for change in [{'legible': False}, {'raw': 'P76 1,2,9', 'exercises': [1, 2, 9]},
                       {'exercises': [1, 2, 4, 9]}, {'page': 67}, {'exercises': [True, 2, 4]},
                       {'raw': 'P76 1,2,4', 'exercises': [1, 2, 4, 4]}]:
            data = response(); data['frames'][0]['references'][0].update(change)
            self.assertEqual(validated_frames(data, 2)[0]['references'], [])
        data = response(); data['frames'][0].update(text='P76 678', references=[
            {'raw': 'P76 678', 'page': 76, 'exercises': [6, 7, 8], 'legible': True}])
        self.assertEqual(validated_frames(data, 2)[0]['references'], [])

    def test_multi_frame_vision_is_corroboration_not_model_confidence(self):
        rows = validated_frames(response(), 2)
        for row, seconds in zip(rows, [100, 120]): row.update(seconds=seconds, candidate_id='cue')
        result = assess_visual({'candidate_ids': ['cue'], 'frames': rows})
        self.assertEqual(result['reference_status'], 'supported')
        self.assertNotIn('confidence', rows[0]['references'][0])
        single = assess_visual({'candidate_ids': ['cue'], 'frames': rows[:1]})
        self.assertEqual(single['reference_status'], 'unverified')
        conflicting = assess_visual({'candidate_ids': ['cue'], 'frames': rows},
                                    [{'candidate_id': 'cue', 'status': 'complete', 'cloud_text': '77页第8题'}])
        self.assertEqual(conflicting['reference_status'], 'unverified')

    def test_reserved_before_transport_reused_after_completion(self):
        client = MagicMock(); client_response(client, response())
        ledger, saved = [], []
        def save(): saved.append(copy.deepcopy(ledger))
        results = read_images(client, 'deepseek-flash', inputs(), ledger, save)
        self.assertEqual(saved[0][0]['status'], 'reserved')
        self.assertEqual(saved[-1][0]['status'], 'complete')
        self.assertEqual(read_images(client, 'deepseek-flash', inputs(), ledger, save), results)
        transport = client.with_options.return_value.chat.completions.create
        transport.assert_called_once(); client.with_options.assert_called_once_with(max_retries=0)
        kwargs = transport.call_args.kwargs
        self.assertEqual(kwargs['model'], 'deepseek-flash')
        parts = kwargs['messages'][0]['content']
        images = [p for p in parts if p['type'] == 'image_url']
        self.assertEqual(len(images), 6)
        self.assertTrue(all(p['image_url']['url'].startswith('data:image/jpeg;base64,') for p in images))
        self.assertTrue(all(p['image_url']['detail'] == 'original' for p in images))

    def test_interruption_or_failed_response_never_repeats_request(self):
        for failure in [KeyboardInterrupt(), RuntimeError('private-error')]:
            client = MagicMock(); ledger = []
            client.with_options.return_value.chat.completions.create.side_effect = failure
            with self.assertRaises(type(failure)): read_images(client, 'deepseek-flash', inputs(), ledger, lambda: None)
            self.assertEqual(ledger[0]['status'], 'reserved' if isinstance(failure, KeyboardInterrupt) else 'failed')
            with self.assertRaises(ValueError): read_images(client, 'deepseek-flash', inputs(), ledger, lambda: None)
            client.with_options.return_value.chat.completions.create.assert_called_once()
            self.assertNotIn('private-error', str(ledger))

    def test_changed_input_and_four_cue_limit_do_not_make_requests(self):
        client = MagicMock(); client_response(client, response())
        ledger = []; read_images(client, 'deepseek-flash', inputs(), ledger, lambda: None)
        changed = inputs(); changed[-1]['seconds'] += 1
        with self.assertRaises(ValueError): read_images(client, 'deepseek-flash', changed, ledger, lambda: None)
        client.with_options.return_value.chat.completions.create.assert_called_once()
        ledger = [{'candidate_id': f'old{i}', 'status': 'reserved'} for i in range(4)]
        with self.assertRaises(ValueError): read_images(client, 'deepseek-flash', inputs(), ledger, lambda: None)

    def test_checkpoint_failure_stops_before_transport(self):
        client = MagicMock()
        with self.assertRaises(OSError):
            read_images(client, 'deepseek-flash', inputs(), [], MagicMock(side_effect=OSError('save failed')))
        client.with_options.assert_not_called()

    def test_only_explicit_deepseek_route_can_receive_images(self):
        from src.ai.summarizer import Summarizer
        summarizer = Summarizer.__new__(Summarizer)
        summarizer.providers = [{'name': 'modelscope', 'models': ['DeepSeek-V4-Pro']}]
        summarizer._clients = {'modelscope': MagicMock(), 'deepseek': MagicMock()}
        self.assertIsNone(summarizer.homework_image_reader([], lambda: None))
        summarizer.providers.append({'name': 'deepseek', 'models': ['deepseek-v4-pro']})
        with patch('src.ai.homework_vision.read_images') as read:
            reader = summarizer.homework_image_reader([], lambda: None); reader(inputs())
        self.assertEqual(read.call_args.args[1], 'deepseek-flash')
        summarizer._clients['modelscope'].chat.completions.create.assert_not_called()

    def test_preview_budget_preserves_late_board_native_dimensions(self):
        from src.pipeline.homework_previews import retain_preview
        previews = []
        for t in range(12):
            retain_preview(previews, board_png((1920, 1080)), {'source': 'video_frame', 'seconds': t})
        self.assertEqual([p['seconds'] for p in previews], list(range(6, 12)))
        self.assertTrue(all(p['original_size'] == [1920, 1080] for p in previews))
        self.assertLessEqual(sum(len(p['jpeg_base64']) for p in previews), 1400000)

    def test_vision_ledger_validation_and_stable_notice(self):
        from src.ai.qwen_review_ledger import validate_ledger
        from src.ai.homework_review import homework_prompt
        self.assertIn('不证明老师写完', homework_prompt({'candidates': [{}]}))
        self.assertNotIn('private_recovery_marker', homework_prompt({
            'candidates': [{}], 'vision_calls': [{'candidate_id': 'private_recovery_marker'}]}))
        for calls in [[{'candidate_id': 'cue', 'status': 'complete'}]*2,
                      [{'candidate_id': 'cue', 'status': 'unknown'}], 'wrong']:
            with self.assertRaises(ValueError): validate_ledger({'homework': {'vision_calls': calls}})

    def test_ranges_and_same_numbers_on_different_pages_are_not_merged(self):
        data = response(); data['frames'][0].update(text='P76 1-4', references=[
            {'raw': 'P76 1-4', 'page': 76, 'exercises': [1, 4], 'legible': True}])
        self.assertEqual(validated_frames(data, 2)[0]['references'], [])
        rows = [{'candidate_id': 'cue', 'seconds': seconds, 'references': [
            {'text': '第1、2题', 'source': 'deepseek_vision', 'legible': True, 'page': page}]}
                for seconds, page in [(100, 76), (120, 94)]]
        result = assess_visual({'candidate_ids': ['cue'], 'frames': rows})
        self.assertEqual(result['reference_status'], 'unverified')


if __name__ == '__main__':
    unittest.main()
