import tempfile
import json
import unittest
from unittest.mock import MagicMock, patch

from src.ai import doubao_asr


class DoubaoASRTests(unittest.TestCase):
    def test_profiles_enforce_shared_time_and_clip_caps(self):
        for profile, seconds_cap, clips_cap in [('production', 900, 40), ('pilot15', 900, 18)]:
            for seconds in (1, 60):
                intervals = [{'start_ms': i*60000, 'end_ms': i*60000+seconds*1000, 'text': ''}
                             for i in range(45)]
                with patch.object(doubao_asr, '_encode_chunk', return_value=b'audio'), \
                     patch.object(doubao_asr, '_recognize_chunk', return_value=[{'text': '矩阵'}]):
                    results, spent, failed = doubao_asr.rescue_intervals_pcm(
                        'fake.pcm', 'test-key', intervals, session=MagicMock(),
                        max_seconds=10000, max_clips=100, budget_profile=profile)
                self.assertEqual(len(results), min(clips_cap, seconds_cap//seconds))
                self.assertLessEqual(spent, seconds_cap)
                self.assertFalse(failed)
        from src.ai.segment_rescue import cloud_budget_limits
        self.assertEqual(cloud_budget_limits(), (900, 40))
        with self.assertRaises(ValueError):
            cloud_budget_limits('unbounded')

    def test_optional_hotword_bounds(self):
        self.assertIsNone(doubao_asr.hotword_corpus(None))
        result=doubao_asr.hotword_corpus(['逆矩阵','逆矩阵',None,'','bad\nword','x'*31,'条件数'])
        self.assertEqual(json.loads(result['context']),{'hotwords':[{'word':'逆矩阵'},{'word':'条件数'}]})
        self.assertLessEqual(len(json.loads(doubao_asr.hotword_corpus([str(i) for i in range(100)])['context'])['hotwords']),20)

    def test_hotword_payload_is_opt_in(self):
        for words in (None,['希尔伯特矩阵']):
            submitted=MagicMock(headers={'X-Api-Status-Code':'20000000'})
            queried=MagicMock(headers={'X-Api-Status-Code':'20000000'})
            queried.json.return_value={'result':{'text':'矩阵'}}
            session=MagicMock()
            session.post.side_effect=[submitted,queried]
            doubao_asr._recognize_chunk(b'mp3','test',0,1000,session,hotwords=words)
            request=session.post.call_args_list[0].kwargs['json']['request']
            self.assertEqual('corpus' in request,bool(words))
            self.assertNotIn('correct_table_name',request)

    def test_wait_accepts_large_media_timeline_mismatch(self):
        with tempfile.NamedTemporaryFile() as audio:
            audio.write(b"\0" * doubao_asr.BYTES_PER_SECOND * 8)
            audio.flush()
            process = MagicMock(returncode=0)
            self.assertEqual(
                doubao_asr.wait_for_complete_audio(
                    audio.name, process, [b"Duration: 00:00:20.00"],
                ),
                (8, 20),
            )

    def test_wait_rejects_failed_download(self):
        with tempfile.NamedTemporaryFile() as audio:
            audio.write(b"\0" * doubao_asr.BYTES_PER_SECOND * 9)
            audio.flush()
            process = MagicMock(returncode=1)
            with self.assertRaises(doubao_asr.CloudAudioError):
                doubao_asr.wait_for_complete_audio(audio.name, process, [])

    def test_wait_accepts_moderate_timeline_shortfall(self):
        with tempfile.NamedTemporaryFile() as audio:
            audio.write(b"\0" * doubao_asr.BYTES_PER_SECOND * 9)
            audio.flush()
            process = MagicMock(returncode=0)
            self.assertEqual(
                doubao_asr.wait_for_complete_audio(
                    audio.name, process, [b"Duration: 00:00:11.00"],
                ),
                (9, 11),
            )

    def test_wait_accepts_exactly_half_the_media(self):
        with tempfile.NamedTemporaryFile() as audio:
            audio.write(b"\0" * doubao_asr.BYTES_PER_SECOND * 10)
            audio.flush()
            process = MagicMock(returncode=0)
            self.assertEqual(
                doubao_asr.wait_for_complete_audio(
                    audio.name, process, [b"Duration: 00:00:20.00"],
                ),
                (10, 20),
            )

    def test_submit_query_uses_seed_2_resource_and_relative_timestamps(self):
        submitted = MagicMock()
        submitted.headers = {"X-Api-Status-Code": "20000000"}
        queried = MagicMock()
        queried.headers = {"X-Api-Status-Code": "20000000"}
        queried.json.return_value = {
            "result": {"text": "良态和病态", "utterances": [
                {"start_time": 100, "end_time": 900,
                 "text": "良态和病态"},
            ]}
        }
        session = MagicMock()
        session.post.side_effect = [submitted, queried]
        result = doubao_asr._recognize_chunk(
            b"mp3", "test-key", 1_800_000, 1000, session,
        )
        self.assertEqual(result, [{
            "start_ms": 1_800_100, "end_ms": 1_800_900,
            "text": "良态和病态",
        }])
        args, kwargs = session.post.call_args_list[0]
        self.assertEqual(kwargs["headers"]["X-Api-Resource-Id"],
                         "volc.seedasr.auc")
        self.assertEqual(kwargs["headers"]["X-Api-Key"], "test-key")
        self.assertIn("data", kwargs["json"]["audio"])
        self.assertNotIn("url", kwargs["json"]["audio"])
        self.assertEqual(
            session.post.call_args_list[1].kwargs["headers"]
            ["X-Api-Request-Id"],
            kwargs["headers"]["X-Api-Request-Id"],
        )

    def test_empty_cloud_result_fails_over_instead_of_using_subtitles(self):
        with patch.object(doubao_asr, "_encode_chunk", return_value=b"mp3"), \
             patch.object(doubao_asr, "_recognize_chunk", return_value=[]):
            with self.assertRaises(doubao_asr.CloudASRError):
                doubao_asr.transcribe_pcm("unused", "key", 60,
                                          session=MagicMock())

    def test_chunking_covers_long_lecture_without_public_url(self):
        seen = []
        def recognize(_audio, _key, offset, duration, _session):
            seen.append((offset, duration))
            return [{"start_ms": offset, "end_ms": offset + duration,
                     "text": "课堂内容"}]
        with patch.object(doubao_asr, "_encode_chunk", return_value=b"mp3"), \
             patch.object(doubao_asr, "_recognize_chunk", side_effect=recognize):
            text, segments = doubao_asr.transcribe_pcm(
                "unused", "key", 3601, session=MagicMock(),
            )
        self.assertEqual(len(segments), 3)
        self.assertEqual(seen, [(0, 1_800_000), (1_800_000, 1_800_000),
                                (3_600_000, 1000)])
        self.assertEqual(text, "课堂内容 课堂内容 课堂内容")

    def test_rescue_uploads_only_selected_intervals_and_stops_on_error(self):
        intervals = [
            {"start_ms": 30_000, "end_ms": 40_000, "text": ""},
            {"start_ms": 90_000, "end_ms": 100_000, "text": "嗯"},
            {"start_ms": 150_000, "end_ms": 160_000, "text": ""},
        ]
        with patch.object(doubao_asr, "_encode_chunk",
                          return_value=b"mp3") as encode, \
             patch.object(doubao_asr, "_recognize_chunk", side_effect=[
                 [{"start_ms": 30_000, "end_ms": 40_000,
                   "text": "修复内容"}],
                 doubao_asr.CloudASRError("failed"),
             ]) as recognize:
            recovered, attempted, failed = doubao_asr.rescue_intervals_pcm(
                "unused", "key", intervals, session=MagicMock(),
            )
        self.assertEqual(len(recovered), 1)
        self.assertEqual(attempted, 20)
        self.assertTrue(failed)
        self.assertEqual(encode.call_count, 2)
        self.assertEqual(recognize.call_count, 2)
        self.assertEqual(encode.call_args_list[0].args, ("unused", 30, 10))

    def test_rescue_hard_duration_cap_is_fifteen_minutes(self):
        intervals = [{"start_ms": i * 60_000, "end_ms": (i + 1) * 60_000,
                      "text": ""} for i in range(18)]
        with patch.object(doubao_asr, "_encode_chunk", return_value=b"mp3"), \
             patch.object(doubao_asr, "_recognize_chunk", return_value=[]):
            rescues, attempted, failed = doubao_asr.rescue_intervals_pcm(
                "unused", "key", intervals, session=MagicMock(),
                max_seconds=9999, max_clips=99,
            )
        self.assertEqual(attempted, 900)
        self.assertEqual(len(rescues), 15)
        self.assertFalse(failed)

    def test_rescue_hard_clip_cap_is_forty_even_for_short_windows(self):
        intervals = [{'start_ms': i * 20_000, 'end_ms': (i + 1) * 20_000,
                      'text': ''} for i in range(45)]
        with patch.object(doubao_asr, '_encode_chunk', return_value=b'mp3'), \
             patch.object(doubao_asr, '_recognize_chunk', return_value=[]) as recognize:
            rescues, attempted, failed = doubao_asr.rescue_intervals_pcm(
                'unused', 'key', intervals, session=MagicMock(),
                max_seconds=9999, max_clips=99,
            )
        self.assertEqual(len(rescues), 40)
        self.assertEqual(recognize.call_count, 40)
        self.assertEqual(attempted, 800)
        self.assertFalse(failed)

    def test_rescue_respects_remaining_shared_budget(self):
        intervals = [
            {"start_ms": i * 30_000, "end_ms": i * 30_000 + 20_000,
             "text": ""} for i in range(20)
        ]
        with patch.object(doubao_asr, "_encode_chunk", return_value=b"mp3"), \
             patch.object(doubao_asr, "_recognize_chunk", return_value=[]):
            rescues, attempted, failed = doubao_asr.rescue_intervals_pcm(
                "unused", "key", intervals, session=MagicMock(),
                max_seconds=35, max_clips=2,
            )
        self.assertEqual(attempted, 20)
        self.assertEqual(len(rescues), 1)
        self.assertFalse(failed)


if __name__ == "__main__":
    unittest.main()
