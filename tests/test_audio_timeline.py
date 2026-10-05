"""Timestamp gaps must survive decoding to timestamp-free PCM."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from src.runtime.scheduler import AudioDownloader, _PendingSpawn


class AudioTimelineTests(unittest.TestCase):
    def extraction_command(self, preserve):
        with tempfile.TemporaryDirectory() as tmp:
            downloader=AudioDownloader(tmp,max_concurrent=1)
            pending=_PendingSpawn();downloader._active['1']=pending
            client=MagicMock();client.get_video_url.return_value='selected-private-url'
            client.get_stream_params.return_value=('authenticated-private-url','private-header')
            process=MagicMock();process.stderr=[];process.poll.return_value=0
            with patch('src.runtime.scheduler.subprocess.Popen',return_value=process) as spawn:
                downloader._spawn_when_ready(client,'10','1',pending,preserve)
            cmd=spawn.call_args.args[0]
            self.assertEqual(downloader.get('1').timeline_preserved,preserve)
            client.get_video_url.assert_called_once_with('10','1')
            client.get_stream_params.assert_called_once_with('selected-private-url')
            downloader.shutdown()
            return cmd

    def test_legacy_extraction_and_authenticated_playback_selection_stay_intact(self):
        legacy=self.extraction_command(False);pilot=self.extraction_command(True)
        self.assertNotIn('-af',legacy)
        self.assertEqual(pilot[pilot.index('-af')+1],'aresample=async=1:first_pts=0')
        self.assertEqual(pilot[pilot.index('-i')+1],legacy[legacy.index('-i')+1])
        self.assertEqual(pilot[pilot.index('-headers')+1],legacy[legacy.index('-headers')+1])

    @unittest.skipUnless(shutil.which('ffmpeg'),'FFmpeg required')
    def test_real_timestamp_gap_is_silence_and_does_not_shift_later_speech(self):
        import numpy as np
        filter_text=self.extraction_command(True)
        filter_text=filter_text[filter_text.index('-af')+1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'gaps.mkv'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                'sine=frequency=440:sample_rate=16000:duration=20','-af',
                r'asetpts=PTS+if(gte(T\,10)\,40/TB\,0)','-c:a','pcm_f32le',str(source)],check=True)
            durations={}
            for name,extra in [('plain',[]),('timeline',['-af',filter_text])]:
                target=tmp/(name+'.raw')
                subprocess.run(['ffmpeg','-v','error','-i',str(source),*extra,
                    '-ar','16000','-ac','1','-f','f32le',str(target)],check=True)
                samples=np.fromfile(target,dtype=np.float32);durations[name]=len(samples)/16000
                if name=='timeline':
                    self.assertEqual(float(np.max(np.abs(samples[12*16000:48*16000]))),0)
                    self.assertGreater(float(np.max(np.abs(samples[52*16000:58*16000]))),.01)
            self.assertEqual(durations,{'plain':20,'timeline':60})
            # A contiguous audio-only source ends at its real endpoint. The
            # filter does not pad arbitrarily to a declared video duration.
            target=tmp/'contiguous.raw'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i',
                'sine=sample_rate=16000:duration=3','-af',filter_text,
                '-ar','16000','-ac','1','-f','f32le',str(target)],check=True)
            self.assertEqual(target.stat().st_size/64000,3)
