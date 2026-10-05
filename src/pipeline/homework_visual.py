"""Bounded, native-resolution board OCR with delayed frame coverage."""
import io
import math
from pathlib import Path
import subprocess
import tempfile

from src.ai.homework_review import MAX_FOCUS, nearby_pages
from src.ai.homework_visual_evidence import assess_visual, candidate_key, references

MAX_VIDEO_FRAMES = 24
FRAMES_PER_CUE = 6


def visual_window(candidate, interval, audio_seconds=None):
    start = max(0, interval['quote_start_ms']/1000-10)
    end = interval['quote_end_ms']/1000+90
    limit = audio_seconds if isinstance(audio_seconds, (int, float)) and math.isfinite(audio_seconds) else candidate['block_end']
    end = min(end, limit)
    if end <= start:
        return None
    return {'start_ms': round(start*1000), 'end_ms': round(end*1000)}


def frame_times(interval, window):
    start, end = window['start_ms']/1000, window['end_ms']/1000
    anchor_end = interval['quote_end_ms']/1000
    times = {min(max(start, t), max(start, end-.1))
             for t in [start, anchor_end, anchor_end+20, anchor_end+40, anchor_end+60, anchor_end+90]}
    return sorted(times)


def image_views(image):
    from PIL import Image
    with Image.open(io.BytesIO(image)) as source:
        source.load(); original = source.convert('RGB')
    w, h = original.size
    # Overlapping halves cover either board side without claiming a fixed
    # region is the blackboard. Keep native full-frame pixels available.
    views = [('full', original)]
    if w >= 640 and h >= 360:
        for name, box in [('board_left', (0, 0, math.ceil(w*2/3), h)),
                          ('board_right', (math.floor(w/3), 0, w, h))]:
            crop = original.crop(box)
            factor = max(1, min(2, 2560/max(crop.size)))
            if factor > 1:
                crop = crop.resize(tuple(round(s*factor) for s in crop.size), Image.Resampling.LANCZOS)
            views.append((name, crop))
    result = []
    for name, view in views:
        buf = io.BytesIO(); view.save(buf, format='PNG')
        result.append((name, buf.getvalue(), view.size))
    return result


def read_frame(image, ocr):
    if not image:
        return {'status': 'capture_failed', 'text': '', 'references': [], 'views': []}
    try:
        views = image_views(image)
    except Exception:
        return {'status': 'image_decode_failed', 'text': '', 'references': [], 'views': []}
    texts, refs, audits = [], {}, []
    for name, data, size in views:
        try:
            blocks = ocr(data)
            # Legacy string OCR stays visible but cannot establish confidence.
            if isinstance(blocks, str):
                blocks = [{'text': blocks, 'confidence': None}] if blocks else []
            view_text = []
            for block in blocks:
                text = block.get('text', '') if isinstance(block, dict) else block.text
                score = block.get('confidence') if isinstance(block, dict) else block.confidence
                if not text.strip():
                    continue
                view_text.append(text)
                if isinstance(score, (int, float)) and math.isfinite(score):
                    for ref in references(text):
                        refs[ref] = max(refs.get(ref, 0), score)
            texts.extend(view_text)
            audits.append({'region': name, 'size': list(size), 'status': 'ok' if view_text else 'no_text'})
        except Exception as error:
            audits.append({'region': name, 'size': list(size), 'status': 'ocr_failed', 'error_type': type(error).__name__})
    text = '\n'.join(dict.fromkeys(texts))[:4000]
    return {'status': 'ok' if text else 'ocr_failed' if any(v['status'] == 'ocr_failed' for v in audits) else 'no_text',
            'text': text, 'references': [{'text': t, 'confidence': s} for t, s in refs.items()], 'views': audits}


def collect_visual_evidence(client, course_id, sub_id, candidates, intervals, *, audio_seconds=None,
                            screenshot_fetcher=None, ocr=None, frame_observer=None):
    if screenshot_fetcher is None:
        from src.api.icourse import fetch_ppt_image
        screenshot_fetcher = fetch_ppt_image
    if ocr is None:
        from src.ai.ocr import ocr_image_strict
        ocr = ocr_image_strict
    try:
        pages = client.get_ppt_list(course_id, sub_id)
    except Exception:
        pages = []
    results, seen, windows, ids = [], set(), [], []
    video_params = None; video_checked = False; video_count = 0
    for candidate in candidates[:MAX_FOCUS]:
        cid = candidate_key(candidate['id'], candidate['quote']); ids.append(cid)
        interval = next((row for row in intervals if row.get('chunk_id') == candidate['id']
                         and row.get('text') == candidate['quote']), None)
        window = visual_window(candidate, interval, audio_seconds) if interval else None
        windows.append({'candidate_id': cid, 'range': window, 'aligned': bool(interval)})
        shots = nearby_pages(pages, window or candidate)
        if isinstance(audio_seconds, (int, float)) and math.isfinite(audio_seconds):
            shots = [shot for shot in shots if 0 <= shot['created_sec'] <= audio_seconds]
        for shot in shots:
            identity = (cid, shot.get('id'), shot['created_sec'])
            if identity in seen:
                continue
            seen.add(identity)
            image = None
            try:
                image = screenshot_fetcher(client, shot, max_attempts=1, timeout=15)
                row = read_frame(image, ocr)
            except Exception as error:
                row = {'status': 'capture_failed', 'error_type': type(error).__name__, 'text': '', 'references': [], 'views': []}
            results.append(dict(row, source='platform_screenshot', seconds=shot['created_sec'], candidate_id=cid))
            if callable(frame_observer):
                frame_observer(image, results[-1])
        # Ordinary formula OCR is never a reason to skip delayed board frames.
        # Unaligned keywords cannot create a guessed video seek position.
        if window:
            if not video_checked:
                video_checked = True
                try:
                    url = client.get_video_url(course_id, sub_id)  # unchanged fallback chain
                    if url:
                        video_params = client.get_stream_params(url)
                except Exception:
                    video_params = None
            times = frame_times(interval, window)[:FRAMES_PER_CUE]
            for seconds in times:
                if video_count >= MAX_VIDEO_FRAMES:
                    break
                video_count += 1
                image = video_frame(video_params, seconds) if video_params else None
                row = read_frame(image, ocr)
                results.append(dict(row, source='video_frame', seconds=seconds, candidate_id=cid))
                if callable(frame_observer):
                    frame_observer(image, results[-1])
    valid = [r for r in results if r['status'] in ('ok', 'no_text')]
    complete = results and len(valid) == len(results) and all(
        all(v['status'] in ('ok', 'no_text') for v in r['views']) for r in valid)
    capture = 'complete' if complete else 'partial' if valid else 'unavailable'
    return assess_visual({'schema_version': 2, 'capture_status': capture, 'candidate_ids': ids,
                          'windows': windows, 'frames': results})


def video_frame(params, seconds):
    url, headers = params
    with tempfile.TemporaryDirectory(prefix='icourse-homework-frame-') as tmp:
        path = Path(tmp)/'frame.png'
        try:
            # Keep original resolution. Credentials/errors stay out of logs.
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-rw_timeout', '15000000',
                            '-headers', headers, '-ss', str(seconds), '-i', url,
                            '-frames:v', '1', '-y', str(path)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
            return path.read_bytes()
        except (OSError, subprocess.SubprocessError):
            return None
