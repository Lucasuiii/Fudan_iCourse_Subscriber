"""Bounded local OCR of nearby platform screenshots or original video frames."""
from pathlib import Path
import subprocess
import tempfile

from src.ai.homework_review import MAX_FOCUS, nearby_pages


def collect_visual_evidence(client, course_id, sub_id, candidates, intervals, *, screenshot_fetcher=None, ocr=None):
    # Do not pass OCR through generic slide dedup/classification: a short page
    # number or handwritten assignment can otherwise be discarded as noise.
    if screenshot_fetcher is None:
        from src.api.icourse import fetch_ppt_image
        screenshot_fetcher = fetch_ppt_image
    if ocr is None:
        from src.ai.ocr import ocr_image_text
        ocr = ocr_image_text
    try:
        pages = client.get_ppt_list(course_id, sub_id)
    except Exception:
        pages = []
    results, seen = [], set()
    video_params = None
    video_checked = False
    for candidate in candidates[:MAX_FOCUS]:
        interval = next((row for row in intervals if row.get('chunk_id') == candidate['id']
                         and row.get('text') == candidate['quote']), None)
        shots = nearby_pages(pages, interval or candidate)
        for shot in shots:
            identity = (shot.get('id'), shot['created_sec'])
            if identity in seen:
                continue
            seen.add(identity)
            image = screenshot_fetcher(client, shot, max_attempts=1, timeout=15)
            text = ocr(image) if image else ''
            results.append({'source': 'platform_screenshot', 'seconds': shot['created_sec'],
                            'text': text[:2000], 'status': 'ok' if text.strip() else 'empty_or_unavailable'})
        # Sparse snapshots need not cover a spoken instruction. Seek actual
        # video only at aligned audio times; never guess a keyword's location.
        if interval and not any(r['status'] == 'ok' and interval['start_ms']/1000-15 <= r['seconds']
                                <= interval['end_ms']/1000+15 for r in results):
            if not video_checked:
                video_checked = True
                url = client.get_video_url(course_id, sub_id)  # unchanged fallback chain
                if url:
                    video_params = client.get_stream_params(url)
            if video_params:
                for seconds in sorted({interval['start_ms']/1000,
                                       (interval['quote_start_ms']+interval['quote_end_ms'])/2000,
                                       max(interval['start_ms']/1000, interval['end_ms']/1000-.5)}):
                    image = video_frame(video_params, seconds)
                    text = ocr(image) if image else ''
                    results.append({'source': 'video_frame', 'seconds': seconds, 'text': text[:2000],
                                    'status': 'ok' if text.strip() else 'empty_or_unavailable'})
    return {'status': 'ok' if any(r['status'] == 'ok' for r in results) else 'unavailable', 'frames': results}


def video_frame(params, seconds):
    url, headers = params
    with tempfile.TemporaryDirectory(prefix='icourse-homework-frame-') as tmp:
        path = Path(tmp)/'frame.png'
        try:
            # Credentials stay in process arguments; never print the command,
            # transport error body, URL or cookies into a public Actions log.
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-rw_timeout', '15000000',
                            '-headers', headers, '-ss', str(seconds), '-i', url,
                            '-frames:v', '1', '-vf', 'scale=1280:-2', '-y', str(path)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30)
            return path.read_bytes()
        except (OSError, subprocess.SubprocessError):
            return None
