"""Per-lecture state machine: prefetch → ASR → OCR drain → summarize → release.

One ``LectureRunner`` instance drives one lecture from "started" to either
"summary saved" or "deliberately skipped".  The class is single-use — make a
new instance per lecture so error state can't leak across runs.

Phases (named like the original ``main.process_lecture`` for diff-friendly
log greps):

  A  short-circuit ``summary already exists`` → mark processed, return.
  B  ``PPTPipeline.submit`` with ``defer_ocr=True``: stages 1-3 (fetch,
     register, dedup) run inline; the OCR jobs are held on the returned
     ``PPTAsyncHandle`` and only enter the pool at drain time, so ASR in
     Phase D gets the CPU to itself.
  C  schedule **next** lecture's prefetch (images always; audio only when
     the next lecture will actually be ASR-transcribed) so its download
     overlaps with the current lecture's ASR.
  D  transcript: cached → official iCourse transcript (config-gated,
     completeness-checked) → ASR.  For ASR, ``Scheduler.audio_downloader
     .get`` blocks for the ffmpeg spawn scheduled earlier, then
     ``Transcriber.transcribe_tail`` reads PCM from the disk file with
     tail-f semantics while ffmpeg keeps writing.
  E  ``handle.drain()`` submits the deferred OCR jobs and blocks for them.
  E1 reject consistently empty audio + visual material before the LLM.
  E2 ``PPTPipeline.prefetch_and_ocr`` spawns a background thread that
     collects + dedups + OCRs the next lecture's pages, overlapping with
     this lecture's LLM wait; leftovers are absorbed by the next
     ``submit``.
  F  ``bucketer.assemble`` builds the prompt; ``Summarizer.summarize``
     calls the LLM round-robin until one succeeds.
  G  persist ``update_summary``, ``mark_processed``, ``clear_error``.
  H  ``audio_downloader.release`` kills ffmpeg + deletes scratch file.

The runner never owns the WebVPN session check; the orchestrator
(LectureRunner's caller) calls ``_check_session`` between lectures so all
threads pick up refreshed cookies through the shared ``ICourseClient``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from src.ai import bucketer
from src.pipeline.ppt_pipeline import PPTPipeline
from src.ai.transcriber import IncompleteAudioError, NoAudioStreamError
from src.runtime import config
from src.runtime.content_quality import assess_content_quality
from src.runtime.transcript_policy import (
    assess_official_transcript,
    merge_timed_segments,
)

if TYPE_CHECKING:
    from src.data.database import Database
    from src.api.icourse import ICourseClient
    from src.runtime.reporter import Reporter
    from src.runtime.scheduler import Scheduler
    from src.ai.summarizer import Summarizer
    from src.ai.transcriber import Transcriber


class LectureRunner:
    """Run one lecture end-to-end.  Construct once per orchestration session,
    call ``run`` for each lecture in order."""

    def __init__(self, client: "ICourseClient", db: "Database",
                 scheduler: "Scheduler", transcriber: "Transcriber",
                 summarizer: "Summarizer", reporter: "Reporter"):
        self._client = client
        self._db = db
        self._scheduler = scheduler
        self._transcriber = transcriber
        self._summarizer = summarizer
        self._reporter = reporter
        self._ppt = PPTPipeline(db, scheduler, reporter)
        # Official-transcript segments fetched during the prefetch decision
        # (``_needs_audio``), keyed by sub_id, so ``_get_transcript`` doesn't
        # re-fetch them one lecture later.
        self._official_cache: dict[str, list[dict]] = {}
        self._transcript_source = "unknown"
        self._asr_actual_duration = 0.0
        self._asr_expected_duration = 0.0

    # ── Public entry point ──────────────────────────────────────────────

    def run(self, course_id: str, course_title: str, lecture: dict,
            next_info: Optional[tuple[str, str]] = None) -> Optional[str]:
        """Process one lecture.  Returns the summary text or None.

        ``next_info``: ``(course_id, sub_id)`` of the next lecture, used to
        kick off its prefetch concurrently.  Pass ``None`` for the last
        lecture in the batch.
        """
        sub_id = str(lecture["sub_id"])
        sub_title = lecture.get("sub_title", sub_id)
        date = lecture.get("date", "")
        t_start = time.time()
        self._transcript_source = "unknown"
        self._asr_actual_duration = 0.0
        self._asr_expected_duration = 0.0
        self._reporter.lecture_start(course_title, sub_title, date)

        existing = self._db.get_lecture(sub_id)
        # ── Phase A — short-circuit if a summary already exists ─────────
        if self._has_summary(existing):
            self._reporter.lecture_skip_v2_done(
                sub_title, len(existing["summary"])
            )
            self._schedule_next(next_info)
            self._db.mark_processed(sub_id)
            self._db.clear_error(sub_id)
            # The return value feeds the email batch — suppress it when
            # this summary already went out so it isn't re-sent.
            if existing.get("emailed_at"):
                return None
            return existing["summary"]

        # ── Phase B — submit PPT pipeline (fetch + dedup, no OCR yet) ──
        # OCR is deferred (defer_ocr=True) so ASR in Phase D gets exclusive
        # CPU.  OCR will be submitted in Phase E (handle.drain()).
        ppt_handle = self._ppt.submit(
            self._client, course_id, sub_id, defer_ocr=True,
        )

        # ── Phase C — schedule next lecture's prefetch ─────────────────
        # Done BEFORE ASR so the next audio download can start filling its
        # AudioDownloader slot while we transcribe.  Both audio + image
        # prefetches are idempotent so this is safe to call any time.
        self._schedule_next(next_info)

        # ── Phase D — ASR transcription ────────────────────────────────
        transcript, transcript_segments = self._get_transcript(
            existing, course_id, sub_id,
        )
        if transcript is None:
            # _get_transcript already logged + persisted the skip reason.
            # Still drain the PPT handle: with defer_ocr the OCR jobs are
            # only submitted at drain() time, so skipping it would leave
            # the pages 'pending' forever and force the retry run to redo
            # download + dedup from scratch.
            ppt_handle.drain()
            return None

        # ── Phase E — drain remaining OCR work ─────────────────────────
        ppt_stats = ppt_handle.drain()

        # ── Phase E1 — reject empty/uncertain material before the LLM ───
        # A complete long recording with almost no speech and no useful PPT
        # is terminal "No Content" (for example, the teacher never arrived).
        # Ambiguous technical cases are retried instead of being summarized.
        ppt_pages = self._db.get_done_ppt_pages(sub_id)
        ppt_status_counts = self._db.get_ppt_status_counts(sub_id)
        ppt_uncertain = (
            ppt_status_counts.get("failed", 0)
            + ppt_status_counts.get("pending", 0)
        )
        quality = assess_content_quality(
            transcript=transcript,
            segments=transcript_segments,
            ppt_pages=ppt_pages,
            transcript_source=self._transcript_source,
            actual_audio_seconds=self._asr_actual_duration,
            expected_audio_seconds=self._asr_expected_duration,
            ppt_failed_count=max(ppt_stats.failed, ppt_uncertain),
        )
        if quality.action != "summarize":
            self._reporter.info(
                "    [Quality] Insufficient lecture material "
                f"(audio={self._asr_actual_duration / 60:.1f} min, "
                f"transcript={quality.transcript_chars} chars/"
                f"{quality.segment_count} segments, "
                f"PPT={quality.ppt_page_count} usable pages)."
            )
            self._release_audio(sub_id)
            if quality.action == "skip_no_content":
                self._reporter.info(
                    "    [SKIP] No effective lecture content detected; "
                    "summary and email suppressed."
                )
                self._db.update_transcript(sub_id, transcript)
                self._db.mark_processed(sub_id)
                self._db.clear_error(sub_id)
            else:
                self._reporter.info(
                    "    [RETRY] Material quality is uncertain; will retry."
                )
                self._db.clear_transcript(sub_id)
                self._db.update_error(
                    sub_id,
                    "content_quality",
                    "insufficient reliable lecture material",
                )
            return None

        # Persist only after the material has passed the gate. This preserves
        # ASR work across a later LLM failure without allowing a crash between
        # transcription and quality assessment to bypass the gate next run.
        self._db.update_transcript(sub_id, transcript)

        # ── Phase E2 — kick off next lecture's OCR (runs during LLM) ───
        # Images were prefetched in Phase C; prefetch_and_ocr spawns a
        # background thread that collects them, dedups and submits OCR,
        # so the whole thing genuinely overlaps with this lecture's LLM
        # wait.  Whatever isn't finished when the LLM returns is absorbed
        # by the next lecture's own PPTPipeline.submit().
        if next_info:
            next_course, next_sub = next_info
            self._ppt.prefetch_and_ocr(self._client, next_course, next_sub)

        # ── Phase F — bucketed-prompt LLM summary ──────────────────────
        if not transcript.strip():
            self._reporter.info("    Empty transcript, skipping summary.")
            self._release_audio(sub_id)
            self._db.mark_processed(sub_id)
            self._db.clear_error(sub_id)
            return None

        summary = self._summarize(
            sub_id, course_title, transcript, transcript_segments,
        )
        if summary is None:
            self._release_audio(sub_id)
            return None

        # ── Phase G — persist + clear errors ───────────────────────────
        self._db.mark_processed(sub_id)
        self._db.clear_error(sub_id)

        # ── Phase H — release audio resources (ffmpeg + file) ──────────
        self._release_audio(sub_id)

        elapsed = time.time() - t_start
        self._reporter.lecture_done(course_title, sub_title, elapsed)
        return summary

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _has_summary(existing: dict | None) -> bool:
        return bool(
            existing
            and existing.get("summary")
        )

    def prefetch_first(self, course_id: str, sub_id: str) -> None:
        """Prefetch for the first lecture in the batch — same decision
        logic (skip the audio download when transcription won't need it)
        as the in-loop Phase C prefetch."""
        self._schedule_next((course_id, sub_id))

    def _schedule_next(self, next_info: Optional[tuple[str, str]]):
        if next_info is None:
            return
        next_course, next_sub = next_info
        # Image prefetch is always useful; the audio download — a full
        # ffmpeg pull of the lecture — only when the next lecture will
        # actually be ASR-transcribed.  Both are idempotent so repeated
        # invocations are no-ops; the audio side blocks on its semaphore
        # until a download slot frees.
        self._scheduler.prefetch_lecture(
            self._client, next_course, next_sub,
            audio=self._needs_audio(next_course, next_sub),
        )

    def _needs_audio(self, course_id: str, sub_id: str) -> bool:
        """False when transcription won't need the audio stream: a cached
        transcript exists, or the official transcript looks usable.  Keeps
        prefetching from spending a download slot (and a full lecture of
        bandwidth) on audio that would just be killed in Phase H."""
        existing = self._db.get_lecture(sub_id)
        if existing and existing.get("transcript"):
            return False
        if config.USE_OFFICIAL_TRANSCRIPT:
            try:
                segments = self._client.get_transcript_segments(sub_id)
                if segments:
                    self._official_cache[sub_id] = segments
                # No tail hint at prefetch time — the lecture's PPT rows
                # aren't registered yet.  _get_transcript re-checks with
                # the hint and schedules the download then if needed.
                if self._official_transcript_usable(segments):
                    return False
            except Exception as e:
                self._reporter.info(
                    f"    [Official transcript] probe failed: {type(e).__name__}"
                )
        return True

    @staticmethod
    def _official_transcript_usable(segments: list[dict] | None,
                                    max_gap_minutes: int = 20,
                                    duration_hint_s: int = 0) -> bool:
        """True if the official transcript is complete enough to use.

        Looks for a >``max_gap_minutes`` hole in three places: before the
        first segment (head truncation), between segments, and — when a
        duration hint is known (the last PPT screenshot offset, a lower
        bound on lecture length) — after the last segment (tail
        truncation)."""
        # Keep the legacy signature for callers/tests. The production threshold
        # remains 20 minutes; non-default values use the original gap-only path.
        if max_gap_minutes != 20:
            if not segments:
                return False
            max_gap_ms = max_gap_minutes * 60_000
            ordered = sorted(segments, key=lambda s: s["start_ms"])
            if ordered[0]["start_ms"] > max_gap_ms:
                return False
            prev_end = ordered[0]["end_ms"]
            for seg in ordered[1:]:
                if seg["start_ms"] - prev_end > max_gap_ms:
                    return False
                prev_end = max(prev_end, seg["end_ms"])
            return not (
                duration_hint_s
                and duration_hint_s * 1000 - prev_end > max_gap_ms
            )
        mode, _ = assess_official_transcript(segments, duration_hint_s)
        return mode == "complete"

    def _probe_media_duration(self, course_id: str, sub_id: str) -> float | None:
        """Probe the signed media URL locally; return None on any failure."""
        try:
            video_url = self._client.get_video_url(course_id, sub_id)
            if not video_url:
                return None
            stream_url, headers = self._client.get_stream_params(video_url)
            return self._transcriber.probe_duration(stream_url, headers)
        except Exception as exc:
            self._reporter.info(
                f"    [Official transcript] duration probe failed: "
                f"{type(exc).__name__}"
            )
            return None

    def _get_transcript(self, existing: dict | None, course_id: str,
                        sub_id: str) -> tuple[Optional[str], Optional[list]]:
        """Return (transcript, segments) or (None, None) on skip.

        Tries the official iCourse transcript first — when available and
        complete-enough (no >20 min silence gaps) it replaces the ASR
        step entirely, saving ~5 min of CPU time per lecture.
        """
        if existing and existing.get("transcript"):
            self._transcript_source = "cached"
            self._reporter.info(
                f"    Transcript exists "
                f"({len(existing['transcript'])} chars), "
                f"skipping transcription."
            )
            return existing["transcript"], None

        # Try official transcript before firing up ASR (config-gated).
        official: list[dict] | None = None
        hybrid_intervals: list[tuple[float, float]] = []
        official_duration: float | None = None
        if config.USE_OFFICIAL_TRANSCRIPT:
            try:
                official = self._official_cache.pop(sub_id, None)
                if official is None:
                    official = self._client.get_transcript_segments(sub_id)
                # Prefer the actual media duration. The final PPT timestamp is
                # only a lower-bound fallback when ffprobe cannot inspect the
                # signed stream.
                media_duration = self._probe_media_duration(course_id, sub_id)
                official_duration = media_duration
                duration_hint = (
                    media_duration or self._db.get_max_ppt_created_sec(sub_id)
                )
                mode, gaps = assess_official_transcript(official, duration_hint)
                if mode == "complete":
                    text = " ".join(s["text"] for s in official)
                    self._reporter.info(
                        f"    Using official transcript "
                        f"({len(text)} chars, {len(official)} segments)"
                    )
                    self._db.update_transcript(sub_id, text)
                    self._transcript_source = "official"
                    # The audio may have been prefetched before we knew the
                    # official transcript was usable — stop that download
                    # now instead of letting it run until Phase H.
                    self._release_audio(sub_id)
                    return text, official
                if mode == "hybrid" and media_duration:
                    hybrid_intervals = gaps
                    self._reporter.info(
                        f"    Official transcript has {len(gaps)} long gap(s); "
                        "running local ASR only for missing intervals."
                    )
                else:
                    self._reporter.info(
                        "    Official transcript is incomplete; falling back "
                        "to full local ASR."
                    )
            except Exception as e:
                self._reporter.info(
                    f"    [Official transcript] unavailable, falling back "
                    f"to ASR: {type(e).__name__}"
                )

        # Pull the audio handle.  ``schedule`` is idempotent — usually the
        # previous lecture already kicked it off (Phase C), but for the
        # first lecture in the batch we still need to fire it ourselves.
        downloader = self._scheduler.audio_downloader
        downloader.schedule(self._client, course_id, sub_id)
        try:
            handle = downloader.get(sub_id, timeout=120)
        except TimeoutError as e:
            self._reporter.info("    [SKIP] Audio download timed out.")
            self._db.update_error(sub_id, "transcribe", str(e))
            return None, None
        if handle is None:
            # AudioDownloader returns None when get_video_url() returned
            # None — i.e. the lecture has no playable video.  Record an
            # error so the lecture is retried (the video may appear later)
            # and then paused with a private notice after max_errors.
            # The "no_video" stage is a contract with the frontend, which
            # renders it as a gray "无视频" hint instead of a red failure.
            self._reporter.lecture_skip_no_video(
                existing.get("sub_title", sub_id) if existing else sub_id
            )
            self._db.update_error(sub_id, "no_video", "no playable video URL")
            return None, None

        try:
            used_full_local_asr = not (hybrid_intervals and official)
            if hybrid_intervals and official:
                gap_text, gap_segments = self._transcriber.transcribe_tail_intervals(
                    handle.path, handle.process, handle.stderr_chunks,
                    hybrid_intervals,
                    expected_duration_s=official_duration,
                )
                if gap_text.strip():
                    segments = merge_timed_segments(official, gap_segments)
                    transcript = " ".join(s["text"] for s in segments)
                else:
                    self._reporter.info(
                        "    Gap-only ASR returned no text; retrying full local ASR."
                    )
                    transcript, segments = self._transcriber.transcribe_tail(
                        handle.path, handle.process, handle.stderr_chunks,
                    )
                    used_full_local_asr = True
            else:
                transcript, segments = self._transcriber.transcribe_tail(
                    handle.path, handle.process, handle.stderr_chunks,
                )
        except NoAudioStreamError as e:
            self._reporter.info("    [SKIP] Video-only (no audio stream).")
            # Do not mark this as processed: retry it like other media
            # failures, then surface it through the attention notice.
            self._db.update_error(sub_id, "no_audio", str(e))
            self._release_audio(sub_id)
            return None, None
        except IncompleteAudioError as e:
            # Truncated download (ffmpeg may even exit 0 on a server-side
            # cut).  Don't persist the partial transcript — it would
            # short-circuit the retry — just record the error so the
            # lecture is retried up to max_errors times.
            self._reporter.info(
                "    [SKIP] Incomplete audio, will retry next run."
            )
            self._db.update_error(sub_id, "transcribe", str(e))
            self._release_audio(sub_id)
            return None, None
        except Exception as e:
            self._reporter.info(
                f"    [FAIL] Transcription error: {type(e).__name__}"
            )
            self._db.update_error(sub_id, "transcribe", str(e))
            self._release_audio(sub_id)
            raise

        if used_full_local_asr:
            self._transcript_source = "local_asr"
            self._asr_actual_duration = self._transcriber.last_audio_duration
            self._asr_expected_duration = (
                self._transcriber.last_media_duration or 0.0
            )
        else:
            self._transcript_source = "hybrid"
        return transcript, segments

    def _summarize(self, sub_id: str, course_title: str, transcript: str,
                   transcript_segments: list[dict] | None) -> Optional[str]:
        try:
            kept_pages = self._db.get_done_ppt_pages(sub_id)
            prompt_text, mode = bucketer.assemble(
                transcript, transcript_segments, kept_pages,
            )
            self._reporter.info(
                f"    [Time] Generating summary at "
                f"{time.strftime('%H:%M:%S')}"
                f" — mode={mode}, prompt={len(prompt_text)} chars"
            )
            summary, model_used = self._summarizer.summarize(
                course_title, prompt_text,
            )
            self._reporter.info(
                f"    [OK] Summary by {model_used}: {len(summary)} chars"
            )
            self._db.update_summary(sub_id, summary, model_used)
            return summary
        except Exception as e:
            self._reporter.info(
                f"    [FAIL] Summarization error: {type(e).__name__}"
            )
            self._db.update_error(sub_id, "summarize", str(e))
            raise

    def _release_audio(self, sub_id: str):
        try:
            self._scheduler.audio_downloader.release(sub_id)
        except Exception as e:
            self._reporter.info(
                f"    [WARN] audio release failed: {type(e).__name__}"
            )
