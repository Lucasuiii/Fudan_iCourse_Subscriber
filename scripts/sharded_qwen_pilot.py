"""Manual encrypted prepare -> original-block workers -> isolated review/summary.

No SMTP, database publisher, production course selection, or media fallback changes.
"""
from __future__ import annotations
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import wave
import zipfile

from scripts.qwen_sharding import (MAX_COURSES, MAX_RUNNERS, parse_baselines, build_plan, build_audio_plan,
    validate_plan, validate_result, assemble, fingerprint)
from scripts.qwen_segmentation import reaches_acquisition_limit

MAX_BUNDLE = 512 * 1024 * 1024


def root():
    path = Path(os.environ['RUNNER_TEMP']) / 'qwen-shards'
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def key():
    if os.environ.get('QWEN_PRODUCTION_TASK') == 'true':
        from scripts.parallel_courses import bundle_key
        return bundle_key(os.environ['DB_ENCRYPTION_KEY'])
    value = base64.b64decode(os.environ['QWEN_ASR_TEST_KEY'], validate=True)
    if len(value) != 32:
        raise ValueError('Invalid test encryption key')
    return value


def context(role, slot=None):
    slot = int(os.environ.get('COURSE_SLOT', '0')) if slot is None else slot
    production = os.environ.get('QWEN_PRODUCTION_TASK') == 'true'
    from scripts.qwen_sharding import MAX_TASKS
    if not 0 <= slot < (MAX_TASKS if production else MAX_COURSES) or not str(os.environ['GITHUB_RUN_ID']).isdigit():
        raise ValueError('Invalid encrypted artifact identity')
    prefix = 'icourse-qwen-production-v1' if production else 'icourse-qwen-shards-v1'
    return f"{prefix}:{os.environ['GITHUB_RUN_ID']}:{slot}:{role}".encode()


def seal(files, role, path, slot=None):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, value in files.items():
            if Path(name).name != name:
                raise ValueError('Bundle paths must be flat')
            archive.writestr(name, value)
    payload = data.getvalue()
    if len(payload) > MAX_BUNDLE:
        raise ValueError('Encrypted bundle too large')
    nonce = os.urandom(12)
    blob = b'QSP1' + nonce + AESGCM(key()).encrypt(nonce, payload, context(role, slot))
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_bytes(blob)
    temp.replace(path)


def unseal(path, role, slot=None):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if path.stat().st_size > MAX_BUNDLE + 32:
        raise ValueError('Oversized encrypted bundle')
    blob = path.read_bytes()
    if blob[:4] != b'QSP1':
        raise ValueError('Invalid shard artifact header')
    raw = AESGCM(key()).decrypt(blob[4:16], blob[16:], context(role, slot))
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (len(names) > 600 or len(set(names)) != len(names)
                or any(Path(n).name != n for n in names)
                or sum(info.file_size for info in infos) > MAX_BUNDLE):
            raise ValueError('Invalid shard archive')
        return {name: archive.read(name) for name in names}


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode()


def legacy_report(path):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if path.stat().st_size > 20*1024*1024:
        raise ValueError('Oversized baseline report')
    data = path.read_bytes()
    if data[:5] != b'QASR1':
        raise ValueError('Invalid baseline artifact')
    return json.loads(AESGCM(key()).decrypt(data[5:17], data[17:], b'qwen-asr-benchmark-v1'))


def command(args, *, timeout=120):
    subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                   check=True, timeout=timeout)


@contextmanager
def environment(values):
    prior = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in prior.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def outputs(**values):
    with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
        for name, value in values.items():
            stream.write(f'{name}={json.dumps(value, separators=(",", ":"))}\n')


@contextmanager
def mirrored_results(benchmark, destination):
    """Persist every quota checkpoint before external calls, including on SIGKILL."""
    original = benchmark.save_encrypted
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    def save(report):
        original(report)
        temp = destination.with_suffix('.tmp')
        temp.write_bytes((benchmark.workspace()/'result.enc').read_bytes())
        temp.replace(destination)
    benchmark.save_encrypted = save
    try:
        yield
    finally:
        benchmark.save_encrypted = original


def plan():
    refs = parse_baselines(os.environ['BASELINES'])
    outputs(courses={'include': [{'course_slot': i, **reference} for i, reference in enumerate(refs)]})
    print(f'Planned {len(refs)} private course slots; global runner cap={MAX_RUNNERS}', flush=True)


def prepare():
    from scripts import benchmark_qwen_asr as benchmark
    if os.environ.get('REUSE_SHARDED_RUN_ID'):
        prepare_reuse()
        return
    started = time.perf_counter()
    slot = int(os.environ['COURSE_SLOT'])
    reference = {'run_id': os.environ['BASELINE_RUN_ID'], 'artifact': os.environ['BASELINE_ARTIFACT']}
    parse_baselines(json.dumps([reference]))
    mode = os.environ.get('PREPARE_MODE', 'baseline')
    if mode not in ('baseline', 'private_latest'):
        raise ValueError('Unknown acquisition mode')
    if mode == 'private_latest':
        if reference['artifact'] not in ('qwen-asr-encrypted-result-0', 'qwen-asr-encrypted-result-1'):
            raise ValueError('Independent preparation requires an existing private course slot')
        fetch_env = {'TEST_COURSE_SLOT': reference['artifact'][-1], 'LATEST_LECTURE': 'true'}
    else:
        baseline_dir = root() / 'baseline'
        command(['gh', 'run', 'download', reference['run_id'], '--repo', os.environ['GITHUB_REPOSITORY'],
                 '--name', reference['artifact'], '--dir', str(baseline_dir)])
        baseline = legacy_report(baseline_dir / 'result.enc')
        # Validate baseline before authenticating or fetching any private media.
        build_plan(baseline, reference=reference, course_slot=slot,
                   run_id=os.environ['GITHUB_RUN_ID'], audio_sha256='', mode=os.environ['SHARD_MODE'])
        selection = baseline['selection']
        request = {'course_id': selection['course_id'], 'sub_id': selection['sub_id'], 'offset': 0, 'duration': 600}
        fetch_env = {'TEST_COURSE_SLOT': '-1', 'QWEN_ASR_TEST_REQUEST': json.dumps(request), 'LATEST_LECTURE': 'false'}
    began_fetch = time.perf_counter()
    with environment({**fetch_env,
                      'FULL_LECTURE': 'true', 'QUALITY_SAMPLE': 'false',
                      'LONG_CHUNK_SAMPLE': 'false'}):
        benchmark.fetch()  # Existing private selection and playback fallback chain.
    fetch_seconds = time.perf_counter()-began_fetch
    wav = benchmark.workspace() / 'audio.wav'
    with wave.open(str(wav), 'rb') as source:
        if (reaches_acquisition_limit(source.getnframes()/source.getframerate())
                and os.environ.get('ALLOW_PARTIAL_COMPARISON') != 'true'):
            raise ValueError('Audio reached the full-lecture acquisition cap; ASR forbidden')
    with wav.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if mode == 'private_latest':
        from src.ai.qwen_transcriber import QwenTranscriber
        from src.ai.course_glossary import course_terms
        evidence = json.loads((benchmark.workspace()/'evidence.json').read_bytes())
        selection = evidence.get('selection') or {}
        with wave.open(str(wav), 'rb') as source:
            duration = source.getnframes()/source.getframerate()
        raw = root()/'audio.raw'
        command(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(wav), '-f', 'f32le',
                 '-ac', '1', '-ar', '16000', '-y', str(raw)], timeout=300)
        transcriber = QwenTranscriber()
        with raw.open('rb') as stream:
            chunks = transcriber.prepare_pcm_stream(stream.read, lambda: True, lambda: b'', lambda: 0,
                                                    audio_path=str(raw), timeout=600)
        planning = {'selection': selection, 'audio_seconds': duration,
                    'full_chunks': [{'start': a, 'end': b} for a, b in chunks],
                    'recognition_terms': course_terms(selection.get('course_title', '')),
                    'vad_windows': transcriber.last_vad_windows}
        manifest = build_audio_plan(planning, reference=reference, course_slot=slot,
            run_id=os.environ['GITHUB_RUN_ID'], audio_sha256=digest, mode=os.environ['SHARD_MODE'],
            allow_partial=os.environ.get('ALLOW_PARTIAL_COMPARISON') == 'true')
        manifest['preparation_mode'] = 'independent_vad'
        baseline = {'source': 'pending_baseline', 'reference': reference, 'reference_evidence': evidence}
    else:
        manifest = build_plan(baseline, reference=reference, course_slot=slot,
            run_id=os.environ['GITHUB_RUN_ID'], audio_sha256=digest, mode=os.environ['SHARD_MODE'])
    with wave.open(str(wav), 'rb') as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2 or source.getframerate() != 16000:
            raise ValueError('Expected mono 16kHz PCM16 acquisition')
        if abs(source.getnframes()/16000-manifest['audio_seconds']) > 0.1:
            raise ValueError('Reacquired audio duration differs from baseline')
        for block in manifest['blocks']:
            source.setpos(round(block['start']*16000))
            data = source.readframes(block['samples'])
            if len(data) != block['samples']*2:
                raise ValueError('Reacquired block incomplete')
            chunk_wav = root() / 'chunk.wav'
            with wave.open(str(chunk_wav), 'wb') as destination:
                destination.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                destination.writeframes(data)
            flac = root() / f"chunk-{block['chunk_id']}.flac"
            command(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(chunk_wav), '-c:a', 'flac', '-y', str(flac)])
            block['flac_sha256'] = hashlib.sha256(flac.read_bytes()).hexdigest()
    validate_plan(manifest)
    payload = encoded(manifest)
    out = root() / 'out'
    seal({'manifest.json': payload}, 'plan', out / 'plan.enc')
    for shard in manifest['shards']:
        files = {'manifest.json': payload}
        files.update({f'chunk-{i}.flac': (root()/f'chunk-{i}.flac').read_bytes() for i in shard['chunk_ids']})
        seal(files, f"input-{shard['shard_id']}", out / f"shard-{shard['shard_id']}.enc")
    lecture = root() / 'lecture.flac'
    command(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(wav), '-c:a', 'flac', '-y', str(lecture)], timeout=300)
    timings = {'prepare_seconds': time.perf_counter()-started, 'fetch_seconds': fetch_seconds}
    seal({'manifest.json': payload, 'baseline.json': encoded(baseline),
          'evidence.json': encoded(baseline.get('reference_evidence', {})),
          'timings.json': encoded(timings), 'lecture.flac': lecture.read_bytes()}, 'gather-input', out/'input.enc')
    print(f"Prepared {len(manifest['blocks'])} original VAD blocks across {len(manifest['shards'])} shards; "
          f"pending audio={manifest['pending_audio_seconds']:.1f}s", flush=True)


def prepare_reuse():
    """New quota experiment from verified completed shards, without repeating ASR."""
    source_run = os.environ['REUSE_SHARDED_RUN_ID']
    if not source_run.isdigit() or source_run == os.environ['GITHUB_RUN_ID']:
        raise ValueError('Invalid ASR reuse source')
    started = time.perf_counter()
    slot = int(os.environ['COURSE_SLOT'])
    source_slot = int(os.environ.get('REUSE_SOURCE_SLOT') or slot)
    if not 0 <= source_slot < MAX_COURSES:
        raise ValueError('Invalid ASR reuse source slot')
    def download(name):
        folder = root()/'reuse'/name
        command(['gh', 'run', 'download', source_run, '--repo', os.environ['GITHUB_REPOSITORY'],
                 '--name', name, '--dir', str(folder)])
        return folder
    input_path = download(f'qwen-shard-input-{source_slot}')/'input.enc'
    partial = os.environ.get('ALLOW_PARTIAL_COMPARISON') == 'true'
    final = None
    if partial:
        listing = json.loads(subprocess.check_output(['gh', 'api',
            f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{source_run}/artifacts?per_page=100"],
            stderr=subprocess.PIPE, timeout=60))
        if listing['total_count'] > 100:
            raise ValueError('Reuse artifact listing exceeds bounded page')
        if any(a['name'] == f'qwen-shard-final-{source_slot}' for a in listing['artifacts']):
            raise ValueError('ASR-only finalization cannot reset an existing finalization or cloud quota')
    else:
        final = legacy_report(download(f'qwen-shard-final-{source_slot}')/'result.enc')
    with environment({'GITHUB_RUN_ID': source_run}):
        files = unseal(input_path, 'gather-input', slot=source_slot)
    original = json.loads(files['manifest.json'])
    validate_plan(original)
    if (original['run_id'] != source_run or original['course_slot'] != source_slot
            or original['strategy'] != os.environ['SHARD_MODE']
            or original['reference'] != {'run_id': os.environ['BASELINE_RUN_ID'], 'artifact': os.environ['BASELINE_ARTIFACT']}
            or (not partial and (final.get('source') != 'authorized_sharded_runtime' or final.get('complete') is not True
                or final.get('sharding', {}).get('plan_hash') != fingerprint(original)
                or not final.get('shard_review_complete') or not final.get('test_summary')))):
        raise ValueError('Reuse requires a completed matching experiment; keep baseline order and shard mode unchanged')
    if not partial and json.loads(files['baseline.json']).get('source') == 'pending_baseline':
        baseline = final.get('baseline_snapshot')
        if not baseline:
            raise ValueError('Completed independent experiment lacks its verified baseline snapshot')
        validate_baseline_match(original, baseline)
        files['baseline.json'] = encoded(baseline)
        files['evidence.json'] = encoded(baseline.get('reference_evidence', {}))
    manifest = {**original, 'run_id': os.environ['GITHUB_RUN_ID'], 'course_slot': slot,
                'asr_reuse_run_id': source_run, 'asr_reuse_source_slot': source_slot}
    if partial:
        manifest['allow_partial_comparison'] = True
    out = root()/'out'
    seal({'manifest.json': encoded(manifest)}, 'plan', out/'plan.enc')
    original_results = []
    for shard in original['shards']:
        n = shard['shard_id']
        audio_path = download(f'qwen-shard-audio-{source_slot}-{n}')/f'shard-{n}.enc'
        result_path = download(f'qwen-shard-result-{source_slot}-{n}')/'worker-result.enc'
        with environment({'GITHUB_RUN_ID': source_run}):
            audio = unseal(audio_path, f'input-{n}', slot=source_slot)
            result = json.loads(unseal(result_path, f'result-{n}', slot=source_slot)['result.json'])
        if json.loads(audio['manifest.json']) != original:
            raise ValueError('Reuse audio belongs to another plan')
        validate_result(original, result, n, require_complete=True)
        original_results.append(result)
        completed = {'plan_hash': fingerprint(manifest), 'shard_id': n, 'complete': True,
                     'chunks': result['chunks'], 'attempts': [], 'seconds': 0,
                     'reused_from_run_id': source_run, 'source_worker_seconds': result.get('seconds')}
        audio['manifest.json'] = encoded(manifest)
        audio['completed.json'] = encoded(completed)
        seal(audio, f'input-{n}', out/f'shard-{n}.enc')
    if json.loads(files['baseline.json']).get('source') != 'pending_baseline':
        assemble(original, original_results, json.loads(files['baseline.json']))
    elif not any(row['text'].strip() for r in original_results for row in r['chunks']):
        raise ValueError('All completed ASR blocks empty')
    files['manifest.json'] = encoded(manifest)
    files['timings.json'] = encoded({'prepare_seconds': time.perf_counter()-started,
                                     'asr_reuse_run_id': source_run, 'fetch_seconds': 0})
    seal(files, 'gather-input', out/'input.enc')
    print('Verified completed ASR reused; no new audio acquisition or model decoding', flush=True)


def workers():
    refs = parse_baselines(os.environ['BASELINES'])
    matrix = []
    course_matrix = []
    courses = set()
    for slot in range(len(refs)):
        manifest = json.loads(unseal(root()/'plans'/f'qwen-shard-plan-{slot}'/'plan.enc', 'plan', slot)['manifest.json'])
        validate_plan(manifest)
        if (manifest['course_slot'] != slot or manifest['run_id'] != os.environ['GITHUB_RUN_ID']
                or manifest['reference'] != {k: refs[slot][k] for k in ('run_id', 'artifact')}):
            raise ValueError('Course plan belongs to another slot, run or baseline')
        identity = (manifest['selection']['course_id'], manifest['selection']['sub_id'])
        if identity in courses:
            raise ValueError('Duplicate lecture in batch')
        courses.add(identity)
        local_workers = [{'course_slot': slot, 'shard_id': shard['shard_id']} for shard in manifest['shards']]
        matrix.extend(local_workers)
        course_matrix.append({'course_slot': slot, 'workers': {'include': local_workers}})
    if len(matrix) > MAX_RUNNERS:
        raise ValueError('Batch exceeds total runner cap')
    outputs(workers={'include': matrix}, courses={'include': course_matrix})
    print(f'Planned {len(matrix)} ASR workers in {len(course_matrix)} independent course pipelines; '
          f'global runner cap={MAX_RUNNERS}', flush=True)


def worker():
    import soundfile as sf
    from src.ai.qwen_transcriber import QwenTranscriber
    shard_id = int(os.environ['SHARD_ID'])
    files = unseal(root()/'inbox'/f'shard-{shard_id}.enc', f'input-{shard_id}')
    manifest = json.loads(files['manifest.json'])
    validate_plan(manifest)
    shard = manifest['shards'][shard_id]
    expected_names = {'manifest.json'} | {f'chunk-{n}.flac' for n in shard['chunk_ids']}
    if set(files) not in (expected_names, expected_names | {'completed.json'}):
        raise ValueError('Shard audio contents differ from allocation')
    prior = root()/'previous'/'worker-result.enc'
    if prior.exists():
        report = json.loads(unseal(prior, f'result-{shard_id}')['result.json'])
        done = validate_result(manifest, report, shard_id)
    elif 'completed.json' in files:
        report = json.loads(files['completed.json'])
        done = validate_result(manifest, report, shard_id,
                               require_complete=os.environ.get('QWEN_PRODUCTION_TASK') != 'true')
    else:
        report = {'plan_hash': fingerprint(manifest), 'shard_id': shard_id,
                  'complete': False, 'chunks': [], 'attempts': []}
        done = set()
    remaining = [manifest['blocks'][i] for i in shard['chunk_ids'] if i not in done]
    attempt = {'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT', '1'), 'reused_blocks': len(done),
               'decoded_chunk_ids': [], 'seconds': 0}
    report['attempts'].append(attempt)
    report['complete'] = False
    report.pop('error_type', None)
    started = time.perf_counter()
    out = root()/'out'/'worker-result.enc'

    def checkpoint(rows):
        attempt['seconds'] = time.perf_counter()-started
        attempt['decoded_chunk_ids'] = [row['chunk_id'] for row in rows]
        report['chunks'] = previous_rows + list(rows)
        report['seconds'] = sum(a['seconds'] for a in report['attempts'])
        seal({'result.json': encoded(report)}, f'result-{shard_id}', out)

    previous_rows = list(report['chunks'])
    checkpoint([])

    def load(block):
        data = files[f"chunk-{block['chunk_id']}.flac"]
        if hashlib.sha256(data).hexdigest() != block['flac_sha256']:
            raise ValueError('Audio block checksum mismatch')
        samples, rate = sf.read(io.BytesIO(data), dtype='float32')
        if rate != 16000 or samples.ndim != 1:
            raise ValueError('Unexpected shard audio format')
        return samples

    try:
        if remaining:
            transcriber = QwenTranscriber()
            transcriber.set_terms(manifest['recognition_terms'])
            transcriber.recognize_blocks(remaining, load, checkpoint=checkpoint)
        report['complete'] = True
        validate_result(manifest, report, shard_id, require_complete=True)
    except Exception as error:
        report['error_type'] = type(error).__name__
        raise
    finally:
        attempt['seconds'] = time.perf_counter()-started
        report['seconds'] = sum(a['seconds'] for a in report['attempts'])
        report['peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**3 if sys.platform == 'darwin' else 1024**2)
        seal({'result.json': encoded(report)}, f'result-{shard_id}', out)
    print(f"Shard complete: reused={len(done)}, decoded={len(remaining)}, runtime={attempt['seconds']:.1f}s", flush=True)


def restore():
    """Only a confirmed absent artifact permits starting fresh; API failures abort."""
    name = os.environ['RESTORE_ARTIFACT']
    prefix = f"qwen-shard-result-{int(os.environ['COURSE_SLOT'])}-"
    if name != f"qwen-shard-final-{int(os.environ['COURSE_SLOT'])}" and name not in [prefix+str(i) for i in range(3)]:
        raise ValueError('Invalid resume artifact name')
    repo, run = os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_RUN_ID']
    data = json.loads(subprocess.check_output(['gh', 'api',
        f'repos/{repo}/actions/runs/{run}/artifacts?per_page=100'], stderr=subprocess.PIPE, timeout=60))
    if data['total_count'] > 100:
        raise ValueError('Resume artifact listing exceeds bounded page')
    found = [a for a in data['artifacts'] if a['name'] == name]
    if not found:
        print('No previous artifact; starting the first attempt', flush=True)
        return
    if len(found) != 1 or found[0]['expired']:
        raise ValueError('Resume artifact expired or ambiguous; fresh quota forbidden')
    command(['gh', 'run', 'download', run, '--repo', repo, '--name', name,
             '--dir', str(root()/'previous')])
    filename = 'result.enc' if name.startswith('qwen-shard-final-') else 'worker-result.enc'
    source = root()/'previous'/filename
    if not source.is_file():
        raise ValueError('Existing resume artifact lacks its checkpoint; fresh quota forbidden')
    (root()/'out').mkdir(mode=0o700, exist_ok=True)
    shutil.copyfile(source, root()/'out'/filename)


def timing_snapshot(reference):
    """Capture through-summary times; final upload/cleanup remains external overhead."""
    def read(args):
        return json.loads(subprocess.check_output(['gh', 'api', *args], stderr=subprocess.PIPE, timeout=60))
    repo = os.environ['GITHUB_REPOSITORY']
    run = os.environ['GITHUB_RUN_ID']
    jobs = read([f'repos/{repo}/actions/runs/{run}/jobs?per_page=100'])['jobs']
    info = read([f'repos/{repo}/actions/runs/{run}'])
    now = datetime.now(timezone.utc)
    stamp = lambda value: datetime.fromisoformat(value.replace('Z', '+00:00'))
    durations = [{'name': j['name'], 'seconds': ((stamp(j['completed_at']) if j.get('completed_at') else now)
                 -stamp(j['started_at'])).total_seconds(), 'status': j['status']} for j in jobs if j.get('started_at')]
    base_jobs = read([f"repos/{repo}/actions/runs/{reference['run_id']}/jobs?per_page=100"])['jobs']
    artifact = reference['artifact']
    base_name = 'benchmark' if artifact == 'qwen-asr-encrypted-result' else f"benchmark ({artifact[-1]})"
    matching = [j for j in base_jobs if j['name'] == base_name and j.get('completed_at')]
    return {'snapshot_at': now.isoformat(), 'elapsed_from_dispatch_seconds': (now-stamp(info['created_at'])).total_seconds(),
            'batch_runner_seconds_through_snapshot': sum(j['seconds'] for j in durations), 'jobs': durations,
            'baseline_course_runner_seconds': sum((stamp(j['completed_at'])-stamp(j['started_at'])).total_seconds() for j in matching),
            'baseline_course_completed_successfully': len(matching) == 1 and matching[0]['conclusion'] == 'success',
            'note': 'Snapshot includes queue latency in wall time; cumulative runner time excludes queued time. '
                    'Current upload/cleanup and earlier failed workflow runs are not included.'}


def gather():
    from scripts import benchmark_qwen_asr as benchmark
    slot = int(os.environ['COURSE_SLOT'])
    files = unseal(root()/'inbox'/'input.enc', 'gather-input')
    manifest = json.loads(files['manifest.json'])
    baseline = json.loads(files['baseline.json'])
    validate_plan(manifest)
    previous = root()/'previous'/'result.enc'
    cached = legacy_report(previous) if previous.exists() else None
    if baseline.get('source') == 'pending_baseline':
        reference = manifest['reference']
        if cached and cached.get('sharding', {}).get('plan_hash') == fingerprint(manifest) and cached.get('baseline_snapshot'):
            baseline = cached['baseline_snapshot']
        else:
            command(['gh', 'run', 'download', reference['run_id'], '--repo', os.environ['GITHUB_REPOSITORY'],
                     '--name', reference['artifact'], '--dir', str(root()/'baseline')])
            baseline = legacy_report(root()/'baseline'/'result.enc')
        validate_baseline_match(manifest, baseline)
        files['evidence.json'] = encoded(baseline.get('reference_evidence', {}))
    results = [json.loads(unseal(root()/'results'/f'qwen-shard-result-{slot}-{s["shard_id"]}'/'worker-result.enc',
                               f'result-{s["shard_id"]}')['result.json']) for s in manifest['shards']]
    assembled = assemble(manifest, results, baseline)  # No cloud calls before full validation.
    if manifest.get('allow_partial_comparison'):
        reasons = []
        if not assembled['baseline_comparison']['same_audio_duration_verified']:
            reasons.append('Reacquired audio duration differs from baseline; end-to-end speedup is not a controlled comparison.')
        if reaches_acquisition_limit(manifest['audio_seconds']):
            reasons.append('Recognized input is near the acquisition cutoff; whole-lecture completeness is unverified.')
        if baseline.get('complete') is not True:
            reasons.append('The baseline ASR result is incomplete.')
        if not assembled['baseline_comparison']['same_block_timeline_verified']:
            reasons.append('VAD block timelines differ; row-by-row accuracy comparison is unavailable.')
        assembled['comparison_is_partial'] = bool(reasons)
        assembled['comparison_limitations'] = reasons
        assembled['full_lecture_complete'] = not bool(reasons)
    profile = os.environ.get('QWEN_REVIEW_PROFILE', 'production')
    if previous.exists():
        report = cached
        if report.get('sharding', {}).get('plan_hash') != fingerprint(manifest) or report.get('review_profile') != profile:
            raise ValueError('Cached finalization belongs to another plan or quota profile')
    else:
        report = assembled
        if manifest.get('preparation_mode') == 'independent_vad':
            report['baseline_snapshot'] = baseline
        report['review_profile'] = profile
        report['reference_evidence'] = json.loads(files['evidence.json'])
        report['preparation_metrics'] = json.loads(files['timings.json'])
    lecture = root()/'lecture.flac'
    lecture.write_bytes(files['lecture.flac'])
    command(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(lecture), '-y',
             str(benchmark.workspace()/'audio.wav')], timeout=300)
    (benchmark.workspace()/'evidence.json').write_bytes(files['evidence.json'])
    out = root()/'out'
    out.mkdir(mode=0o700, exist_ok=True)
    with mirrored_results(benchmark, out/'result.enc'):
        benchmark.save_encrypted(report)
        try:
            with environment({'FULL_LECTURE': 'true', 'REUSE_RUN_ID': '', 'LATEST_LECTURE': 'false'}):
                if not report.get('shard_review_complete'):
                    if report.get('shard_review_started'):
                        raise RuntimeError('Interrupted cloud review; quota state uncertain, automatic repeat forbidden')
                    report['shard_review_started'] = True
                    benchmark.save_encrypted(report)
                    benchmark.quality_review()
                    report = legacy_report(benchmark.workspace()/'result.enc')
                    report['shard_review_complete'] = True
                    benchmark.save_encrypted(report)
                if not report.get('test_summary'):
                    benchmark.generate_summary()
            report = legacy_report(benchmark.workspace()/'result.enc')
            limits = report['quality_limits']
            cloud = report['cloud_review']
            report['review_diagnostics'] = {
                'selected': len(report.get('review_suspects', [])),
                'selection_limit_reached': len(report.get('review_suspects', [])) >= limits['max_suspects'],
                'unresolved': report.get('localization', {}).get('unresolved', []),
                'remaining_seconds': max(0, limits['cloud_seconds']-cloud['attempted_audio_seconds']),
                'remaining_clips': max(0, limits['max_clips']-cloud.get('attempted_clips', cloud['completed_clips'])),
                'important_errors_uncovered': 'requires listening audit',
                'actual_correction_effect': 'requires listening audit'}
            try:
                report['workflow_timing'] = timing_snapshot(manifest['reference'])
            except Exception as error:
                report['workflow_timing'] = {'error_type': type(error).__name__}
            benchmark.save_encrypted(report)
        finally:
            shutil.copyfile(benchmark.workspace()/'result.enc', out/'result.enc')
    print('All shards verified; isolated review and summary completed, encrypted output only', flush=True)


def validate_baseline_match(manifest, baseline):
    partial = manifest.get('allow_partial_comparison') is True
    if partial:
        from src.ai.qwen_transcriber import MODEL, REVISION
        if (baseline.get('source') != 'authorized_full_runtime'
                or baseline.get('model') != MODEL or baseline.get('revision') != REVISION):
            raise ValueError('Partial comparison still requires the same production recognizer')
        build_audio_plan(baseline, reference=manifest['reference'], course_slot=manifest['course_slot'],
                         run_id=manifest['run_id'], audio_sha256='', mode=manifest['strategy'], allow_partial=True)
    else:
        build_plan(baseline, reference=manifest['reference'], course_slot=manifest['course_slot'],
                   run_id=manifest['run_id'], audio_sha256='', mode=manifest['strategy'])
    if (any(str(manifest['selection'][k]) != str(baseline['selection'][k]) for k in ('course_id', 'sub_id'))
            or (not partial and abs(manifest['audio_seconds']-baseline['audio_seconds']) > 0.1)
            or manifest['recognition_terms'] != baseline.get('recognition_terms', [])):
        raise ValueError('Independent acquisition differs from baseline lecture, duration or terminology')


def clean():
    from scripts import benchmark_qwen_asr as benchmark
    benchmark.clean()
    shutil.rmtree(root())


def main():
    mode = sys.argv[1] if len(sys.argv) == 2 else 'invalid'
    try:
        {'plan': plan, 'prepare': prepare, 'workers': workers, 'worker': worker, 'restore': restore,
         'gather': gather, 'clean': clean}[mode]()
    except Exception as error:
        if mode != 'clean':
            try:
                seal({'diagnostic.json': encoded({'mode': mode, 'error_type': type(error).__name__,
                     'private_error': str(error)[-2000:]})}, 'diagnostic', root()/'out'/'diagnostic.enc')
            except Exception:
                pass
        print(f'Shard pilot {mode} failed ({type(error).__name__}); private details withheld', flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
