"""Read a completed state artifact; return only summary and terminology audit.

The database key stays on the runner. Output is encrypted for an ephemeral
recipient public key; no SMTP, model calls, audio fetch, or database publication.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PREFIX = b'ICEX1'
MAX_EXPORT = 2 * 1024 * 1024


def identity(run, slot):
    if not str(run).isascii() or not str(run).isdigit() or type(slot) is not int or not 0 <= slot < 256:
        raise ValueError('Invalid export identity')
    return f'icourse-summary-export-v1:{run}:{slot}'.encode()


def derived_key(private, public, aad):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=hashlib.sha256(aad).digest(),
                info=PREFIX).derive(private.exchange(public))


def encrypt(payload, recipient, run, slot):
    if len(payload) > MAX_EXPORT: raise ValueError('Summary export too large')
    aad = identity(run, slot)
    recipient = X25519PublicKey.from_public_bytes(recipient)
    ephemeral = X25519PrivateKey.generate()
    public = ephemeral.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    nonce = os.urandom(12)
    return PREFIX + public + nonce + AESGCM(derived_key(ephemeral, recipient, aad)).encrypt(nonce, payload, aad)


def decrypt(blob, private, run, slot):
    if blob[:5] != PREFIX or not 65 <= len(blob) <= MAX_EXPORT + 65:
        raise ValueError('Invalid summary export')
    aad = identity(run, slot)
    private = X25519PrivateKey.from_private_bytes(private)
    public = X25519PublicKey.from_public_bytes(blob[5:37])
    return AESGCM(derived_key(private, public, aad)).decrypt(blob[37:49], blob[49:], aad)


def summary_payload(files):
    spec = json.loads(files['specification.json']); review = json.loads(files['review.json'])
    from src.ai.qwen_review_ledger import validate_ledger
    validate_ledger(review)
    course, lecture = spec['course_id'], spec['lecture']
    # Backup snapshots retain WAL header flags. deserialize() into :memory:
    # cannot open their journal; use a private temporary, read-only file.
    workspace = tempfile.TemporaryDirectory(prefix='icourse-summary-export-')
    path = Path(workspace.name)/'state.db'; path.write_bytes(files['database.db'])
    conn = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True); conn.row_factory = sqlite3.Row
    try:
        row = conn.execute('SELECT * FROM lectures WHERE sub_id=? AND course_id=?',
                           (str(lecture['sub_id']), str(course))).fetchone()
        if not row or not row['processed_at'] or not row['summary'] or row['deleted_at']:
            raise ValueError('Completed summary unavailable')
        plan = spec.get('plan', {})
        terms = plan.get('recognition_terms', spec.get('material', {}).get('recognition_terms', []))
        attempts = review.get('attempts', [])
        return {
            'course_title': spec['course_title'], 'date': lecture.get('date'),
            'sub_title': lecture.get('sub_title'), 'summary': row['summary'],
            'summary_model': row['summary_model'], 'recognition_terms': terms,
            'automatic_terms_saved': bool(conn.execute('SELECT 1 FROM meta WHERE key=?',
                ('auto_glossary:'+str(course)+':'+str(lecture['sub_id']),)).fetchone()),
            'review': {'seconds': review.get('seconds', 0), 'clips': len(attempts),
                'weak_clips': sum(a['interval'].get('kind') == 'weak' for a in attempts),
                'suspect_clips': sum(a['interval'].get('kind') != 'weak' for a in attempts),
                'unresolved_count': len(review.get('unresolved', [])),
                'variant_count': len(review.get('material', {}).get('variants', [])),
                'complete': bool(review.get('complete')), 'failed': bool(review.get('failed'))},
        }
    finally:
        conn.close()
        workspace.cleanup()


def export():
    from scripts import production_qwen as pipeline
    from scripts import sharded_qwen_pilot as shards
    run, slot = os.environ['SOURCE_RUN_ID'], int(os.environ['SOURCE_SLOT'])
    identity(run, slot)
    recipient = base64.b64decode(os.environ['RECIPIENT_PUBLIC_KEY'], validate=True)
    if len(recipient) != 32: raise ValueError('Invalid recipient public key')
    info = json.loads(subprocess.check_output(['gh','api',
        f'repos/{os.environ["GITHUB_REPOSITORY"]}/actions/runs/{run}'], stderr=subprocess.PIPE, timeout=60))
    if (info['status'] != 'completed' or info['conclusion'] != 'success'
            or info['path'].split('@')[0] != '.github/workflows/parallel_pilot.yml'):
        raise ValueError('Source is not a completed production pilot')
    target = pipeline.root()/'export-source'
    pipeline.artifact(f'qwen-production-state-{slot}', target, run=run, required=True)
    with shards.environment({'GITHUB_RUN_ID':run, 'COURSE_SLOT':str(slot)}):
        files = pipeline.decode(target/'state.enc', 'state')
    payload = json.dumps(summary_payload(files), ensure_ascii=False, allow_nan=False).encode()
    pipeline.out('summary-export.enc').write_bytes(encrypt(payload, recipient, run, slot))
    print('Completed summary exported encrypted; no models, audio acquisition, email or publication')


if __name__ == '__main__':
    os.environ['QWEN_PRODUCTION_TASK'] = 'true'
    try:
        export()
    except Exception as error:
        print(f'Summary export failed ({type(error).__name__}); private details withheld')
        sys.exit(1)
