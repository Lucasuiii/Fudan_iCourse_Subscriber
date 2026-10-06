"""Bounded, authenticated volumes for prepared audio; legacy QSP1 stays readable.

The ZIP is spooled to an anonymous temporary file, not duplicated in RAM.
Only encrypted parts are published, followed by the atomic encrypted manifest.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import zipfile
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAX_BYTES = 2 * 1024 * 1024 * 1024
PART_BYTES = 64 * 1024 * 1024
MAX_PARTS = 32
MAX_FILES = 600
MAX_MANIFEST = 64 * 1024


def atomic_write(path, data):
    temporary = path.with_suffix('.tmp')
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def part_name(stem, bundle, index):
    return f'{stem}-{bundle}-part-{index:04d}.enc'


def part_aad(aad, bundle, index):
    return aad + f':multipart-v1:{bundle}:part:{index}'.encode()


def validate_files(files):
    if not files or len(files) > MAX_FILES:
        raise ValueError('Prepared bundle has too many files')
    if any(not isinstance(n, str) or not n or Path(n).name != n for n in files):
        raise ValueError('Bundle paths must be flat')
    if sum(len(value) for value in files.values()) > MAX_BYTES:
        raise ValueError('Prepared bundle exceeds total size limit')


def seal(files, path, key, aad):
    path = Path(path)
    validate_files(files)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', path.stem):
        raise ValueError('Invalid prepared manifest filename')
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    bundle = os.urandom(16).hex()
    created = []
    try:
        with tempfile.TemporaryFile() as archive_file:
            with zipfile.ZipFile(archive_file, 'w', compression=zipfile.ZIP_STORED) as archive:
                for name, value in files.items():
                    archive.writestr(name, value)
            size = archive_file.tell()
            if size > MAX_BYTES or math.ceil(size / PART_BYTES) > MAX_PARTS:
                raise ValueError('Prepared bundle exceeds total size limit')
            archive_file.seek(0)
            if size <= PART_BYTES:
                # Small checkpoints keep the existing single-file contract.
                nonce = os.urandom(12)
                atomic_write(path, b'QSP1' + nonce + AESGCM(key).encrypt(
                    nonce, archive_file.read(), aad))
                return
            digest = hashlib.sha256()
            index = 0
            while payload := archive_file.read(PART_BYTES):
                digest.update(payload)
                nonce = os.urandom(12)
                target = path.parent / part_name(path.stem, bundle, index)
                atomic_write(target, b'QPP1' + nonce + AESGCM(key).encrypt(
                    nonce, payload, part_aad(aad, bundle, index)))
                created.append(target)
                index += 1
            manifest = json.dumps({'schema': 1, 'bundle': bundle, 'stem': path.stem,
                'archive_bytes': size, 'archive_sha256': digest.hexdigest(),
                'part_bytes': PART_BYTES, 'parts': index}, separators=(',', ':')).encode()
            nonce = os.urandom(12)
            atomic_write(path, b'QSP2' + nonce + AESGCM(key).encrypt(
                nonce, manifest, aad + b':multipart-v1:manifest'))
    except BaseException:
        # A failed new write never damages an existing manifest/checkpoint.
        for target in created:
            target.unlink(missing_ok=True)
        raise


def read_manifest(path, key, aad):
    if not 32 <= path.stat().st_size <= MAX_MANIFEST + 32:
        raise ValueError('Invalid prepared manifest size')
    blob = path.read_bytes()
    if blob[:4] != b'QSP2':
        raise ValueError('Invalid prepared manifest header')
    manifest = json.loads(AESGCM(key).decrypt(blob[4:16], blob[16:],
                         aad + b':multipart-v1:manifest'))
    if (not isinstance(manifest, dict) or manifest.get('schema') != 1
            or not isinstance(manifest.get('bundle'), str)
            or not re.fullmatch(r'[0-9a-f]{32}', manifest['bundle'])
            or not isinstance(manifest.get('stem'), str)
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', manifest['stem'])
            or type(manifest.get('archive_bytes')) is not int
            or not 0 < manifest['archive_bytes'] <= MAX_BYTES
            or type(manifest.get('part_bytes')) is not int
            or not 0 < manifest['part_bytes'] <= PART_BYTES
            or type(manifest.get('parts')) is not int
            or not 1 <= manifest['parts'] <= MAX_PARTS
            or manifest['parts'] != math.ceil(manifest['archive_bytes'] / manifest['part_bytes'])
            or not isinstance(manifest.get('archive_sha256'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', manifest['archive_sha256'])):
        raise ValueError('Invalid prepared manifest')
    return manifest


def unseal(path, key, aad):
    path = Path(path)
    manifest = read_manifest(path, key, aad)
    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as archive_file:
        for index in range(manifest['parts']):
            target = path.parent / part_name(manifest['stem'], manifest['bundle'], index)
            expected = min(manifest['part_bytes'],
                           manifest['archive_bytes'] - index * manifest['part_bytes'])
            if target.stat().st_size != expected + 32:
                raise ValueError('Prepared volume size differs from manifest')
            blob = target.read_bytes()
            if blob[:4] != b'QPP1':
                raise ValueError('Invalid prepared volume header')
            payload = AESGCM(key).decrypt(blob[4:16], blob[16:],
                                         part_aad(aad, manifest['bundle'], index))
            digest.update(payload)
            archive_file.write(payload)
        if digest.hexdigest() != manifest['archive_sha256']:
            raise ValueError('Prepared archive hash mismatch')
        archive_file.seek(0)
        with zipfile.ZipFile(archive_file) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (not infos or len(infos) > MAX_FILES or len(set(names)) != len(names)
                    or any(not n or Path(n).name != n for n in names)
                    or sum(info.file_size for info in infos) > MAX_BYTES):
                raise ValueError('Invalid prepared archive')
            return {name: archive.read(name) for name in names}
