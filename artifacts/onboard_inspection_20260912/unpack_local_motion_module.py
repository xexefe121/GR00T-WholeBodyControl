"""Decode a downloaded UPK and copy regular files into a local analysis folder.

No network, robot access, firmware installation, or execution of extracted code.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import warnings

FOLDER = Path(r'E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('module', choices=['ai_sport', 'state_estimator', 'motion_switcher', 'g1_arm_example'])
    args = parser.parse_args()
    source = FOLDER / (args.module + '.upk')
    destination = (FOLDER / (args.module + '_files')).resolve()
    destination.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location('reviewed_upk_reference', FOLDER / 'UniTEABag_reference.py')
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    with source.open('rb') as stream:
        header = stream.read(112)
        if header[:5] != b'UTPK\x00':
            raise ValueError('Invalid UPK header')
        payload_size = int.from_bytes(header[16:24], 'little')
        if payload_size != source.stat().st_size - 112:
            raise ValueError('UPK size mismatch')
        digest = hashlib.md5()
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
        if digest.digest() != header[32:48]:
            raise ValueError('UPK payload checksum mismatch')
        stream.seek(112)
        if stream.read(4) != b'TEA\x00':
            raise ValueError('Unexpected payload type')
        first = stream.read(min(65536, payload_size - 4))
        selected = None
        for number, key in enumerate(decoder.generate_code_key(header[28:32]), 1):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                plain = decoder.decrypt_chunk_np(first, key)
            try:
                with tarfile.open(fileobj=io.BytesIO(plain), mode='r|*') as archive:
                    archive.next()
                selected = (number, key)
                break
            except tarfile.TarError:
                continue
        if selected is None:
            raise ValueError('Neither known decoder produced a tar header')
        number, key = selected
        stream.seek(116)
        tar_path = FOLDER / (args.module + '.tar')
        with tar_path.open('wb') as output:
            while chunk := stream.read(4 * 1024 * 1024):
                if len(chunk) % 8:
                    raise ValueError('Unaligned encrypted payload')
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    output.write(decoder.decrypt_chunk_np(chunk, key))
    print('Checksum and decryption verified:', source.name, flush=True)
    records = []
    total = 0
    with tarfile.open(tar_path, mode='r:*') as archive:
        for member in archive:
            entry = {'name': member.name, 'bytes': member.size, 'kind': member.type.decode('ascii', 'replace')}
            records.append(entry)
            if not member.isfile():
                entry['copied'] = False
                continue
            parts = PurePosixPath(member.name.lstrip('/')).parts
            if not parts or any(p in {'.', '..'} or '\\' in p or ':' in p or '\x00' in p for p in parts):
                raise ValueError('Unsafe archive member path')
            target = destination.joinpath(*parts).resolve()
            target.relative_to(destination)
            total += member.size
            if member.size > 2 * 1024**3 or total > 8 * 1024**3:
                raise ValueError('Archive size exceeds analysis limit')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as data, target.open('wb') as output:
                shutil.copyfileobj(data, output, 1024 * 1024)
            if target.stat().st_size != member.size:
                raise ValueError('Extracted file length mismatch')
            entry['copied'] = True
            entry['local_path'] = str(target)
    report = {'module': args.module, 'package_name': header[48:112].rstrip(b'\x00').decode(),
              'payload_md5_verified': True, 'decoder_version': number,
              'total_regular_bytes': total, 'files': records, 'robot_modified': False}
    (FOLDER / (args.module + '_index.json')).write_text(json.dumps(report, indent=2))
    print('Regular files copied:', sum(bool(r['copied']) for r in records), 'bytes:', total, flush=True)
    for record in records:
        if record['copied']:
            print(record['name'], record['bytes'], flush=True)


if __name__ == '__main__':
    main()
