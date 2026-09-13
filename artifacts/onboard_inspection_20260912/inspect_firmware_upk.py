"""Inspect public vendor firmware locally. Never install or execute firmware.

Uses only the key derivation and block decoder from the locally reviewed
UniTEABag reference. Its CLI and archive extraction functions are never called.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import urllib.request
import warnings

FOLDER = Path(r'E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1')
URL = 'https://unitree-firmware.oss-cn-hangzhou.aliyuncs.com/firmware/release/package_1.4.5.0_G1_Edu%2B_1759976671033.upk'


class FirmwareReader(io.RawIOBase):
    def __init__(self):
        self.position = 0
        self.network_bytes = 0
        self.network_reads = 0
        spec = importlib.util.spec_from_file_location('reviewed_upk_reference', FOLDER / 'UniTEABag_reference.py')
        self.decoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.decoder)
        header = self.read_encrypted(0, 116)
        if header[:5] != b'UTPK\x00' or header[112:116] != b'TEA\x00':
            raise ValueError('Unexpected UPK header')
        self.size = int.from_bytes(header[16:24], 'little') - 4
        self.package = header[48:112].rstrip(b'\x00').decode()
        self.key = self.decoder.generate_code_key(header[28:32])[1]
        sample = self.read(512)
        tarfile.TarInfo.frombuf(sample, 'utf-8', 'surrogateescape')
        self.seek(0)

    def read_encrypted(self, offset, count):
        for path in [FOLDER / 'g1_eduplus_1.4.5.0.upk', FOLDER / 'g1_eduplus_1.4.5.0.upk.part']:
            if path.exists() and path.stat().st_size >= offset + count:
                with path.open('rb') as stream:
                    stream.seek(offset)
                    data = stream.read(count)
                if len(data) == count:
                    return data
        request = urllib.request.Request(URL, headers={'Range': f'bytes={offset}-{offset + count - 1}'})
        with urllib.request.urlopen(request, timeout=25) as response:
            if response.status != 206:
                raise RuntimeError('Server did not honor bounded byte range')
            data = response.read(count + 1)
        if len(data) != count:
            raise RuntimeError('Incomplete or excessive range response')
        self.network_reads += 1
        self.network_bytes += len(data)
        return data

    def read(self, size=-1):
        if size < 0 or size > 64 * 1024 * 1024:
            raise ValueError('Only bounded firmware reads are allowed')
        size = min(size, max(0, self.size - self.position))
        if not size:
            return b''
        start = self.position // 8 * 8
        end = (self.position + size + 7) // 8 * 8
        encrypted = self.read_encrypted(116 + start, end - start)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            plain = self.decoder.decrypt_chunk_np(encrypted, self.key)
        result = plain[self.position - start:self.position - start + size]
        self.position += len(result)
        return result

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset
        if position < 0 or position > self.size:
            raise ValueError('Firmware seek out of bounds')
        self.position = position
        return position

    def tell(self):
        return self.position

    def readable(self):
        return True

    def seekable(self):
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--copy-module', choices=['ai_sport', 'state_estimator', 'motion_switcher', 'g1_arm_example'])
    args = parser.parse_args()
    reader = FirmwareReader()
    with tarfile.open(fileobj=reader, mode='r:') as archive:
        members = archive.getmembers()
        inventory = [{'name': m.name, 'bytes': m.size, 'regular_file': m.isfile()} for m in members]
        (FOLDER / 'firmware_index.json').write_text(json.dumps({'package': reader.package, 'members': inventory}, indent=2))
        print('Package:', reader.package, flush=True)
        for member in members:
            if member.isfile():
                print(member.name, member.size, flush=True)
        if args.copy_module:
            candidates = [m for m in members if m.isfile() and f'/module/{args.copy_module}/' in m.name and m.name.endswith('.upk')]
            if len(candidates) != 1:
                raise RuntimeError(f'Expected one module, found {len(candidates)}')
            member = candidates[0]
            destination = FOLDER / (args.copy_module + '.upk')
            partial = FOLDER / (args.copy_module + '.upk.partial')
            digest = hashlib.sha256()
            with archive.extractfile(member) as source, partial.open('wb') as output:
                while True:
                    chunk = source.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
            if partial.stat().st_size != member.size:
                raise RuntimeError('Extracted module has wrong length')
            partial.replace(destination)
            print('Copied local module:', destination, 'SHA256', digest.hexdigest(), flush=True)
    print('Additional remote range bytes:', reader.network_bytes, 'requests:', reader.network_reads)


if __name__ == '__main__':
    main()
