"""Fetch one publisher-linked PICO sensor/optical pair, without loading pickle.

Use HTTP range requests on the public ZIP; never extract arbitrary paths or
download videos. Selection is the first lexicographic complete pair, not a
motion-quality selection. This is source intake, not robot control.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import requests

PUBLISHER = "https://github.com/Pico-AI-Team/HMD-Poser"
FILE_ID = "1xEj3J0vJilx-jPCPbbsX6a6IF9LQ5URu"
VIEW_URL = f"https://drive.google.com/file/d/{FILE_ID}/view"
ARCHIVE_SIZE = 497077901
PAIR_NAMES = ("gt_body_parms.pt", "hmd_sensor_data.pt")
MAX_MEMBER_SIZE = 64 * 1024 * 1024
MAX_TRANSFER = 128 * 1024 * 1024


def select_pair(infos):
    """Reject unsafe/ambiguous entries; select by names before reading payloads."""
    by_name = {}
    for entry in infos:
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
            raise ValueError("unsafe archive member path")
        if entry.filename in by_name:
            raise ValueError("duplicate archive member")
        by_name[entry.filename] = entry
    parents = sorted(
        {
            str(PurePosixPath(name).parent)
            for name in by_name
            if PurePosixPath(name).name == PAIR_NAMES[0]
            and str(PurePosixPath(name).with_name(PAIR_NAMES[1])) in by_name
        }
    )
    if not parents:
        raise ValueError("no complete optical/sensor pair in public archive")
    selected = [by_name[str(PurePosixPath(parents[0]) / name)] for name in PAIR_NAMES]
    for entry in selected:
        if (
            entry.is_dir()
            or entry.flag_bits & 1
            or not 0 < entry.file_size <= MAX_MEMBER_SIZE
            or not 0 < entry.compress_size <= MAX_MEMBER_SIZE
            or entry.file_size > 1000 * entry.compress_size
        ):
            raise ValueError("selected pair violates bounded archive intake")
    return parents[0], selected, len(parents)


class PublicRangeZip(io.RawIOBase):
    def __init__(self, session, url):
        self.session, self.url, self.position = session, url, 0
        self.transferred = 0
        self.requests_log = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = offset + (self.position if whence == 1 else ARCHIVE_SIZE if whence == 2 else 0)
        if whence not in (0, 1, 2) or not 0 <= value <= ARCHIVE_SIZE:
            raise ValueError("archive seek outside published size")
        self.position = value
        return value

    def read(self, size=-1):
        size = ARCHIVE_SIZE - self.position if size < 0 else min(size, ARCHIVE_SIZE - self.position)
        if size == 0:
            return b""
        if size > MAX_MEMBER_SIZE or self.transferred + size > MAX_TRANSFER:
            raise ValueError("range request exceeds bounded source intake")
        start, stop = self.position, self.position + size - 1
        with self.session.get(
            self.url, headers={"Range": f"bytes={start}-{stop}"}, stream=True, timeout=45
        ) as response:
            if response.status_code != 206 or response.headers.get("Content-Range") != (
                f"bytes {start}-{stop}/{ARCHIVE_SIZE}"
            ):
                raise ValueError("public ZIP server did not honor exact bounded range")
            result = response.raw.read(size + 1)
            if len(result) != size:
                raise ValueError("public ZIP range length differs")
        self.position += size
        self.transferred += size
        self.requests_log.append(dict(start=start, stop=stop, sha256=hashlib.sha256(result).hexdigest()))
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_directory.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("PICO FreeDancing intake refuses overwrite")
    with requests.Session() as session:
        # Public download confirmation is Google's file-size/virus-scan notice,
        # not a sign-in, credentials, entitlement, or access-control workaround.
        page = session.get(
            "https://drive.usercontent.google.com/uc",
            params={"id": FILE_ID, "export": "download"},
            timeout=30,
        )
        page.raise_for_status()
        if len(page.content) > 100000:
            raise ValueError("unexpected public download notice")
        uuid = re.search(r'name="uuid" value="([a-zA-Z0-9-]+)"', page.text)
        if uuid is None or "Google Drive can't scan this file for viruses" not in page.text:
            raise ValueError("public archive requires a different access flow")
        request = requests.Request(
            "GET",
            "https://drive.usercontent.google.com/download",
            params={"id": FILE_ID, "export": "download", "confirm": "t", "uuid": uuid.group(1)},
        ).prepare()
        remote = PublicRangeZip(session, request.url)
        with zipfile.ZipFile(remote) as archive:
            pair, selected, pair_count = select_pair(archive.infolist())
            print(
                json.dumps(
                    {
                        "selected_pair": pair,
                        "complete_pairs": pair_count,
                        "members": [{"name": i.filename, "bytes": i.file_size} for i in selected],
                    }
                ),
                flush=True,
            )
            output.mkdir(parents=True, exist_ok=False)
            rows = []
            for entry in selected:
                data = archive.read(entry)  # zipfile verifies CRC; no deserialization.
                path = output / PurePosixPath(entry.filename).name
                with path.open("xb") as stream:
                    stream.write(data)
                rows.append(
                    dict(
                        member=entry.filename,
                        file=path.name,
                        bytes=len(data),
                        zip_crc32=entry.CRC,
                        sha256=hashlib.sha256(data).hexdigest(),
                    )
                )
    report = dict(
        kind="pico_freedancing_one_public_pair_intake_v1",
        publisher=PUBLISHER,
        publisher_download=VIEW_URL,
        published_archive_size_bytes=ARCHIVE_SIZE,
        selection="first_lexicographic_complete_pair_before_payload_inspection",
        selected_pair=pair,
        complete_pairs=pair_count,
        members=rows,
        downloaded_bytes=remote.transferred,
        range_requests=remote.requests_log,
        range_crc_and_local_sha_verified=True,
        publisher_whole_archive_sha_available=False,
        sensor_and_optical_provenance="publisher_claim_not_independent_sensor_verification",
        pickle_or_tensor_payload_loaded=False,
        source_frame_rate_verified=False,
        raw_xr24_sdk_compatibility_claimed=False,
        retargeting_run=False,
        policy_run=False,
        physics_integrated=False,
        hardware_authorized=False,
        deployment_ready=False,
    )
    with (output / "download.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(output), "downloaded_bytes": remote.transferred, "members": rows}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
