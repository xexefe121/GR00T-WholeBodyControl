"""Decode local FMX assets in isolated ARM emulation. No robot or OS execution.

Only three pure data functions from the public firmware's bridge_test ELF are
called. External calls are restricted to in-memory allocation/copy/free. Every
decoded file must match the eight MD5 bytes recorded in its FMX header.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import struct
import sys

ROOT = Path(r"E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1")
sys.path.insert(0, str(ROOT / "python_deps"))
from elftools.elf.elffile import ELFFile
from unicorn import Uc, UC_ARCH_ARM64, UC_MODE_ARM, UC_HOOK_CODE
from unicorn.arm64_const import UC_ARM64_REG_X0, UC_ARM64_REG_X1, UC_ARM64_REG_X2, UC_ARM64_REG_X3, UC_ARM64_REG_X30, UC_ARM64_REG_SP, UC_ARM64_REG_PC

BASE = ROOT / "ai_sport_files/ai_sport_8.4.2.222/module/ai_sport/file/unitree/module/ai_sport"


class LocalDecoder:
    def __init__(self):
        elf = ELFFile(io.BytesIO((BASE / "build/common/bridge_test").read_bytes()))
        self.uc = u = Uc(UC_ARCH_ARM64, UC_MODE_ARM)
        u.mem_map(0, 0x300000)
        u.mem_map(0x1000000, 0x1000000)
        u.mem_map(0x3000000, 0x100000)
        for seg in elf.iter_segments():
            if seg["p_type"] == "PT_LOAD":
                u.mem_write(seg["p_vaddr"], seg.data())
        self.symbols = {s.name: s["st_value"] for s in elf.get_section_by_name(".symtab").iter_symbols() if s["st_shndx"] != "SHN_UNDEF"}
        # Resolve only data relocations needed by the three selected routines.
        for sec in elf.iter_sections():
            if sec["sh_type"] != "SHT_RELA":
                continue
            symtab = elf.get_section(sec["sh_link"])
            for rel in sec.iter_relocations():
                if rel["r_info_type"] == 1027:  # R_AARCH64_RELATIVE
                    u.mem_write(rel["r_offset"], struct.pack("<Q", rel["r_addend"]))
                elif rel["r_info_sym"]:
                    s = symtab.get_symbol(rel["r_info_sym"])
                    if s.name == "__stack_chk_guard":
                        u.mem_write(rel["r_offset"], struct.pack("<Q", 0x1008000))
                    elif s["st_shndx"] != "SHN_UNDEF" and rel["r_info_type"] in (257, 1025):
                        u.mem_write(rel["r_offset"], struct.pack("<Q", s["st_value"] + rel["r_addend"]))
        self.heap = 0x1800000
        # Reviewed PLT entries in this exact ELF. Unknown external calls fail.
        u.hook_add(UC_HOOK_CODE, self.external, begin=0x55000, end=0x58000)

    def external(self, u, address, size, data):
        x0, x1, x2 = (u.reg_read(r) for r in (UC_ARM64_REG_X0, UC_ARM64_REG_X1, UC_ARM64_REG_X2))
        if address == 0x567c0:  # memcpy
            if x2 > 0x400000:
                raise ValueError("oversized emulated copy")
            u.mem_write(x0, bytes(u.mem_read(x1, x2)))
        elif address == 0x56630:  # operator new[]
            if x0 > 0x400000 or self.heap + x0 >= 0x2000000:
                raise ValueError("oversized emulated allocation")
            u.reg_write(UC_ARM64_REG_X0, self.heap)
            self.heap += (x0 + 15) & ~15
        elif address == 0x563e0:  # operator delete[]
            pass
        else:
            raise RuntimeError(f"unapproved external call: {address:#x}")
        u.reg_write(UC_ARM64_REG_PC, u.reg_read(UC_ARM64_REG_X30))

    def call(self, name, *args):
        u = self.uc
        for reg, value in zip((UC_ARM64_REG_X0, UC_ARM64_REG_X1, UC_ARM64_REG_X2, UC_ARM64_REG_X3), args):
            u.reg_write(reg, value)
        u.reg_write(UC_ARM64_REG_SP, 0x30ff000)
        u.reg_write(UC_ARM64_REG_X30, 0x2f0000)
        u.emu_start(self.symbols[name], 0x2f0000, timeout=20_000_000, count=100_000_000)
        if u.reg_read(UC_ARM64_REG_PC) != 0x2f0000:
            raise RuntimeError("decoder exceeded bounded emulation")

    def decode(self, blob):
        if blob[:4] != b"FMX\x01":
            raise ValueError("unexpected FMX header")
        seed, size = struct.unpack("<II", blob[4:12])
        if size != len(blob) - 20 or size > 0x400000:
            raise ValueError("invalid local asset length")
        for version, name in [(1, "_ZN7unitree8security14GenerateGenKeyEjPhj"), (2, "_ZN7unitree8security15GenerateGenKey2EjPhj")]:
            self.heap = 0x1800000
            self.call(name, seed, 0x1010000, 937)
            self.call("_ZN8Blowfish6SetKeyEPKhi", 0x1020000, 0x1010000, 937)
            self.uc.mem_write(0x1100000, blob[20:])
            self.call("_ZNK8Blowfish7DecryptEPhPKhi", 0x1020000, 0x1500000, 0x1100000, size)
            clear = bytes(self.uc.mem_read(0x1500000, size))
            if hashlib.md5(clear).digest()[:8] == blob[12:20]:
                return clear, version
        raise ValueError("decoded content did not match FMX checksum")


def main():
    decoder = LocalDecoder()
    files = list(BASE.rglob("*.yaml"))
    if len(sys.argv) > 1:
        files = [BASE / sys.argv[1]]
    out = ROOT / "decoded_configs"
    rows = []
    for path in files:
        rel = path.relative_to(BASE)
        clear, ver = decoder.decode(path.read_bytes())
        clear.decode("utf-8")  # Must be text before saving as YAML.
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(clear)
        rows.append({"path": str(rel), "bytes": len(clear), "version": ver, "checksum_verified": True})
        print(json.dumps(rows[-1]), flush=True)
    (out / "decode_manifest.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
