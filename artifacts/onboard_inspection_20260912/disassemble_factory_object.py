"""Read named ARM64 functions and relocations; never execute robot code."""
import argparse
from pathlib import Path
import sys
fw=Path(r'E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1')
sys.path.insert(0,str(fw/'python_deps'))
from elftools.elf.elffile import ELFFile
from capstone import Cs,CS_ARCH_ARM64,CS_MODE_ARM
ap=argparse.ArgumentParser();ap.add_argument('object');ap.add_argument('patterns',nargs='+');ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
p=fw/'ai_sport_files/ai_sport_8.4.2.222/module/ai_sport/file/unitree/module/ai_sport/build/fsm/CMakeFiles/fsm.dir'/args.object
rows=[]
with p.open('rb') as stream:
    elf=ELFFile(stream);symbols=elf.get_section_by_name('.symtab');md=Cs(CS_ARCH_ARM64,CS_MODE_ARM)
    for symbol in symbols.iter_symbols():
        if symbol['st_info']['type']!='STT_FUNC' or not symbol['st_size'] or not any(x in symbol.name for x in args.patterns):continue
        section=elf.get_section(symbol['st_shndx']);address=symbol['st_value'];size=symbol['st_size']
        relocs={}
        for rel in elf.iter_sections():
            if rel['sh_type']=='SHT_RELA' and rel['sh_info']==symbol['st_shndx']:
                table=elf.get_section(rel['sh_link'])
                for r in rel.iter_relocations():
                    target=table.get_symbol(r['r_info_sym']);note=target.name or str(target['st_shndx'])
                    if isinstance(target['st_shndx'],int):
                        dest=elf.get_section(target['st_shndx']);data=dest.data();offset=target['st_value']+r['r_addend']
                        if dest.name.startswith('.rodata'):
                            note+=' '+repr(data[max(0,offset):max(0,offset)+48])
                    relocs[r['r_offset']]=note+f" addend={r['r_addend']}"
        rows.append('\n'+symbol.name)
        for ins in md.disasm(section.data()[address:address+size],address):
            rows.append(f'{ins.address:06x} {ins.mnemonic:8} {ins.op_str:55} {relocs.get(ins.address,"")}')
args.output.write_text('\n'.join(rows))
print(args.output)
