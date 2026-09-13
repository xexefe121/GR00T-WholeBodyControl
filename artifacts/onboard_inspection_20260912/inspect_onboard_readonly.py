"""Read software inventory over SSH; never publish DDS or change robot services."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import socket
import time

import os
import paramiko


REMOTE_INVENTORY = r'''
import json, os, pathlib, subprocess

def read(path, limit=12000):
    try:
        return pathlib.Path(path).read_text(errors='replace')[:limit].replace('\x00', '')
    except OSError as exc:
        return {'error': str(exc)}

def run(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=12)
        return {'returncode': result.returncode, 'stdout': result.stdout[:80000], 'stderr': result.stderr[:2000]}
    except Exception as exc:
        return {'error': str(exc)}

def entries(path):
    try:
        result = []
        for p in sorted(pathlib.Path(path).iterdir()):
            if p.name.startswith('.'):
                continue
            try:
                s = p.lstat()
                result.append({'name': p.name, 'directory': p.is_dir(), 'symlink': p.is_symlink(), 'bytes': s.st_size})
            except OSError:
                pass
        return result[:300]
    except OSError as exc:
        return {'error': str(exc)}

result = {
    'hostname': read('/etc/hostname'),
    'os_release': read('/etc/os-release'),
    'board': read('/proc/device-tree/model'),
    'jetson_release': read('/etc/nv_tegra_release'),
    'uname': run(['uname', '-a']),
    'identity': run(['id']),
    'network': run(['ip', '-j', '-4', 'addr', 'show']),
    'process_names': run(['ps', '-eo', 'pid,user,comm', '--sort=comm']),
    'services': run(['systemctl', 'list-units', '--type=service', '--state=running', '--no-pager', '--no-legend']),
    'directories': {p: entries(p) for p in ['/home/unitree', '/opt', '/usr/local', '/etc/systemd/system']},
}
print(json.dumps(result, indent=2))
'''

REMOTE_CONTROLLER_DETAILS = r'''
import json, os, pathlib, re, subprocess, time
roots = ['/home/unitree/unitree', '/opt/ota_package', '/home/unitree/g1plus_pc4_unitree_install', '/home/unitree/deploy', '/home/unitree/g1_true23_onboard']
result = {'roots': {}, 'services': {}, 'processes': []}
suffixes = {'.onnx','.pt','.pth','.jit','.engine','.plan','.mnn','.rknn','.bin','.yaml','.yml','.json','.toml','.ini','.cfg','.xml','.sh'}
deadline = time.monotonic() + 20
for root in roots:
    matches, directories, errors = [], [], []
    visited = 0
    def onerror(exc): errors.append(str(exc))
    for current, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        depth = len(pathlib.Path(current).relative_to(root).parts)
        dirs[:] = [d for d in dirs if d not in {'.git','__pycache__','node_modules','.venv','venv','site-packages'} and not d.startswith('.')]
        if depth < 3: directories.append(current)
        if depth >= 6: dirs[:] = []
        for name in files:
            visited += 1
            p = pathlib.Path(current) / name
            if p.suffix.lower() in suffixes or re.search(r'policy|loco|sport|motion|controller|deploy|version',name,re.I):
                try:
                    stat = p.lstat()
                    matches.append({'path': str(p), 'bytes': stat.st_size, 'mtime': stat.st_mtime})
                except OSError as exc: errors.append(str(exc))
        if visited > 20000 or len(matches) > 1200 or time.monotonic() > deadline:
            errors.append('bounded scan stopped'); break
    result['roots'][root] = {'matches': matches, 'directories': directories, 'errors': errors, 'visited': visited}
for service in ['master_service.service','ota_pipe.service','robot-ai.service']:
    run = subprocess.run(['systemctl','show',service,'--property=FragmentPath,ExecStart,WorkingDirectory,MainPID'],capture_output=True,text=True,timeout=5)
    # Do not output environment variables or authentication material.
    result['services'][service] = re.sub(r'(?i)((?:token|password|api[_-]?key)[= :]+)[^ ;\n]+',r'\1[REDACTED]',run.stdout)
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit(): continue
    try:
        comm = (p/'comm').read_text().strip()
        if re.search(r'master_service|ota_pipe|sport|loco|motion|controller|python|main_process',comm,re.I):
            item = {'pid': int(p.name), 'comm': comm}
            for key in ['exe','cwd']:
                try: item[key] = os.readlink(p/key)
                except OSError as exc: item[key] = str(exc)
            result['processes'].append(item)
    except OSError: pass
print(json.dumps(result,indent=2))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.123.164')
    parser.add_argument('--username', default='unitree')
    parser.add_argument('--source-address', help='Bind SSH to this laptop address, useful for mirrored WSL networking.')
    parser.add_argument('--connect-address', help='Another observed address of the same robot; verify against --host host key.')
    parser.add_argument('--controller-details', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'pc2_inventory.json')
    args = parser.parse_args()
    known_hosts = Path(__file__).parent / 'known_hosts'
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    if known_hosts.exists():
        client.load_host_keys(str(known_hosts))

    class RecordNewHost(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            digest = base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
            print(f'First connection: {hostname} {key.get_name()} SHA256:{digest}', flush=True)
            client.get_host_keys().add(hostname, key.get_name(), key)
            client.save_host_keys(str(known_hosts))

    client.set_missing_host_key_policy(RecordNewHost())
    try:
        # Configure SSH credentials through UNITREE_SSH_PASSWORD.
        # Do not save this password in inventory, attempt alternate passwords, or escalate.
        connection = socket.create_connection((args.connect_address or args.host, 22), timeout=5,
                       source_address=(args.source_address, 0) if args.source_address else None)
        client.connect(args.host, username=args.username, password=os.environ['UNITREE_SSH_PASSWORD'], sock=connection,
                       allow_agent=False, look_for_keys=False, timeout=5,
                       auth_timeout=8, banner_timeout=5)
        key = client.get_transport().get_remote_server_key()
        client.get_host_keys().add(args.host, key.get_name(), key)
        client.save_host_keys(str(known_hosts))
        stdin, stdout, stderr = client.exec_command('python3 -', timeout=40)
        stdin.write(REMOTE_CONTROLLER_DETAILS if args.controller_details else REMOTE_INVENTORY)
        stdin.channel.shutdown_write()
        payload = stdout.read().decode('utf-8', 'replace')
        errors = stderr.read().decode('utf-8', 'replace')
        status = stdout.channel.recv_exit_status()
        if status:
            raise RuntimeError(f'Read-only inventory exited {status}: {errors[:2000]}')
        inventory = json.loads(payload)
        inventory['inspection'] = {'host': args.host, 'username': args.username,
                                   'unix_time': time.time(), 'read_only': True}
        args.output.write_text(json.dumps(inventory, indent=2), encoding='utf-8')
        if not args.quiet:
            print(json.dumps(inventory, indent=2))
        print(f'Saved {args.output}')
    finally:
        client.close()


if __name__ == '__main__':
    main()
