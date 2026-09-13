"""Run from local WSL. Restore this laptop's temporary address and inventory G1.

Remote payloads only read software metadata. No mode changes, motion commands,
service changes, package installations, or writes to robot files.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ADDRESS = '192.168.123.223'
ROBOT = '192.168.123.164'


def ensure_local_address():
    addresses = json.loads(subprocess.check_output(['ip', '-j', '-4', 'addr', 'show', 'dev', 'eth0']))
    if not addresses or addresses[0].get('operstate') != 'UP':
        raise RuntimeError('This laptop Ethernet link is down; reconnect its cable before robot SSH. No network configuration changed.')
    if any(a.get('local') == ADDRESS for entry in addresses for a in entry.get('addr_info', [])):
        return
    # Probe only our proposed address; never change a peer's address or ARP table.
    mac = bytes.fromhex(Path('/sys/class/net/eth0/address').read_text().strip().replace(':', ''))
    target = socket.inet_aton(ADDRESS)
    with socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0806)) as sock:
        sock.bind(('eth0', 0))
        arp = struct.pack('!HHBBH', 1, 0x0800, 6, 4, 1) + mac + bytes(4) + bytes(6) + target
        for _ in range(2):
            sock.send(b'\xff' * 6 + mac + b'\x08\x06' + arp)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if not select.select([sock], [], [], max(0, deadline - time.monotonic()))[0]:
                    break
                packet = sock.recv(2048)
                if (len(packet) >= 42 and packet[12:14] == b'\x08\x06'
                        and packet[22:28] != mac
                        and (packet[28:32] == target
                             or (packet[28:32] == bytes(4) and packet[38:42] == target))):
                    raise RuntimeError(f'{ADDRESS} is in use or being claimed; no address added.')
    subprocess.run(['ip', 'addr', 'add', ADDRESS + '/24', 'dev', 'eth0'], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['inventory', 'controller', 'factory'], default='inventory')
    args = parser.parse_args()
    tree = ast.parse((ROOT / 'inspect_onboard_readonly.py').read_text())
    constant = 'REMOTE_INVENTORY' if args.stage == 'inventory' else 'REMOTE_CONTROLLER_DETAILS'
    payload = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == constant for t in node.targets))
    if args.stage == 'factory':
        payload = payload.replace(
            "roots = ['/home/unitree/unitree', '/opt/ota_package', '/home/unitree/g1plus_pc4_unitree_install', '/home/unitree/deploy', '/home/unitree/g1_true23_onboard']",
            "roots = ['/unitree', '/home/robot_emb']")
    # Let mirrored WSL finish its initial interface update before adding our address.
    time.sleep(2)
    ensure_local_address()
    result = subprocess.run(
        ['sshpass', '-e', 'ssh', '-b', ADDRESS, '-o', 'ConnectTimeout=5',
         '-o', 'StrictHostKeyChecking=yes', '-o', 'NumberOfPasswordPrompts=1',
         f'unitree@{ROBOT}', 'python3', '-'],
        input=payload, text=True, capture_output=True, timeout=40,
        env=dict(os.environ, SSHPASS=os.environ['UNITREE_SSH_PASSWORD']))
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    data = json.loads(result.stdout)
    data['inspection'] = {'host': ROBOT, 'local_address': ADDRESS,
                          'read_only': True, 'unix_time': time.time()}
    destination = ROOT / f'latest_{args.stage}.json'
    destination.write_text(json.dumps(data, indent=2))
    print(f'Connected to unitree@{ROBOT} from {ADDRESS}. Saved {destination}')


if __name__ == '__main__':
    main()
