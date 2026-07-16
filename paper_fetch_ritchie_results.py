"""Rapatriement des résultats du benchmark article depuis Ritchie (job OAR)."""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import paramiko
from ritchie_secret import RITCHIE_PASSWORD

REMOTE_DIR = "tucker_fusion"
JOB_ID = sys.argv[1] if len(sys.argv) > 1 else "5544"

FILES = [
    "results/paper_benchmark.json",
    "results/paper_benchmark_table.md",
    "results/paper_v0_reconstruction.tif",
    "results/paper_v1_reconstruction.tif",
    "results/paper_v4_selfsup_reconstruction.tif",
    "results/paper_v4_oracle_reconstruction.tif",
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('ritchie.univ-littoral.fr', 22, 'hbouda', RITCHIE_PASSWORD)
sftp = client.open_sftp()

os.makedirs("results", exist_ok=True)
for r in FILES:
    try:
        sftp.get(f"{REMOTE_DIR}/{r}", r)
        print(f"ok    : {r}")
    except Exception as e:
        print(f"échec : {r} ({e})")

for ext in ("out", "err"):
    try:
        sftp.get(f"{REMOTE_DIR}/PAPER_BENCH_{JOB_ID}.{ext}",
                 f"results/PAPER_BENCH_{JOB_ID}.{ext}")
        print(f"ok    : PAPER_BENCH_{JOB_ID}.{ext}")
    except Exception as e:
        print(f"échec : log .{ext} ({e})")

sftp.close()
client.close()
print("Rapatriement terminé.")
