"""
Soumet v3_unsupervised_paviau.py sur le cluster Ritchie (GPU A100).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import paramiko
from ritchie_secret import RITCHIE_PASSWORD, os, time

HOST = "ritchie.univ-littoral.fr"
USER = "hbouda"
PASS = RITCHIE_PASSWORD
REMOTE_DIR = "tucker_fusion"

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS)
    sftp = ssh.open_sftp()

    # Upload des fichiers nécessaires
    scripts = [
        "v3_unsupervised_paviau.py",
        "v3_non_lineaire_conjoint_nljtae.py",
        "v0_lineaire_math_utils.py",
    ]
    for s in scripts:
        local  = os.path.join(r"c:\Users\hamza\Desktop\tucker_fusion", s)
        remote = f"{REMOTE_DIR}/{s}"
        print(f"Upload {s}...")
        sftp.put(local, remote)

    # Script OAR
    job_sh = """#!/bin/bash
#OAR -n V3_UNSUPERVISED_PAVIAU
#OAR -l /nodes=1/core=8,walltime=01:00:00
#OAR -p gpumodel='A100'
#OAR -O V3_UNSUP_PAVIAU_%jobid%.out
#OAR -E V3_UNSUP_PAVIAU_%jobid%.err

echo "== NL-JTAE v3 NON-SUPERVISE PaviaU =="
hostname
nvidia-smi

source /etc/profile
module load conda/23.7
source activate tucker || conda activate tucker

python -u v3_unsupervised_paviau.py
"""
    remote_sh = f"{REMOTE_DIR}/submit_unsup_paviau.sh"
    with sftp.file(remote_sh, "wb") as f:
        f.write(job_sh.replace("\r\n", "\n").encode("utf-8"))
    sftp.chmod(remote_sh, 0o755)
    sftp.close()

    # Soumission
    stdin, stdout, stderr = ssh.exec_command(
        f"cd {REMOTE_DIR} && oarsub -S ./submit_unsup_paviau.sh"
    )
    out = stdout.read().decode("utf-8", errors="ignore")
    err = stderr.read().decode("utf-8", errors="ignore")
    print(out)
    if err: print("ERR:", err)

    job_id = None
    for line in out.splitlines():
        if "OAR_JOB_ID=" in line:
            job_id = line.split("OAR_JOB_ID=")[1].strip()
            break

    if not job_id:
        print("Job ID non trouvé."); ssh.close(); return

    print(f"Job {job_id} soumis. Surveillance...")

    # Attente
    while True:
        _, so, _ = ssh.exec_command(
            f"oarstat -j {job_id} -f 2>/dev/null | grep state"
        )
        state = so.read().decode("utf-8", errors="ignore").strip()
        print(state[-1] if state else ".", end="", flush=True)
        if "Terminated" in state or "Error" in state:
            print()
            break
        time.sleep(10)

    # Lecture des logs
    _, so, _ = ssh.exec_command(
        f"cat {REMOTE_DIR}/V3_UNSUP_PAVIAU_{job_id}.out 2>/dev/null"
    )
    print("\n=== LOGS ===")
    print(so.read().decode("utf-8", errors="ignore"))

    ssh.close()

if __name__ == "__main__":
    main()
