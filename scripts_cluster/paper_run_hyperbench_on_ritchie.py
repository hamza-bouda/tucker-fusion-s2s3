"""
paper_run_hyperbench_on_ritchie.py
---------------------------------------------------------------------------
Soumission de l'étude de faisabilité HyperBench du NL-JTAE sur Ritchie
(GPU A100, OAR), puis rapatriement du CSV et du comparatif.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import paramiko
from ritchie_secret import RITCHIE_PASSWORD
import os
import time

REMOTE_DIR = "tucker_fusion"
JOB_NAME = "NLJTAE_HB"
FILES = ["nljtae_hyperbench.py", "run_nljtae_hyperbench.py"]


def main():
    print("Connexion SSH à ritchie...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('ritchie.univ-littoral.fr', 22, 'hbouda', RITCHIE_PASSWORD)
    sftp = ssh.open_sftp()

    for f in FILES:
        sftp.put(f, f"{REMOTE_DIR}/{f}")
        print(f"Téléversé : {f}")

    submit = f"""#!/bin/bash
#OAR -n {JOB_NAME}
#OAR -l /nodes=1/core=8,walltime=02:00:00
#OAR -p gpumodel='A100'
#OAR -O {JOB_NAME}_%jobid%.out
#OAR -E {JOB_NAME}_%jobid%.err

echo "Étude de faisabilité HyperBench — NL-JTAE"
hostname
nvidia-smi

source /etc/profile
module load conda/23.7
source activate tucker || conda activate tucker

python -u run_nljtae_hyperbench.py fast 3000
"""
    remote_submit = f"{REMOTE_DIR}/submit_nljtae_hb.sh"
    with sftp.file(remote_submit, "wb") as f:
        f.write(submit.replace("\r\n", "\n").encode('utf-8'))
    sftp.chmod(remote_submit, 0o755)
    sftp.close()

    _, stdout, stderr = ssh.exec_command(
        f"cd {REMOTE_DIR} && oarsub -S ./submit_nljtae_hb.sh")
    out = stdout.read().decode('utf-8', errors='ignore')
    print(out)
    err = stderr.read().decode('utf-8', errors='ignore')
    if err.strip():
        print("STDERR:", err)

    job_id = None
    for line in out.splitlines():
        if "OAR_JOB_ID=" in line:
            job_id = line.split("OAR_JOB_ID=")[1].strip()
    if not job_id:
        print("Erreur : job non soumis.")
        return

    print(f"Job {job_id} soumis. Surveillance...")
    while True:
        _, stdout, _ = ssh.exec_command(f"oarstat -j {job_id}")
        status = stdout.read().decode('utf-8', errors='ignore')
        job_line = next((l for l in status.splitlines() if l.strip().startswith(job_id)), None)
        if not job_line:
            print("\nJob terminé.")
            break
        state = job_line.split()[1] if len(job_line.split()) >= 2 else "?"
        if state in ('T', 'F', 'E'):
            print(f"\nJob terminé ({state}).")
            break
        print("." if state == 'R' else state.lower(), end="", flush=True)
        time.sleep(10)

    print("\n=== LOG ===")
    _, stdout, _ = ssh.exec_command(f"cat ~/{REMOTE_DIR}/{JOB_NAME}_{job_id}.out")
    print(stdout.read().decode('utf-8', errors='replace') or "(vide)")
    _, stdout, _ = ssh.exec_command(f"cat ~/{REMOTE_DIR}/{JOB_NAME}_{job_id}.err")
    errlog = stdout.read().decode('utf-8', errors='replace')
    if errlog.strip():
        print("=== ERREURS ===")
        print(errlog[-4000:])

    os.makedirs("results", exist_ok=True)
    sftp = ssh.open_sftp()
    for r in ["results_nljtae_fast.csv", "results_tucker_fast.csv", "results_bicubic_fast.csv"]:
        try:
            sftp.get(f"{REMOTE_DIR}/{r}", f"results/{r}")
            print(f"ok : {r}")
        except Exception as e:
            print(f"échec : {r} ({e})")
    try:
        sftp.get(f"{REMOTE_DIR}/{JOB_NAME}_{job_id}.out", f"results/{JOB_NAME}_{job_id}.out")
    except Exception:
        pass
    sftp.close()
    ssh.close()
    print("Terminé.")


if __name__ == "__main__":
    main()
