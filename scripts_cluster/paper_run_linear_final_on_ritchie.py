"""
paper_run_linear_final_on_ritchie.py
---------------------------------------------------------------------------
Soumet le calcul FINAL des métriques complètes (PSNR/SAM/Q2n/ERGAS) des
modèles linéaires sur Pavia + Indian Pines, en dépendance OAR du job de
sweep en cours (démarre après lui pour profiter des meilleurs rangs).

Usage : python paper_run_linear_final_on_ritchie.py [job_id_anterieur]
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
FILES = ["linear_tucker_torch.py", "run_linear_hyperbench.py", "run_linear_final.py",
         "v0_lineaire_baseline_tucker_als.py"]
ANTERIOR = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != '-' else None
EXTRA = " ".join(sys.argv[2:])                 # ex : --only als
import re as _re
_suffix = _re.sub(r'[^A-Za-z0-9]+', '_', sys.argv[3])[:20].strip('_').upper() \
    if len(sys.argv) > 3 else ""
JOB_NAME = "LIN_FINAL_METRICS" + ("_" + _suffix if _suffix else "")


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
#OAR -l /nodes=1/core=8,walltime=03:00:00
#OAR -p gpumodel='A100'
#OAR -O {JOB_NAME}_%jobid%.out
#OAR -E {JOB_NAME}_%jobid%.err

echo "Métriques complètes modèles linéaires — Pavia + Indian Pines"
hostname
source /etc/profile
module load conda/23.7
source activate tucker || conda activate tucker
python -u run_linear_final.py {EXTRA}
"""
    remote_submit = f"{REMOTE_DIR}/submit_lin_final.sh"
    with sftp.file(remote_submit, "wb") as f:
        f.write(submit.replace("\r\n", "\n").encode('utf-8'))
    sftp.chmod(remote_submit, 0o755)
    sftp.close()

    dep = f" -a {ANTERIOR}" if ANTERIOR else ""
    _, stdout, stderr = ssh.exec_command(
        f"cd {REMOTE_DIR} && oarsub{dep} -S ./submit_lin_final.sh")
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
    print(f"Job {job_id} soumis"
          + (f" (démarre après le job {ANTERIOR})" if ANTERIOR else "")
          + ". Surveillance...")

    while True:
        _, stdout, _ = ssh.exec_command(f"oarstat -j {job_id}")
        status = stdout.read().decode('utf-8', errors='ignore')
        job_line = next((l for l in status.splitlines()
                         if l.strip().startswith(job_id)), None)
        if not job_line:
            print("\nJob terminé.")
            break
        state = job_line.split()[1] if len(job_line.split()) >= 2 else "?"
        if state in ('T', 'F', 'E'):
            print(f"\nJob terminé ({state}).")
            break
        print("." if state == 'R' else state.lower(), end="", flush=True)
        time.sleep(20)

    print("\n=== LOG (fin) ===")
    _, stdout, _ = ssh.exec_command(f"tail -60 ~/{REMOTE_DIR}/{JOB_NAME}_{job_id}.out")
    print(stdout.read().decode('utf-8', errors='replace') or "(vide)")
    _, stdout, _ = ssh.exec_command(f"tail -20 ~/{REMOTE_DIR}/{JOB_NAME}_{job_id}.err")
    errlog = stdout.read().decode('utf-8', errors='replace')
    if 'Traceback' in errlog or 'Error' in errlog:
        print("=== ERREURS ===")
        print(errlog)

    os.makedirs("results", exist_ok=True)
    sftp = ssh.open_sftp()
    _, stdout, _ = ssh.exec_command(
        f"ls ~/{REMOTE_DIR}/results_final_linear_* ~/{REMOTE_DIR}/results_linear_*_fast.csv 2>/dev/null")
    for path in stdout.read().decode().split():
        name = os.path.basename(path)
        try:
            sftp.get(f"{REMOTE_DIR}/{name}", f"results/{name}")
            print(f"ok : {name}")
        except Exception as e:
            print(f"échec : {name} ({e})")
    sftp.close()
    ssh.close()
    print("Terminé.")


if __name__ == "__main__":
    main()
