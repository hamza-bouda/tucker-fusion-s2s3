"""
paper_run_beta_sweep_on_ritchie.py
---------------------------------------------------------------------------
Balayage du poids de parcimonie beta pour le modèle linéaire Tucker :
    beta dans {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}  x  {adam_prox, adam_l1}
sur le protocole HyperBench Pavia (rangs 30,30,10, lr 1e-2 cosine).
Produit la courbe parcimonie-fidélité : sparsité exacte de G et PSNR vs beta.
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
JOB_NAME = "LIN_BETA_SWEEP"
FILES = ["linear_tucker_torch.py", "run_linear_hyperbench.py"]
BETAS = sys.argv[1:] or ["1e-6", "1e-5", "1e-4", "1e-3", "1e-2"]


def main():
    print("Connexion SSH à ritchie...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('ritchie.univ-littoral.fr', 22, 'hbouda', RITCHIE_PASSWORD)
    sftp = ssh.open_sftp()
    for f in FILES:
        sftp.put(f, f"{REMOTE_DIR}/{f}")
        print(f"Téléversé : {f}")

    runs = "\n".join(
        f'python -u run_linear_hyperbench.py --method {m} --iters 10000 '
        f'--lr 0.01 --scheduler cosine --ranks 30,30,10 --beta {b} '
        f'--tag beta_{m}_{b}'
        for m in ("adam_prox", "adam_l1") for b in BETAS)

    submit = f"""#!/bin/bash
#OAR -n {JOB_NAME}
#OAR -l /nodes=1/core=8,walltime=03:00:00
#OAR -p gpumodel='A100'
#OAR -O {JOB_NAME}_%jobid%.out
#OAR -E {JOB_NAME}_%jobid%.err

echo "Balayage beta — courbe parcimonie-fidélité (Pavia)"
hostname
source /etc/profile
module load conda/23.7
source activate tucker || conda activate tucker
{runs}
"""
    remote_submit = f"{REMOTE_DIR}/submit_beta_sweep.sh"
    with sftp.file(remote_submit, "wb") as f:
        f.write(submit.replace("\r\n", "\n").encode('utf-8'))
    sftp.chmod(remote_submit, 0o755)
    sftp.close()

    _, stdout, stderr = ssh.exec_command(
        f"cd {REMOTE_DIR} && oarsub -S ./submit_beta_sweep.sh")
    out = stdout.read().decode('utf-8', errors='ignore')
    print(out)
    job_id = None
    for line in out.splitlines():
        if "OAR_JOB_ID=" in line:
            job_id = line.split("OAR_JOB_ID=")[1].strip()
    if not job_id:
        print("Erreur : job non soumis.", stderr.read().decode(errors='ignore'))
        return
    print(f"Job {job_id} soumis. Surveillance...")

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
        print(".", end="", flush=True)
        time.sleep(30)

    os.makedirs("results", exist_ok=True)
    sftp = ssh.open_sftp()
    _, stdout, _ = ssh.exec_command(
        f"ls ~/{REMOTE_DIR}/results_linear_beta_* 2>/dev/null")
    for path in stdout.read().decode().split():
        name = os.path.basename(path)
        try:
            sftp.get(f"{REMOTE_DIR}/{name}", f"results/{name}")
            print(f"ok : {name}")
        except Exception as e:
            print(f"échec : {name} ({e})")
    _, stdout, _ = ssh.exec_command(f"tail -30 ~/{REMOTE_DIR}/{JOB_NAME}_{job_id}.out")
    print(stdout.read().decode('utf-8', errors='replace'))
    sftp.close()
    ssh.close()
    print("Terminé.")


if __name__ == "__main__":
    main()
