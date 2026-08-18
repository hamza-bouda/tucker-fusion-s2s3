"""
paper_run_benchmark_on_ritchie.py
---------------------------------------------------------------------------
Téléversement et soumission du benchmark complet de l'article
(v0 ALS, v1 CTAE, v4 MS-NL-JTAE auto-supervisé + oracle) sur la
plateforme de calcul Ritchie (GPU A100, ordonnanceur OAR).
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
JOB_NAME = "PAPER_BENCH"
EPOCHS = 6000

FILES = [
    "paper_common.py",
    "paper_v0_tucker_als.py",
    "paper_v1_ctae.py",
    "paper_v4_msnljtae.py",
    "paper_benchmark.py",
]

RESULTS_TO_FETCH = [
    "results/paper_benchmark.json",
    "results/paper_benchmark_table.md",
    "results/paper_v0_reconstruction.tif",
    "results/paper_v1_reconstruction.tif",
    "results/paper_v4_selfsup_reconstruction.tif",
    "results/paper_v4_oracle_reconstruction.tif",
]


def create_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('ritchie.univ-littoral.fr', 22, 'hbouda', RITCHIE_PASSWORD)
    return client


def main():
    print("Connexion SSH à ritchie.univ-littoral.fr...")
    ssh = create_ssh_client()
    sftp = ssh.open_sftp()

    try:
        sftp.mkdir(REMOTE_DIR)
    except IOError:
        pass

    for f in FILES:
        print(f"Téléversement de {f}...")
        sftp.put(f, f"{REMOTE_DIR}/{f}")

    # Dataset PaviaU (déjà présent lors des runs v2/v3 normalement)
    try:
        sftp.mkdir(f"{REMOTE_DIR}/data")
    except IOError:
        pass
    try:
        sftp.stat(f"{REMOTE_DIR}/data/PaviaU.mat")
        print("PaviaU.mat déjà présent sur le serveur.")
    except IOError:
        print("Téléversement de data/PaviaU.mat (34.8 MB)...")
        sftp.put("data/PaviaU.mat", f"{REMOTE_DIR}/data/PaviaU.mat")

    submit_script = f"""#!/bin/bash
#OAR -n {JOB_NAME}
#OAR -l /nodes=1/core=8,walltime=02:00:00
#OAR -p gpumodel='A100'
#OAR -O {JOB_NAME}_%jobid%.out
#OAR -E {JOB_NAME}_%jobid%.err

echo "Démarrage du benchmark article (v0 / v1 / v4 selfsup / v4 oracle)"
hostname
nvidia-smi

source /etc/profile
module load conda/23.7
source activate tucker || conda activate tucker

python -u paper_benchmark.py --epochs {EPOCHS}
"""
    remote_submit = f"{REMOTE_DIR}/submit_paper_benchmark.sh"
    with sftp.file(remote_submit, "wb") as f:
        f.write(submit_script.replace("\r\n", "\n").encode('utf-8'))
    sftp.chmod(remote_submit, 0o755)
    sftp.close()

    print("Soumission du job OAR...")
    _, stdout, stderr = ssh.exec_command(
        f"cd {REMOTE_DIR} && oarsub -S ./submit_paper_benchmark.sh")
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    print(out)
    if err.strip():
        print("STDERR:", err)

    job_id = None
    for line in out.splitlines():
        if "OAR_JOB_ID=" in line:
            job_id = line.split("OAR_JOB_ID=")[1].strip()
            break
    if not job_id:
        print("Erreur : ID de job OAR introuvable.")
        ssh.close()
        return

    print(f"Job {job_id} soumis. Surveillance (10 s)...")
    while True:
        _, stdout, _ = ssh.exec_command(f"oarstat -j {job_id}")
        status = stdout.read().decode('utf-8', errors='ignore')
        job_line = next((l.strip() for l in status.splitlines()
                         if l.strip().startswith(job_id)), None)
        if not job_line:
            print(f"\nJob {job_id} terminé.")
            break
        state = job_line.split()[1] if len(job_line.split()) >= 2 else "?"
        if state in ('T', 'F', 'E'):
            print(f"\nJob {job_id} terminé avec le statut {state}.")
            break
        print("." if state == 'R' else state.lower(), end="", flush=True)
        time.sleep(10)

    print("\n=== LOG DU JOB ===")
    _, stdout, _ = ssh.exec_command(f"cat ~/{REMOTE_DIR}/{JOB_NAME}_{job_id}.out")
    print(stdout.read().decode('utf-8', errors='ignore') or "(vide)")
    _, stdout, _ = ssh.exec_command(f"cat ~/{REMOTE_DIR}/{JOB_NAME}_{job_id}.err")
    err_log = stdout.read().decode('utf-8', errors='ignore')
    if err_log.strip():
        print("=== ERREURS ===")
        print(err_log)

    print("\nRapatriement des résultats...")
    os.makedirs("results", exist_ok=True)
    sftp = ssh.open_sftp()
    for r in RESULTS_TO_FETCH:
        try:
            sftp.get(f"{REMOTE_DIR}/{r}", r)
            print(f"  ok : {r}")
        except Exception as e:
            print(f"  échec : {r} ({e})")
    try:
        sftp.get(f"{REMOTE_DIR}/{JOB_NAME}_{job_id}.out",
                 f"results/{JOB_NAME}_{job_id}.out")
    except Exception:
        pass
    sftp.close()
    ssh.close()
    print("Terminé.")


if __name__ == "__main__":
    main()
