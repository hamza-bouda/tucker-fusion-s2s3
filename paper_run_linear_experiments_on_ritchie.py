"""
paper_run_linear_experiments_on_ritchie.py
─────────────────────────────────────────────────────────────────────────────
Téléverse et soumet le script d'expériences run_linear_experiments.py
sur Ritchie (A100 GPU), surveille l'exécution puis rapatrie tous les
fichiers de résultats CSV et JSON générés.
─────────────────────────────────────────────────────────────────────────────
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import paramiko
from ritchie_secret import RITCHIE_PASSWORD
import os
import time

REMOTE_DIR = "tucker_fusion"
FILES_TO_UPLOAD = [
    "linear_tucker_torch.py",
    "run_linear_hyperbench.py",
    "run_linear_experiments.py"
]

def main():
    print("Connexion SSH à ritchie.univ-littoral.fr...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('ritchie.univ-littoral.fr', 22, 'hbouda', RITCHIE_PASSWORD)
    sftp = ssh.open_sftp()

    # Création du dossier si nécessaire
    try:
        sftp.mkdir(REMOTE_DIR)
        print(f"Dossier distant {REMOTE_DIR} créé.")
    except IOError:
        pass

    for f in FILES_TO_UPLOAD:
        sftp.put(f, f"{REMOTE_DIR}/{f}")
        print(f"Téléversé : {f}")

    # Script de soumission OAR
    name = "LIN_EXP_GSCOTT"
    submit_content = f"""#!/bin/bash
#OAR -n {name}
#OAR -l /nodes=1/core=8,walltime=04:00:00
#OAR -p gpumodel='A100'
#OAR -O {name}_%jobid%.out
#OAR -E {name}_%jobid%.err

echo "Suite complète d'expériences linéaires G-SCOTT-Tucker"
hostname
nvidia-smi
source /etc/profile
module load conda/23.7
source activate tucker || conda activate tucker
python -u run_linear_experiments.py
"""
    remote_submit = f"{REMOTE_DIR}/submit_lin_exp.sh"
    with sftp.file(remote_submit, "wb") as f:
        f.write(submit_content.replace("\r\n", "\n").encode('utf-8'))
    sftp.chmod(remote_submit, 0o755)

    print("Soumission du job d'expériences...")
    # Supprimer les CSVs partiels (ranks_30_30_10 interrompu) avant de soumettre
    _, stdout, stderr = ssh.exec_command(
        f"cd {REMOTE_DIR} && rm -f results_linear_ranks_30_30_10_fast.csv "
        f"results_linear_ranks_40_40_15_fast.csv "
        f"results_linear_beta_sweep_* "
        f"results_linear_adam_prox_final_fast.csv "
        f"results_linear_adam_l1_final_fast.csv "
        f"results_linear_sgd_prox_final_fast.csv "
        f"results_linear_lbfgs_prox_final_fast.csv "
        f"&& oarsub -S ./submit_lin_exp.sh"
    )
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    
    job_id = None
    for line in out.splitlines():
        if "OAR_JOB_ID=" in line:
            job_id = line.split("OAR_JOB_ID=")[1].strip()
            break
            
    if not job_id:
        print(f"ÉCHEC de soumission du job : {out} {err}")
        sftp.close()
        ssh.close()
        return

    print(f"Job {job_id} soumis avec succès ! Surveillance active...")

    # Surveillance du job
    while True:
        time.sleep(15)
        _, stdout, _ = ssh.exec_command(f"oarstat -j {job_id}")
        status = stdout.read().decode('utf-8', errors='ignore')
        job_line = next((l for l in status.splitlines() if l.strip().startswith(job_id)), None)
        state = job_line.split()[1] if job_line and len(job_line.split()) >= 2 else None
        
        if state is None or state in ('T', 'F', 'E'):
            print(f"\nJob terminé avec le statut : {state}")
            break
        print(".", end="", flush=True)

    # Affichage du journal d'exécution
    print("\n=== JOURNAL D'EXECUTION (OUT) ===")
    _, stdout, _ = ssh.exec_command(f"cat {REMOTE_DIR}/{name}_{job_id}.out")
    print(stdout.read().decode('utf-8', errors='replace'))
    
    _, stdout, _ = ssh.exec_command(f"cat {REMOTE_DIR}/{name}_{job_id}.err")
    errlog = stdout.read().decode('utf-8', errors='replace')
    if errlog.strip():
        print("=== ERREURS (ERR) ===")
        print(errlog)

    # Rapatriement de tous les fichiers de résultats du dossier distant
    print("\nRapatriement des fichiers de résultats...")
    os.makedirs("results", exist_ok=True)
    
    # Lister les fichiers dans REMOTE_DIR pour récupérer tous les csv et json de linear
    files = sftp.listdir(REMOTE_DIR)
    for f in files:
        if f.startswith("results_linear_") and (f.endswith(".csv") or f.endswith(".json")):
            try:
                sftp.get(f"{REMOTE_DIR}/{f}", f"results/{f}")
                print(f"Rapatrié : {f}")
            except Exception as e:
                print(f"Erreur rapatriement {f} : {e}")

    sftp.close()
    ssh.close()
    print("Terminé.")

if __name__ == "__main__":
    main()
