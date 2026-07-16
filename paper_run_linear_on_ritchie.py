"""
paper_run_linear_on_ritchie.py
---------------------------------------------------------------------------
Soumet EN PARALLÈLE les 3 variantes d'optimisation du modèle linéaire
(adam_prox, adam_l1, sgd_prox) sur Ritchie (3 jobs OAR A100 simultanés),
surveille les trois, puis rapatrie CSV + historiques de convergence.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import paramiko
from ritchie_secret import RITCHIE_PASSWORD
import os
import time

REMOTE_DIR = "tucker_fusion"
METHODS = ["adam_prox", "adam_l1", "sgd_prox"]
FILES = ["linear_tucker_torch.py", "run_linear_hyperbench.py"]


def main():
    print("Connexion SSH à ritchie...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('ritchie.univ-littoral.fr', 22, 'hbouda', RITCHIE_PASSWORD)
    sftp = ssh.open_sftp()

    for f in FILES:
        sftp.put(f, f"{REMOTE_DIR}/{f}")
        print(f"Téléversé : {f}")

    # Un script de soumission par méthode
    jobs = {}
    for m in METHODS:
        name = f"LIN_{m.upper()}"
        submit = f"""#!/bin/bash
#OAR -n {name}
#OAR -l /nodes=1/core=4,walltime=01:00:00
#OAR -p gpumodel='A100'
#OAR -O {name}_%jobid%.out
#OAR -E {name}_%jobid%.err

echo "Modèle linéaire Tucker — variante {m}"
hostname
source /etc/profile
module load conda/23.7
source activate tucker || conda activate tucker
python -u run_linear_hyperbench.py {m} 3000
"""
        remote_submit = f"{REMOTE_DIR}/submit_lin_{m}.sh"
        with sftp.file(remote_submit, "wb") as f:
            f.write(submit.replace("\r\n", "\n").encode('utf-8'))
        sftp.chmod(remote_submit, 0o755)

        _, stdout, stderr = ssh.exec_command(
            f"cd {REMOTE_DIR} && oarsub -S ./submit_lin_{m}.sh")
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        job_id = None
        for line in out.splitlines():
            if "OAR_JOB_ID=" in line:
                job_id = line.split("OAR_JOB_ID=")[1].strip()
        if not job_id:
            print(f"ÉCHEC soumission {m} : {out} {err}")
            continue
        jobs[m] = job_id
        print(f"Job {m} soumis : {job_id}")
    sftp.close()

    # Surveillance des trois jobs en parallèle
    print("Surveillance des jobs :", jobs)
    pending = dict(jobs)
    while pending:
        time.sleep(15)
        done = []
        for m, jid in pending.items():
            _, stdout, _ = ssh.exec_command(f"oarstat -j {jid}")
            status = stdout.read().decode('utf-8', errors='ignore')
            job_line = next((l for l in status.splitlines()
                             if l.strip().startswith(jid)), None)
            state = job_line.split()[1] if job_line and len(job_line.split()) >= 2 else None
            if state is None or state in ('T', 'F', 'E'):
                done.append(m)
                print(f"\n[{m}] terminé ({state}).")
        for m in done:
            del pending[m]
        if pending:
            print(".", end="", flush=True)

    # Logs + rapatriement
    os.makedirs("results", exist_ok=True)
    sftp = ssh.open_sftp()
    for m, jid in jobs.items():
        name = f"LIN_{m.upper()}"
        _, stdout, _ = ssh.exec_command(f"tail -20 ~/{REMOTE_DIR}/{name}_{jid}.out")
        print(f"\n=== LOG {m} (fin) ===")
        print(stdout.read().decode('utf-8', errors='replace') or "(vide)")
        _, stdout, _ = ssh.exec_command(f"tail -15 ~/{REMOTE_DIR}/{name}_{jid}.err")
        errlog = stdout.read().decode('utf-8', errors='replace')
        if 'Error' in errlog or 'Traceback' in errlog:
            print(f"=== ERREURS {m} ===")
            print(errlog)
        for r in [f"results_linear_{m}_fast.csv", f"results_linear_{m}_hist.json"]:
            try:
                sftp.get(f"{REMOTE_DIR}/{r}", f"results/{r}")
                print(f"ok : {r}")
            except Exception as e:
                print(f"échec : {r} ({e})")
    sftp.close()
    ssh.close()
    print("Terminé.")


if __name__ == "__main__":
    main()
