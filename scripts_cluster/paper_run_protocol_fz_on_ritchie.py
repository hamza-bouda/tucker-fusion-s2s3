"""
paper_run_protocol_fz_on_ritchie.py
---------------------------------------------------------------------------
Évalue nos modèles linéaires multi-flux dans le protocole EXACT de l'équipe
(1500², 4 flux S2/S3, bilinéaire, sans bruit) — job OAR A100.
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
JOB_NAME = "PROTO_FZ"
FILES = ["protocol_fz.py", "run_linear_hyperbench.py", "linear_tucker_torch.py",
         "v0_lineaire_baseline_tucker_als.py"]
ARGS = " ".join(sys.argv[1:]) or "60,60,15 1e-1"


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

echo "Protocole équipe (FZ) — modèles linéaires multi-flux"
hostname
source /etc/profile
module load conda/23.7
source activate tucker || conda activate tucker
python -u protocol_fz.py {ARGS}
"""
    remote_submit = f"{REMOTE_DIR}/submit_proto_fz.sh"
    with sftp.file(remote_submit, "wb") as f:
        f.write(submit.replace("\r\n", "\n").encode('utf-8'))
    sftp.chmod(remote_submit, 0o755)
    sftp.close()

    _, stdout, stderr = ssh.exec_command(
        f"cd {REMOTE_DIR} && oarsub -S ./submit_proto_fz.sh")
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

    print("\n=== LOG ===")
    _, stdout, _ = ssh.exec_command(
        f"grep -v '^|' ~/{REMOTE_DIR}/{JOB_NAME}_{job_id}.out | tail -40")
    print(stdout.read().decode('utf-8', errors='replace'))
    _, stdout, _ = ssh.exec_command(f"tail -15 ~/{REMOTE_DIR}/{JOB_NAME}_{job_id}.err")
    err = stdout.read().decode('utf-8', errors='replace')
    if 'Traceback' in err or 'Error' in err:
        print("=== ERREURS ===")
        print(err)

    os.makedirs("results", exist_ok=True)
    sftp = ssh.open_sftp()
    try:
        sftp.get(f"{REMOTE_DIR}/results_protocol_fz.json",
                 "results/results_protocol_fz.json")
        print("ok : results_protocol_fz.json")
    except Exception as e:
        print("échec :", e)
    sftp.close()
    ssh.close()
    print("Terminé.")


if __name__ == "__main__":
    main()
