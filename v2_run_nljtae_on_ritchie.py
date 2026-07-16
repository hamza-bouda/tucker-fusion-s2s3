"""
run_nl_jtae_on_ritchie.py
---------------------------------------------------------------------------
Téléversement et soumission du script d'entraînement du modèle 100% Non-linéaire (NL-JTAE)
sur la nouvelle plateforme de calcul Ritchie avec GPU.
"""

import paramiko
from ritchie_secret import RITCHIE_PASSWORD
import os
import time

def create_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('ritchie.univ-littoral.fr', 22, 'hbouda', RITCHIE_PASSWORD)
    return client

def main():
    print("Connexion SSH à ritchie.univ-littoral.fr...")
    ssh = create_ssh_client()
    sftp = ssh.open_sftp()
    
    # Créer le répertoire de destination sur le serveur
    remote_dir = "tucker_fusion"
    try:
        sftp.mkdir(remote_dir)
        print(f"Répertoire distant '{remote_dir}' créé.")
    except IOError:
        pass # Déjà existant
        
    # Fichiers à uploader
    local_file = "v2_non_lineaire_conjoint_nljtae.py"
    remote_file = f"{remote_dir}/v2_non_lineaire_conjoint_nljtae.py"
    
    print(f"Téléversement de {local_file} vers {remote_file}...")
    sftp.put(local_file, remote_file)
    print("Téléversement terminé.")
    
    # Upload du dataset PaviaU.mat s'il n'est pas déjà présent
    local_data = "data/PaviaU.mat"
    remote_data = f"{remote_dir}/data/PaviaU.mat"
    try:
        sftp.mkdir(f"{remote_dir}/data")
    except IOError:
        pass
        
    try:
        sftp.stat(remote_data)
        print("Le jeu de données PaviaU.mat est déjà présent sur le serveur.")
    except IOError:
        print(f"Téléversement de {local_data} vers {remote_data} (34.8 MB)...")
        sftp.put(local_data, remote_data)
        print("Téléversement du dataset terminé.")
    
    # Script OAR de soumission distant pour Ritchie
    # Note: On supprime la contrainte -p orval pour laisser OAR choisir parmi les GPU chimay31-35 (A100/H100/GH200)
    submit_script_content = """#!/bin/bash
#OAR -n NL_JTAE_RITCHIE
#OAR -l /nodes=1/core=8,walltime=02:00:00
#OAR -p gpumodel='A100'
#OAR -O NL_JTAE_%jobid%.out
#OAR -E NL_JTAE_%jobid%.err

echo "Demarrage du job OAR NL-JTAE sur GPU (Ritchie)"
hostname
nvidia-smi

source /etc/profile
module load conda/23.7
source activate tucker || conda activate tucker

python -u v2_non_lineaire_conjoint_nljtae.py --dataset PaviaU --epochs 50000 --ranks 16 16 32 --lr 2e-3
"""
    
    remote_submit_script = f"{remote_dir}/submit_nl_jtae_ritchie.sh"
    print(f"Création du script de soumission OAR distant '{remote_submit_script}'...")
    with sftp.file(remote_submit_script, "wb") as f:
        f.write(submit_script_content.replace("\r\n", "\n").encode('utf-8'))
        
    sftp.chmod(remote_submit_script, 0o755)
    sftp.close()
    
    # Soumission du job OAR
    print("Soumission du job à OAR sur Ritchie...")
    stdin, stdout, stderr = ssh.exec_command(f"cd {remote_dir} && oarsub -S ./submit_nl_jtae_ritchie.sh")
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    
    print("=== SORTIE OARSUB ===")
    print(out)
    if err:
        print("=== ERREUR OARSUB ===")
        print(err)
        
    # Extraire l'ID du job
    job_id = None
    for line in out.splitlines():
        if "OAR_JOB_ID=" in line:
            job_id = line.split("OAR_JOB_ID=")[1].strip()
            break
            
    if job_id:
        print(f"\nJob {job_id} soumis avec succès ! Démarrage de la surveillance...")
        
        # Surveillance du job
        while True:
            stdin, stdout, stderr = ssh.exec_command(f"oarstat -j {job_id}")
            status_out = stdout.read().decode('utf-8', errors='ignore')
            lines = status_out.strip().splitlines()
            job_line = None
            for line in lines:
                if line.strip().startswith(job_id):
                    job_line = line.strip()
                    break
            
            if not job_line:
                print(f"\nLe job {job_id} n'est plus actif (Terminé) !")
                break
                
            parts = job_line.split()
            if len(parts) >= 2:
                state = parts[1]
                if state == 'R':
                    print(".", end="", flush=True)
                elif state == 'W':
                    print("w", end="", flush=True)
                elif state in ['T', 'F', 'E']:
                    print(f"\nLe job {job_id} s'est terminé avec le statut : {state} !")
                    break
                else:
                    print(f"[{state}]", end="", flush=True)
            time.sleep(10)
            
        # Récupération des logs de sortie
        print("\n=== LECTURE DES LOGS ===")
        stdin, stdout, stderr = ssh.exec_command(f"cat ~/tucker_fusion/NL_JTAE_{job_id}.out")
        log_out = stdout.read().decode('utf-8', errors='ignore')
        print(log_out if log_out.strip() else "Aucun log standard trouvé.")
        
        stdin, stdout, stderr = ssh.exec_command(f"cat ~/tucker_fusion/NL_JTAE_{job_id}.err")
        log_err = stderr.read().decode('utf-8', errors='ignore')
        if log_err.strip():
            print("=== ERREURS DU JOB ===")
            print(log_err)
            
        # Rapatriement des figures et de l'image de résultat
        sftp = ssh.open_sftp()
        try:
            print("\nRapatriement de la figure de comparaison...")
            sftp.get("tucker_fusion/v2_nljtae_comparaison_paviau.png", "v2_nljtae_comparaison_paviau.png")
            print("Figure rapatriée avec succès : v2_nljtae_comparaison_paviau.png")
            
            # Rapatrier aussi dans les artifacts
            shutil_dest = "C:\\Users\\hamza\\.gemini\\antigravity\\brain\\15201f33-80da-4d2c-a3f4-2b3479314927\\v2_nljtae_comparaison_paviau.png"
            import shutil
            shutil.copy("v2_nljtae_comparaison_paviau.png", shutil_dest)
            
            print("Rapatriement de l'image HSI de résultat...")
            os.makedirs("results", exist_ok=True)
            sftp.get("tucker_fusion/results/PaviaU_reconstructed_NL_JTAE.tif", "results/PaviaU_reconstructed_NL_JTAE.tif")
            print("Image HSI de résultat rapatriée avec succès : results/PaviaU_reconstructed_NL_JTAE.tif")
        except Exception as e:
            print(f"Erreur lors du rapatriement des fichiers : {e}")
        sftp.close()
        
    else:
        print("Erreur : Impossible d'extraire l'ID du job OAR.")
        
    ssh.close()

if __name__ == "__main__":
    main()
