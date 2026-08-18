"""
v3_run_andorra_fusion_on_ritchie.py
---------------------------------------------------------------------------
Téléversement et soumission du script de fusion d'Andorre (v3_andorra_fusion.py)
sur la plateforme de calcul Ritchie avec GPU.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import paramiko
from ritchie_secret import RITCHIE_PASSWORD
import os
import time
import shutil

def create_ssh_client():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect('ritchie.univ-littoral.fr', 22, 'hbouda', RITCHIE_PASSWORD)
    return client

def main():
    print("Connexion SSH à ritchie.univ-littoral.fr...")
    ssh = create_ssh_client()
    sftp = ssh.open_sftp()
    
    remote_dir = "tucker_fusion"
    try:
        sftp.mkdir(remote_dir)
        print(f"Répertoire distant '{remote_dir}' créé.")
    except IOError:
        pass
        
    try:
        sftp.mkdir(f"{remote_dir}/data")
    except IOError:
        pass

    try:
        sftp.mkdir(f"{remote_dir}/data/andorra")
    except IOError:
        pass
        
    try:
        sftp.mkdir(f"{remote_dir}/data/andorra/20220709")
    except IOError:
        pass
        
    try:
        sftp.mkdir(f"{remote_dir}/results")
    except IOError:
        pass

    # 1. Téléverser les scripts python nécessaires
    files_to_upload = [
        ("v3_andorra_fusion.py", f"{remote_dir}/v3_andorra_fusion.py"),
        ("v3_non_lineaire_conjoint_nljtae.py", f"{remote_dir}/v3_non_lineaire_conjoint_nljtae.py"),
        ("v0_lineaire_math_utils.py", f"{remote_dir}/v0_lineaire_math_utils.py")
    ]
    
    for local_file, remote_file in files_to_upload:
        print(f"Téléversement de {local_file} vers {remote_file}...")
        sftp.put(local_file, remote_file)
    print("Téléversement des scripts terminé.")
    
    # 2. Téléverser les images satellites d'Andorre
    local_data_dir = "./data/andorra/20220709"
    remote_data_dir = f"{remote_dir}/data/andorra/20220709"
    
    andorra_files = [
        "31TCH_20220709_8800_8800_target_s2_10m.tif",
        "31TCH_20220709_8800_8800_target_s2_20m.tif",
        "31TCH_20220709_8800_8800_target_s3.tif"
    ]
    
    for img in andorra_files:
        local_path = os.path.join(local_data_dir, img)
        remote_path = f"{remote_data_dir}/{img}"
        try:
            sftp.stat(remote_path)
            print(f"L'image {img} est déjà présente sur le serveur.")
        except IOError:
            print(f"Téléversement de {img} vers le serveur...")
            sftp.put(local_path, remote_path)
            print(f"Téléversement de {img} terminé.")
            
    # 3. Création du script de soumission OAR
    submit_script_content = """#!/bin/bash
#OAR -n ANDORRA_FUSION_V3_RITCHIE
#OAR -l /nodes=1/core=8,walltime=04:00:00
#OAR -p gpumodel='A100'
#OAR -O ANDORRA_FUSION_V3_%jobid%.out
#OAR -E ANDORRA_FUSION_V3_%jobid%.err

echo "Demarrage de la fusion réelle Sentinel S2/S3 Andorre (v3) sur GPU"
hostname
nvidia-smi

source /etc/profile
module load conda/23.7
source activate tucker || conda activate tucker

python -u v3_andorra_fusion.py
"""
    
    remote_submit_script = f"{remote_dir}/submit_andorra_fusion_v3.sh"
    print(f"Création du script OAR '{remote_submit_script}'...")
    with sftp.file(remote_submit_script, "wb") as f:
        f.write(submit_script_content.replace("\r\n", "\n").encode('utf-8'))
        
    sftp.chmod(remote_submit_script, 0o755)
    sftp.close()
    
    # 4. Soumission à OAR
    print("Soumission du job à OAR sur Ritchie...")
    stdin, stdout, stderr = ssh.exec_command(f"cd {remote_dir} && oarsub -S ./submit_andorra_fusion_v3.sh")
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    
    print("=== SORTIE OARSUB ===")
    print(out)
    if err:
        print("=== ERREUR OARSUB ===")
        print(err)
        
    job_id = None
    for line in out.splitlines():
        if "OAR_JOB_ID=" in line:
            job_id = line.split("OAR_JOB_ID=")[1].strip()
            break
            
    if job_id:
        print(f"\nJob {job_id} soumis avec succès ! Surveillance active en cours...")
        
        # Surveillance
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
            time.sleep(15)
            
        # Récupération des logs
        print("\n=== LECTURE DES LOGS ===")
        stdin, stdout, stderr = ssh.exec_command(f"cat ~/tucker_fusion/ANDORRA_FUSION_V3_{job_id}.out")
        log_out = stdout.read().decode('utf-8', errors='ignore')
        print(log_out if log_out.strip() else "Aucun log standard trouvé.")
        
        stdin, stdout, stderr = ssh.exec_command(f"cat ~/tucker_fusion/ANDORRA_FUSION_V3_{job_id}.err")
        log_err = stderr.read().decode('utf-8', errors='ignore')
        if log_err.strip():
            print("=== ERREURS DU JOB ===")
            print(log_err)
            
        # 5. Récupération des fichiers de résultats via SFTP
        sftp = ssh.open_sftp()
        try:
            print("\nRapatriement de la figure de comparaison...")
            sftp.get("tucker_fusion/v3_andorra_fusion_comparaison.png", "v3_andorra_fusion_comparaison.png")
            print("Figure rapatriée avec succès : v3_andorra_fusion_comparaison.png")
            
            # Copie dans les artifacts
            shutil.copy("v3_andorra_fusion_comparaison.png", "C:\\Users\\hamza\\.gemini\\antigravity\\brain\\15201f33-80da-4d2c-a3f4-2b3479314927\\v3_andorra_fusion_comparaison.png")
            print("Figure copiée dans les artéfacts.")
            
            print("\nRapatriement de l'image super-résolue...")
            os.makedirs("results", exist_ok=True)
            sftp.get("tucker_fusion/results/andorra_super_resolved_v3.tif", "results/andorra_super_resolved_v3.tif")
            print("Image super-résolue rapatriée : results/andorra_super_resolved_v3.tif")
            
            print("\nRapatriement des poids du modèle...")
            sftp.get("tucker_fusion/results/andorra_NL_JTAE_v3_weights.pth", "results/andorra_NL_JTAE_v3_weights.pth")
            print("Poids du modèle rapatriés : results/andorra_NL_JTAE_v3_weights.pth")
        except Exception as e:
            print(f"Erreur lors de la récupération des fichiers : {e}")
        sftp.close()
    else:
        print("Erreur : Impossible d'obtenir l'ID du job OAR.")
        
    ssh.close()

if __name__ == "__main__":
    main()
