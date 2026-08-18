"""
run_linear_experiments.py
─────────────────────────────────────────────────────────────────────────────
Exécute la suite complète d'expériences sur le modèle linéaire G-SCOTT-Tucker :
  1. Grille de réglage (tuning) sur lr (3e-3, 1e-2, 3e-2) et scheduler (constant vs cosine).
  2. Sélection automatique de la meilleure configuration.
  3. Recherche de rangs (20,20,10), (30,30,10), (40,40,15) avec la meilleure configuration.
  4. Balayage de beta (1e-6 à 1e-2) pour courbe parcimonie-fidélité (adam_prox et adam_l1).
  5. Optionnel : variante lbfgs_prox.
─────────────────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import sys
import csv
import subprocess

def run_cmd(cmd):
    print(f"Exec: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error: {res.stderr}")
    return res

def get_avg_psnr(csv_path):
    if not os.path.exists(csv_path):
        return -1.0
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            # Trouver l'indice de 'psnr' ou 'PSNR'
            psnr_idx = -1
            for idx, col in enumerate(header):
                if col.lower() == 'psnr':
                    psnr_idx = idx
                    break
            if psnr_idx == -1:
                return -1.0
            
            psnrs = []
            for row in reader:
                if len(row) > psnr_idx:
                    try:
                        psnrs.append(float(row[psnr_idx]))
                    except ValueError:
                        pass
            return sum(psnrs) / len(psnrs) if psnrs else -1.0
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return -1.0

def main():
    print("=== DEBUT DE LA SUITE D'EXPERIENCES LINEAIRES G-SCOTT ===")
    
    # 1. Grille de réglage (Tuning)
    tuning_cases = [
        # (lr, scheduler, tag)
        (3e-3, "constant", "tuning_lr_3e-3_const"),
        (1e-2, "constant", "tuning_lr_1e-2_const"),
        (3e-2, "constant", "tuning_lr_3e-2_const"),
        (3e-3, "cosine", "tuning_lr_3e-3_cos"),
        (1e-2, "cosine", "tuning_lr_1e-2_cos"),
        (3e-2, "cosine", "tuning_lr_3e-2_cos"),
    ]
    
    tuning_results = {}
    for lr, sched, tag in tuning_cases:
        csv_name = f"results_linear_{tag}_fast.csv"
        # On ne relance pas si le CSV existe déjà
        if not os.path.exists(csv_name):
            run_cmd(f"python run_linear_hyperbench.py --method adam_prox --iters 10000 --lr {lr} --scheduler {sched} --tag {tag}")
        
        avg_psnr = get_avg_psnr(csv_name)
        tuning_results[tag] = (lr, sched, avg_psnr)
        print(f"  > {tag} : PSNR moyen = {avg_psnr:.4f} dB")
        
    # Sélection du meilleur
    best_tag = max(tuning_results, key=lambda k: tuning_results[k][2])
    best_lr, best_sched, best_psnr = tuning_results[best_tag]
    print(f"\nMEILLEURE CONFIGURATION DE TUNING : {best_tag} (lr={best_lr}, scheduler={best_sched}, PSNR={best_psnr:.4f} dB)")
    
    # 2. Recherche de Rangs (Ranks)
    rank_cases = [
        ("20,20,10", "ranks_20_20_10"),
        ("30,30,10", "ranks_30_30_10"),
        ("40,40,15", "ranks_40_40_15"),
    ]
    
    for ranks_str, tag in rank_cases:
        csv_name = f"results_linear_{tag}_fast.csv"
        if not os.path.exists(csv_name):
            run_cmd(f"python run_linear_hyperbench.py --method adam_prox --iters 10000 --lr {best_lr} --scheduler {best_sched} --ranks {ranks_str} --tag {tag}")
        avg_psnr = get_avg_psnr(csv_name)
        print(f"  > Rangs {ranks_str} ({tag}) : PSNR moyen = {avg_psnr:.4f} dB")
        
    # 3. Courbe Parcimonie-Fidélité (Beta sweeps)
    betas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for b in betas:
        # Adam Prox
        tag_prox = f"beta_sweep_prox_{b}"
        csv_prox = f"results_linear_{tag_prox}_fast.csv"
        if not os.path.exists(csv_prox):
            run_cmd(f"python run_linear_hyperbench.py --method adam_prox --iters 10000 --lr {best_lr} --scheduler {best_sched} --beta {b} --tag {tag_prox}")
        print(f"  > Prox beta={b} ({tag_prox}) : PSNR moyen = {get_avg_psnr(csv_prox):.4f} dB")
        
        # Adam L1
        tag_l1 = f"beta_sweep_l1_{b}"
        csv_l1 = f"results_linear_{tag_l1}_fast.csv"
        if not os.path.exists(csv_l1):
            run_cmd(f"python run_linear_hyperbench.py --method adam_l1 --iters 10000 --lr {best_lr} --scheduler {best_sched} --beta {b} --tag {tag_l1}")
        print(f"  > L1 beta={b} ({tag_l1}) : PSNR moyen = {get_avg_psnr(csv_l1):.4f} dB")

    # 4. Optionnel : LBFGS
    tag_lbfgs = "lbfgs_prox"
    csv_lbfgs = f"results_linear_{tag_lbfgs}_fast.csv"
    if not os.path.exists(csv_lbfgs):
        run_cmd(f"python run_linear_hyperbench.py --method lbfgs_prox --iters 1000 --lr {best_lr} --scheduler constant --tag {tag_lbfgs}")
    print(f"  > LBFGS ({tag_lbfgs}) : PSNR moyen = {get_avg_psnr(csv_lbfgs):.4f} dB")
    
    # 5. Comparaison finale des optimiseurs (config optimale, rangs optimaux)
    # Utiliser les rangs qui ont donné le meilleur PSNR (chercher dans les CSVs)
    best_ranks = '20,20,10'  # valeur par défaut ; sera mis à jour
    best_ranks_psnr = -1.0
    for ranks_str, tag in rank_cases:
        p = get_avg_psnr(f"results_linear_{tag}_fast.csv")
        if p > best_ranks_psnr:
            best_ranks_psnr = p
            best_ranks = ranks_str
    print(f"\nMEILLEURS RANGS : {best_ranks} (PSNR={best_ranks_psnr:.4f} dB)")

    optimizer_cases = [
        ("adam_prox",  10000, best_lr, best_sched,  "adam_prox_final"),
        ("adam_l1",    10000, best_lr, best_sched,  "adam_l1_final"),
        ("sgd_prox",   10000, best_lr, best_sched,  "sgd_prox_final"),
        ("lbfgs_prox", 1000,  best_lr, "constant",  "lbfgs_prox_final"),
    ]
    print("\n=== COMPARAISON FINALE DES OPTIMISEURS ===")
    for method, iters, lr, sched, tag in optimizer_cases:
        csv_name = f"results_linear_{tag}_fast.csv"
        if not os.path.exists(csv_name):
            run_cmd(f"python run_linear_hyperbench.py --method {method} --iters {iters} --lr {lr} --scheduler {sched} --ranks {best_ranks} --tag {tag}")
        avg_psnr = get_avg_psnr(csv_name)
        print(f"  > {method} ({tag}) : PSNR moyen = {avg_psnr:.4f} dB")

    print("\n=== FIN DE LA SUITE D'EXPERIENCES ===")

if __name__ == "__main__":
    main()
