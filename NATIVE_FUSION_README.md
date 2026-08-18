# Fusion native Sentinel-2 / Sentinel-3 OLCI

Ce pipeline entraîne un autoencodeur multimodal non supervisé sans rééchantillonner
les images observées. Les entrées co-localisées restent sur leurs grilles natives :

- Sentinel-2 L1C : 4 bandes à 10 m, 6 bandes à 20 m et 2 bandes à 60 m ;
- Sentinel-3 OLCI L1B : 21 bandes à environ 300 m ;
- cible latente : cube de 21 bandes OLCI sur la grille Sentinel-2 à 10 m.

## Architecture

Chaque modalité possède un encodeur convolutionnel sur sa propre grille. Les jetons
portent leurs coordonnées géographiques réelles normalisées. Une attention croisée
produit un unique cœur Tucker parcimonieux partagé. Des dictionnaires continus
spatiaux et un dictionnaire spectral décodent ce cœur directement aux coordonnées de
chaque capteur.

La liste des modalités et `anchor_sensor` sont configurables : deux ou trois images
MSI distinctes peuvent donc être déclarées avec leurs propres nombres de bandes,
coordonnées, SRF et PSF. L'instance Andorre choisit la modalité S2 à 10 m comme ancre
spatiale du cube final.

L'opérateur d'observation applique ensuite :

1. la SRF du capteur pour projeter les 21 bandes latentes vers ses bandes observées ;
2. la PSF officielle S2A, bande par bande, sur la grille native correspondante ;
3. un petit résidu non linéaire borné, régularisé pour empêcher le contournement du
   cœur commun.

OLCI utilise une PSF identité dans cette première expérience : elle indique seulement
qu'aucun flou additionnel n'est supposé au-delà du produit natif. Une PSF OLCI
calibrée pourra être fournie au même emplacement sans modifier l'architecture.

## Préparation des données

```powershell
.\tucker_env\Scripts\python.exe scripts_article\prepare_native_andorra_training.py
```

Le fichier produit est `data/training/andorra_20180118_native_toa.npz`. Les DN S2
L1C et les radiances OLCI L1B sont convertis en réflectance TOA. Seuls des découpages
entiers sont effectués ; aucune image n'est agrandie ou réduite.

## Entraînement local

```powershell
.\tucker_env\Scripts\python.exe -m modeles_non_lineaires.train_native_sparse_tucker `
  --data data\training\andorra_20180118_native_toa.npz `
  --output outputs\native_sparse_tucker_andorra
```

La fonction objectif combine cohérence des observations natives, SAM OLCI,
parcimonie du cœur, orthogonalité du dictionnaire, pénalité des résidus et lissage
spectral faible. Le meilleur checkpoint est choisi sur des fenêtres géographiques de
validation exclues de l'entraînement.

## Évaluation

```powershell
.\tucker_env\Scripts\python.exe -m modeles_non_lineaires.evaluate_native_sparse_tucker `
  --checkpoint outputs\native_sparse_tucker_andorra\best_checkpoint.pt `
  --data data\training\andorra_20180118_native_toa.npz
```

Le rapport contient RMSE, PSNR, SAM, ERGAS, SSIM et UIQI pour la reconstruction de
chaque observation sur sa grille native, ainsi que des diagnostics sans référence du
cube fusionné et de la parcimonie. Ces mesures de cohérence ne remplacent pas une
vérité terrain haute résolution. Une référence synthétique peut être évaluée avec les
options `--reference` et `--fused`.
