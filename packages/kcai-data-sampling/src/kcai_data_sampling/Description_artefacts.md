# Description artefacts

## Samples

Raw_samples
    - samples_raw.parquet : table de référence unique avec colonne split (train/val/test). Clés minimales : sample_id, split, class_name, transform_name, transform_id, image_path|image_bytes.
    - samples_train.parquet, samples_val.parquet, samples_test.parquet : mêmes colonnes que samples_raw mais filtrées par split, pour des traitements cibles (calcule de métriques separement)

Perturbed_samples
    - Corrupted sample (augmentations/corruptions classiques)
        - Colonnes: sample_id , class_name, transform_name, transform_level, transform_param, split, model_name,  orig_pred, perturbed_pred, image_bytes.
    - Adversarial sample (attaques adverses)
        - Colonnes : sample_id, class_name, transform_name (attack_name), transform_level, transform_param (epsilon|steps), split, model_name , class_name, image_bytes.

## Features (based on image_byte) : garder les colonnes et enlever les différentes images.

- By sample
    - raw_sample : une ligne par sample_id, features dérivées de image_bytes . 
        - Corrupted sample + colonnes de features (luminosité, contrast, blur, entropy), 
            - retirer image_bytes.
    - perturbed_sample : Corrupted sample + features (luminosité, contrast, blur, entropy) ; 
        - retirer image_bytes.

## Métriques
### dqm-ml 
#### Representativness [chi-2 goodness, grte, ks, shannon]
    - By feature 
    - By split: train/test/val
        - By class name
        - By models
        - By transform_name

### autres métriques

#### Classification (à envoyer directement sur klarity ? )
    - Accuracy globale et par classe
    - Precision / Recall / F1 ...
    - ROC-AUC et PR-AUC one-vs-rest (macro / weighted)
    - Matrices de confusion (par split et par transform_name)

#### Robustesse (corruptions/augmentations)
    - Robust accuracy par transform_name et transform_level
    - Moyenne sur niveaux de sévérité 
    - Worst-case accuracy sur l'ensemble des corruptions: trouver le couple (transform_name, transform_level) qui minimise l’accuracy.
    - Flip rate (taux de changement de prédiction vs original)
    - Consistency orig_pred vs perturbed_pred (accord modèle)
    - Degradation relative vs baseline propre: (acc_clean - acc_corrupted) / acc_clean

#### Robustesse (adversarial)
    - Robust accuracy par attaque (transform_name) et intensité (epsilon/steps)
    - Courbes accuracy vs epsilon et Aire Sous la Courbe (AUC)
    - Worst-case accuracy sur l'ensemble des attaques
    - Degradation relative vs baseline propre



### Agrégations / découpages
    - By split: train / val / test
    - By model_name
    - By class_name
    - By transform_name et transform_level (sévérité)
    - Moyennes pondérées par fréquence de classe si pertinent
