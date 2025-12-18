# KC-DATA-SAMPLING

## Installation rapide

```bash
uv venv .venv
source .venv/bin/activate
uv sync 
```

## Lancement

### 1. Génération d'images augmentées et adversariales

```bash
# Images corrompues (augmentations)
python augment_run.py --config examples/augment_all.yml

# Images adversariales (FGSM, PGD, C&W, APGD)
python adversarial_run.py --config examples/adversarial_fgsm.yml
python adversarial_run.py --config examples/adversarial_all.yml
```

**Note**: Ces scripts génèrent maintenant les images **sans inférence**. Les colonnes générées sont :
- `sample_uuid` (nouveau UUID unique)
- `parent_sample_uuid` (lien vers l'échantillon d'origine)
- `image_hash` (hash SHA256)
- `sample_id`, `sample_type`, `class_id`, `class_name`, `split`
- `transform_id`, `transform_name`, `transform_level`, `transform_params`
- `image_bytes`

### 2. Inférence sur les images

```bash
# Inférence de base (garde toutes les colonnes + ajoute prédictions)
python kc_inference.py --config examples/kc_inference.yaml

# mode in-place (onmodifie le fichier directement)
python kc_inference.py --config examples/kc_inference.yaml --in-place
```

#### Options d'inférence disponibles :

**Modes de sortie**  :
- `label_only` : Seulement le label prédit
- `label_prob` : Label + probabilité
- `top_k` : Label + prob + top-K labels et probs
- `all_probs` : Label + prob + toutes les probabilités
- `logits` : Label + prob + logits
- `extended` : Label + prob + all_probs + logits (recommandé)
- `full` : Tout (label + prob + top_k + all_probs + logits)
