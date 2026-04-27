# App Mobile — Reconnaissance d'Activités en Temps Réel

Application web mobile qui capture les données de l'accéléromètre et du gyroscope d'un smartphone via le navigateur, envoie ces données à une API Flask locale, et affiche en temps réel l'activité physique détectée par un modèle de Machine Learning.

---

## Architecture Globale

```
┌─────────────────────────────────────┐        ┌──────────────────────────────────┐
│           SMARTPHONE                │        │              PC                  │
│                                     │        │                                  │
│  ┌───────────────────────────────┐  │        │  ┌────────────────────────────┐  │
│  │   Navigateur (Chrome/Safari)  │  │  HTTP  │  │      Flask API (api.py)    │  │
│  │                               │◀─┼────────┼─▶│                            │  │
│  │  • DeviceMotion API           │  │  POST  │  │  • Extraction de features  │  │
│  │  • Capture 2.56s de données   │  │ /predict│  │  • Normalisation           │  │
│  │  • Affichage résultat         │  │        │  │  • Prédiction MLP          │  │
│  └───────────────────────────────┘  │        │  └────────────┬───────────────┘  │
│                                     │        │               │                  │
│         Tunnel HTTPS                │        │  ┌────────────▼───────────────┐  │
│      (localtunnel / ngrok)          │        │  │   Modèle entraîné          │  │
└─────────────────────────────────────┘        │  │  har_mobile_model.pkl      │  │
                                               │  │  har_mobile_scaler.pkl     │  │
                                               │  └────────────────────────────┘  │
                                               └──────────────────────────────────┘
```

---

## Fichiers

```
App_Mobile/
│
├── api.py                    # Serveur Flask — API REST + interface web
├── train_mobile_model.py     # Script d'entraînement du modèle mobile
│
├── templates/
│   └── index.html            # Interface web mobile (HTML + CSS + JS)
│
├── har_mobile_model.pkl      # Modèle MLP entraîné (sklearn)
├── har_mobile_scaler.pkl     # StandardScaler ajusté sur les données d'entraînement
│
├── lancer_api.bat            # Démarre l'API Flask (Windows)
├── ouvrir_port.bat           # Ouvre le port 5001 dans le pare-feu Windows (admin)
│
└── UCI HAR Dataset/          # Dataset UCI HAR (signaux bruts accéléro + gyro)
    ├── train/Inertial Signals/
    └── test/Inertial Signals/
```

---

## Pipeline de Données

Le point clé de cette application est que le **pipeline d'extraction de features est identique** entre l'entraînement (sur UCI HAR) et l'inférence (sur données du téléphone). Cela garantit la cohérence des prédictions.

### 1. Capture (téléphone → navigateur)

La [DeviceMotion API](https://developer.mozilla.org/en-US/docs/Web/API/DeviceMotionEvent) du navigateur capture les données des capteurs à ~50 Hz pendant **2,56 secondes** (128 échantillons) :

```
[ accX, accY, accZ, gyroX, gyroY, gyroZ ]  ×128
     (m/s²)                (rad/s)
```

> **Conversion importante :** L'accéléromètre du téléphone fournit des valeurs en m/s². Le modèle a été entraîné sur des données en **g** (unité de gravité). La conversion `/9.81` est appliquée côté serveur avant l'extraction des features.

### 2. Extraction de Features (serveur → 120 features)

À partir de la fenêtre de 128 échantillons × 6 canaux, les features suivantes sont calculées :

| Signal | Description | Features |
|--------|-------------|----------|
| `body_acc` | Accélération corporelle (après séparation gravité par filtre passe-bas) | 7 stats × 3 axes = 21 |
| `gyro` | Gyroscope brut | 7 stats × 3 axes = 21 |
| `body_jerk` | Dérivée de body_acc × 50 Hz | 7 stats × 3 axes = 21 |
| `gyro_jerk` | Dérivée du gyroscope × 50 Hz | 7 stats × 3 axes = 21 |
| `acc_mag` | Magnitude de body_acc | 7 stats |
| `gyro_mag` | Magnitude du gyroscope | 7 stats |
| `grav_mag` | Magnitude de la gravité | 7 stats |
| `jerk_mag` | Magnitude du jerk | 7 stats |
| Corrélations | Entre axes X/Y, X/Z, Y/Z pour acc et gyro | 6 |
| SMA | Signal Magnitude Area (acc + gyro) | 2 |
| **Total** | | **120 features** |

Les 7 statistiques par signal sont : `mean, std, max, min, rms, iqr, energy`.

La séparation gravité/corps est réalisée avec un **filtre à moyenne glissante** (fenêtre de 20 points) :
```python
gravity  = uniform_filter1d(acc, size=20, axis=0)
body_acc = acc - gravity
```

### 3. Normalisation et Prédiction

```python
feats_sc = scaler.transform(feats)        # StandardScaler (µ=0, σ=1)
probs    = model.predict_proba(feats_sc)  # MLP sklearn → probabilités par classe
```

---

## Modèle — MLPClassifier (scikit-learn)

Le modèle mobile est un **MLP scikit-learn** entraîné sur les signaux **bruts** du dataset UCI HAR (dossier `Inertial Signals`), avec le même pipeline de features que ci-dessus. Ce choix évite toute dépendance à TensorFlow au moment de la prédiction.

| Paramètre | Valeur |
|-----------|--------|
| Architecture | (256, 128, 64) |
| Activation | ReLU |
| Optimiseur | Adam |
| Early stopping | Oui (patience 20) |
| Accuracy test UCI HAR | **88%** |

**Entraînement :**
```bash
python train_mobile_model.py
# Télécharge automatiquement UCI HAR si absent
# Génère : har_mobile_model.pkl + har_mobile_scaler.pkl
```

---

## Interface Web Mobile (index.html)

Interface single-page optimisée pour mobile :

- **Bouton de capture** — démarre une session de 2,56 secondes (requis par iOS pour demander la permission capteurs depuis un geste utilisateur)
- **Affichage résultat** — activité prédite avec emoji + barre de confiance
- **Probabilités par classe** — barres de progression pour les 6 activités
- **Capteurs en temps réel** — valeurs live d'accéléromètre et gyroscope
- **Compatibilité** — Android Chrome, iOS Safari (avec popup permission)

---

## Lancement

### Prérequis
```bash
pip install flask numpy scipy scikit-learn joblib
```

### Étape 1 — Entraîner le modèle (première fois uniquement)
```bash
python train_mobile_model.py
```

### Étape 2 — Démarrer l'API
```bash
# Windows
lancer_api.bat

# Ou directement
python api.py
```

L'API démarre sur `http://0.0.0.0:5001`.

### Étape 3 — Créer un tunnel HTTPS public

Le navigateur mobile requiert **HTTPS** pour accéder aux capteurs de mouvement.

```bash
# Avec localtunnel
npx localtunnel --port 5001

# → URL du type : https://xxxx.loca.lt
```

### Étape 4 — Ouvrir sur le téléphone

Ouvrir l'URL fournie par le tunnel dans **Chrome (Android)** ou **Safari (iOS)**, appuyer sur "Démarrer la capture" et rester immobile ou effectuer une activité.

---

## API Endpoints

### `GET /`
Sert l'interface web mobile (`templates/index.html`).

### `POST /predict`
Reçoit les données capteurs et retourne la prédiction.

**Corps de la requête :**
```json
{
  "sensor_data": [
    [accX, accY, accZ, gyroX, gyroY, gyroZ],
    ...
  ]
}
```
*(minimum 5 mesures, idéalement 128)*

**Réponse :**
```json
{
  "activity": "ALLONGÉ",
  "activity_id": 5,
  "emoji": "🛏️",
  "confidence": 99.8,
  "all_probs": [
    {"label": "MARCHE",              "emoji": "🚶", "prob": 0.0},
    {"label": "MONTER ESCALIERS",    "emoji": "⬆️", "prob": 0.0},
    {"label": "DESCENDRE ESCALIERS", "emoji": "⬇️", "prob": 0.0},
    {"label": "ASSIS",               "emoji": "🪑", "prob": 0.1},
    {"label": "DEBOUT",              "emoji": "🧍", "prob": 0.1},
    {"label": "ALLONGÉ",             "emoji": "🛏️", "prob": 99.8}
  ]
}
```

---

## Notes Techniques

- **Conversion d'unités** : le téléphone fournit l'accélération en **m/s²**, le modèle attend des **g** → division par 9.81 appliquée dans `extract_features()`.
- **Rééchantillonnage** : si le téléphone capture plus ou moins de 128 mesures, une interpolation linéaire ramène la fenêtre exactement à 128 points.
- **Processus zombies Windows** : si l'API plante et qu'un ancien processus Python occupe le port, lancer le serveur sur un port différent (`port=5001` au lieu de `5000`) en modifiant la dernière ligne de `api.py`.
