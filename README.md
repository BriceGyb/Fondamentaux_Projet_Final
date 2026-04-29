# 8INF867 — Mini-Projet 2 : Apprentissage Automatique Avancé

Projet académique couvrant quatre grandes familles d'algorithmes d'apprentissage automatique : réseaux de neurones multicouches (MLP), application mobile embarquée, réseaux convolutifs (CNN) et apprentissage par renforcement profond (DQN).

---

# 🎬 PRÉSENTATION VIDÉO DU PROJET

## ⬇️ CLIQUEZ ICI POUR REGARDER LA PRÉSENTATION ⬇️

[![REGARDER LA PRÉSENTATION VIDÉO](https://img.youtube.com/vi/LfVGydPyV4M/maxresdefault.jpg)](https://youtu.be/LfVGydPyV4M)

### 👉 [https://youtu.be/LfVGydPyV4M](https://youtu.be/LfVGydPyV4M) 👈

---

## Structure du Projet

```
projet-fondamentaux/
│
├── 2_1_MLP_Reconnaissance_Activites.ipynb       # Partie 2.1 — MLP / HAR
├── 2_3_CNN_Fashion_MNIST.ipynb                  # Partie 2.3 — CNN / Fashion-MNIST
├── 2_4_DQN_Apprentissage_Renforcement.ipynb     # Partie 2.4 — DQN / CartPole
│
└── App_Mobile/                                  # Partie 2.2 — App mobile HAR
    ├── api.py                                   # Serveur Flask (API REST + UI)
    ├── train_mobile_model.py                    # Entraînement du modèle mobile
    ├── templates/index.html                     # Interface web mobile
    ├── har_mobile_model.pkl                     # Modèle entraîné (sklearn MLP)
    ├── har_mobile_scaler.pkl                    # Normaliseur StandardScaler
    ├── lancer_api.bat                           # Script de lancement Windows
    ├── ouvrir_port.bat                          # Ouverture pare-feu Windows
    └── UCI HAR Dataset/                         # Dataset UCI HAR (signaux bruts)
```

---

## Partie 2.1 — Reconnaissance d'Activités Humaines (MLP)

**Notebook :** `2_1_MLP_Reconnaissance_Activites.ipynb`

Entraînement d'un réseau de neurones multicouches (MLP) pour classifier 6 activités humaines à partir de données de capteurs IMU (accéléromètre + gyroscope) portés à la ceinture.

**Dataset :** UCI HAR Dataset — 10 299 fenêtres de 2,56 secondes à 50 Hz, 561 features pré-calculées.

**Activités classifiées :**
| Classe | Activité |
|--------|----------|
| 0 | Marche |
| 1 | Monter les escaliers |
| 2 | Descendre les escaliers |
| 3 | Assis |
| 4 | Debout |
| 5 | Allongé |

**Architecture MLP :** couches denses avec régularisation, optimiseur Adam, early stopping.

---

## Partie 2.2 — Application Mobile (HAR en Temps Réel)

**Dossier :** `App_Mobile/`

Application web accessible depuis un smartphone qui utilise les capteurs embarqués (accéléromètre + gyroscope) pour prédire l'activité physique en temps réel via un modèle MLP déployé sur une API Flask.

Voir le [README détaillé de l'App Mobile](App_Mobile/README.md) pour l'architecture complète.

**Lancement rapide :**
```bash
cd App_Mobile
lancer_api.bat              # Démarre le serveur Flask (port 5001)
npx localtunnel --port 5001 # Crée un tunnel HTTPS public
```

---

## Partie 2.3 — Classification d'Images (CNN)

**Notebook :** `2_3_CNN_Fashion_MNIST.ipynb`

Exploration des réseaux de neurones convolutifs (CNN) pour la classification d'images du dataset Fashion-MNIST (70 000 images 28×28 en niveaux de gris, 10 classes de vêtements).

**Expériences menées :**
- Comparaison MLP vs CNN sur les mêmes données
- Impact de la profondeur du réseau (1, 2, 3 blocs convolutifs)
- Régularisation : Dropout, Batch Normalization, L2
- Data Augmentation (rotations, flips, zoom)
- Analyse des feature maps et filtres appris
- Visualisation des erreurs de classification

**Résultat :** accuracy test ~93% avec le meilleur CNN vs ~89% avec MLP seul.

---

## Partie 2.4 — Apprentissage par Renforcement (DQN)

**Notebook :** `2_4_DQN_Apprentissage_Renforcement.ipynb`

Implémentation complète d'un agent Deep Q-Network (DQN) pour résoudre le problème du pendule inversé (CartPole-v1), environnement classique de contrôle automatique.

**Algorithme :** DQN (Mnih et al., DeepMind 2015) avec :
- Experience Replay Buffer (50 000 transitions)
- Target Network (mis à jour toutes les 10 épisodes)
- Politique ε-greedy (exploration → exploitation)
- Reward Shaping (−10 à la chute)

**Résultats :**
| Métrique | Valeur |
|----------|--------|
| Épisodes pour résoudre | 489 |
| Score moyen final | 276.5 / 500 |
| Score test (ε=0, 20 épisodes) | **500 / 500** |
| Amélioration vs aléatoire | **×24** |

---

## Technologies Utilisées

| Outil | Rôle |
|-------|------|
| Python 3.10+ | Langage principal |
| TensorFlow / Keras | Réseaux de neurones (MLP, CNN) |
| scikit-learn | Modèle mobile léger (MLPClassifier) |
| Gymnasium | Environnement RL (CartPole-v1) |
| Flask | API REST + serveur web mobile |
| NumPy / SciPy | Traitement du signal, features |
| Matplotlib / Seaborn | Visualisations |
| JavaScript (DeviceMotion API) | Capture capteurs smartphone |

---

## Prérequis

```bash
pip install tensorflow scikit-learn gymnasium flask numpy scipy matplotlib
```

---

## Cours

**8INF867 — Fondamentaux de l'Apprentissage Automatique**
Université du Québec à Chicoutimi (UQAC)
