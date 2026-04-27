"""
Entraîne un modèle sur les signaux BRUTS UCI HAR avec les mêmes features
que celles calculées depuis le téléphone — pipeline identique garanti.
"""
import numpy as np
import joblib
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

DATASET = 'UCI HAR Dataset'

if not os.path.exists(DATASET):
    import urllib.request, zipfile
    print("Téléchargement du dataset UCI HAR...")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'
    urllib.request.urlretrieve(url, 'UCI_HAR.zip')
    with zipfile.ZipFile('UCI_HAR.zip', 'r') as z:
        z.extractall('.')
    os.remove('UCI_HAR.zip')
    print("Téléchargement terminé.")
else:
    print("Dataset UCI HAR déjà présent.")

# ── Chargement des signaux bruts (128 pas de temps par fenêtre) ───────────────
def load_signals(split):
    folder = f'{DATASET}/{split}/Inertial Signals'
    signals = []
    files = [
        f'total_acc_x_{split}.txt', f'total_acc_y_{split}.txt', f'total_acc_z_{split}.txt',
        f'body_gyro_x_{split}.txt', f'body_gyro_y_{split}.txt', f'body_gyro_z_{split}.txt',
    ]
    for fname in files:
        path = os.path.join(folder, fname)
        signals.append(np.loadtxt(path))   # shape: (N, 128)
    return np.stack(signals, axis=2)       # shape: (N, 128, 6)

print("Chargement des signaux bruts...")
X_train_raw = load_signals('train')   # (7352, 128, 6)
X_test_raw  = load_signals('test')    # (2947, 128, 6)

y_train = np.loadtxt(f'{DATASET}/train/y_train.txt', dtype=int) - 1
y_test  = np.loadtxt(f'{DATASET}/test/y_test.txt',   dtype=int) - 1
print(f"Train: {X_train_raw.shape} | Test: {X_test_raw.shape}")

# ── Extraction des features (identique au téléphone) ─────────────────────────
def extract_features(window):
    """
    window : (128, 6) — [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    Retourne : vecteur de features simples et robustes
    """
    from scipy.ndimage import uniform_filter1d

    acc  = window[:, :3].astype(np.float64)
    gyro = window[:, 3:].astype(np.float64)

    # Séparation gravité / corps (filtre passe-bas simple)
    gravity  = uniform_filter1d(acc, size=20, axis=0)
    body_acc = acc - gravity

    # Jerk (dérivée)
    body_jerk     = np.diff(body_acc, axis=0, prepend=body_acc[:1]) * 50
    gyro_jerk     = np.diff(gyro,     axis=0, prepend=gyro[:1])     * 50

    # Magnitudes
    acc_mag   = np.sqrt(np.sum(body_acc**2,  axis=1))
    gyro_mag  = np.sqrt(np.sum(gyro**2,      axis=1))
    grav_mag  = np.sqrt(np.sum(gravity**2,   axis=1))
    jerk_mag  = np.sqrt(np.sum(body_jerk**2, axis=1))

    def stats(sig):
        """mean, std, max, min, rms, iqr, energy pour chaque axe ou signal 1D"""
        if sig.ndim == 1:
            return [
                np.mean(sig), np.std(sig), np.max(sig), np.min(sig),
                np.sqrt(np.mean(sig**2)),
                float(np.percentile(sig,75) - np.percentile(sig,25)),
                np.sum(sig**2) / len(sig),
            ]
        feats = []
        for i in range(sig.shape[1]):
            feats.extend(stats(sig[:, i]))
        return feats

    f = []
    f.extend(stats(body_acc))    # 7×3 = 21
    f.extend(stats(gyro))        # 7×3 = 21
    f.extend(stats(body_jerk))   # 7×3 = 21
    f.extend(stats(gyro_jerk))   # 7×3 = 21
    f.extend(stats(acc_mag))     # 7
    f.extend(stats(gyro_mag))    # 7
    f.extend(stats(grav_mag))    # 7
    f.extend(stats(jerk_mag))    # 7

    # Corrélations entre axes
    for a, b in [(0,1),(0,2),(1,2)]:
        c = np.corrcoef(body_acc[:,a], body_acc[:,b])[0,1]
        f.append(0.0 if np.isnan(c) else c)
        c = np.corrcoef(gyro[:,a], gyro[:,b])[0,1]
        f.append(0.0 if np.isnan(c) else c)

    # SMA (Signal Magnitude Area)
    f.append(np.sum(np.abs(body_acc)) / 128)
    f.append(np.sum(np.abs(gyro))     / 128)

    return np.array(f, dtype=np.float32)

print("Extraction des features (train)...")
X_train = np.array([extract_features(X_train_raw[i]) for i in range(len(X_train_raw))])
print("Extraction des features (test)...")
X_test  = np.array([extract_features(X_test_raw[i])  for i in range(len(X_test_raw))])
print(f"Nombre de features : {X_train.shape[1]}")

# ── Normalisation ──────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Entraînement ───────────────────────────────────────────────────────────────
print("\nEntraînement du modèle mobile...")
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu', solver='adam',
    max_iter=500, random_state=42,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=20, verbose=True
)
model.fit(X_train_sc, y_train)

# ── Évaluation ─────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
acc    = accuracy_score(y_test, y_pred)
print(f"\nAccuracy test : {acc*100:.2f}%")
print(classification_report(y_test, y_pred,
      target_names=['MARCHE','MONTER','DESCENDRE','ASSIS','DEBOUT','ALLONGE']))

# ── Sauvegarde ─────────────────────────────────────────────────────────────────
joblib.dump(model,  'har_mobile_model.pkl')
joblib.dump(scaler, 'har_mobile_scaler.pkl')
print("\n[OK] har_mobile_model.pkl  +  har_mobile_scaler.pkl sauvegardes")
print("Lance maintenant : lancer_api.bat")
