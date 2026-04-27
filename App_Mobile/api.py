"""
API Flask + Interface mobile — Reconnaissance d'Activités HAR
Utilise directement har_model.keras et har_scaler.pkl (déjà entraîné).
Lance : python api.py
Accès depuis le téléphone (même WiFi) : http://<IP_DU_PC>:5000
"""
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import socket
import os


# ── Chargement du modèle ──────────────────────────────────────────────────────
print("Chargement du modèle mobile...")
model  = joblib.load('har_mobile_model.pkl')
scaler = joblib.load('har_mobile_scaler.pkl')
print("[OK] har_mobile_model.pkl et har_mobile_scaler.pkl charges.")

ACTIVITIES = {
    0: ("MARCHE",              "🚶"),
    1: ("MONTER ESCALIERS",    "⬆️"),
    2: ("DESCENDRE ESCALIERS", "⬇️"),
    3: ("ASSIS",               "🪑"),
    4: ("DEBOUT",              "🧍"),
    5: ("ALLONGÉ",             "🛏️"),
}

# ── Extraction des features (pipeline identique à train_mobile_model.py) ──────
def extract_features(raw_data):
    """Pipeline identique à train_mobile_model.py — garanti cohérent."""
    from scipy.ndimage import uniform_filter1d

    data = np.array(raw_data, dtype=np.float64)
    if len(data) != 128:
        idx  = np.linspace(0, len(data) - 1, 128).astype(int)
        data = data[idx]

    acc  = data[:, :3] / 9.81   # m/s² → g (UCI HAR est en g)
    gyro = data[:, 3:]          # rad/s déjà correct

    gravity   = uniform_filter1d(acc, size=20, axis=0)
    body_acc  = acc - gravity
    body_jerk = np.diff(body_acc, axis=0, prepend=body_acc[:1]) * 50
    gyro_jerk = np.diff(gyro,     axis=0, prepend=gyro[:1])     * 50

    acc_mag  = np.sqrt(np.sum(body_acc**2,  axis=1))
    gyro_mag = np.sqrt(np.sum(gyro**2,      axis=1))
    grav_mag = np.sqrt(np.sum(gravity**2,   axis=1))
    jerk_mag = np.sqrt(np.sum(body_jerk**2, axis=1))

    def stats(sig):
        if sig.ndim == 1:
            return [np.mean(sig), np.std(sig), np.max(sig), np.min(sig),
                    np.sqrt(np.mean(sig**2)),
                    float(np.percentile(sig,75) - np.percentile(sig,25)),
                    np.sum(sig**2) / len(sig)]
        f = []
        for i in range(sig.shape[1]):
            f.extend(stats(sig[:, i]))
        return f

    f = []
    f.extend(stats(body_acc))
    f.extend(stats(gyro))
    f.extend(stats(body_jerk))
    f.extend(stats(gyro_jerk))
    f.extend(stats(acc_mag))
    f.extend(stats(gyro_mag))
    f.extend(stats(grav_mag))
    f.extend(stats(jerk_mag))

    for a, b in [(0,1),(0,2),(1,2)]:
        c = np.corrcoef(body_acc[:,a], body_acc[:,b])[0,1]
        f.append(0.0 if np.isnan(c) else c)
        c = np.corrcoef(gyro[:,a], gyro[:,b])[0,1]
        f.append(0.0 if np.isnan(c) else c)

    f.append(np.sum(np.abs(body_acc)) / 128)
    f.append(np.sum(np.abs(gyro))     / 128)

    return np.array(f, dtype=np.float32)


# ── Routes Flask ──────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    raw  = data.get('sensor_data', [])

    if len(raw) < 5:
        return jsonify({"error": "Donnees insuffisantes (" + str(len(raw)) + " mesures recues)"}), 400

    arr = np.array(raw)
    debug = {
        "n_samples": len(raw),
        "acc_x": {"min": round(float(arr[:,0].min()),3), "max": round(float(arr[:,0].max()),3), "mean": round(float(arr[:,0].mean()),3)},
        "acc_y": {"min": round(float(arr[:,1].min()),3), "max": round(float(arr[:,1].max()),3), "mean": round(float(arr[:,1].mean()),3)},
        "acc_z": {"min": round(float(arr[:,2].min()),3), "max": round(float(arr[:,2].max()),3), "mean": round(float(arr[:,2].mean()),3)},
        "gyr_x": {"min": round(float(arr[:,3].min()),4), "max": round(float(arr[:,3].max()),4), "mean": round(float(arr[:,3].mean()),4)},
        "gyr_y": {"min": round(float(arr[:,4].min()),4), "max": round(float(arr[:,4].max()),4), "mean": round(float(arr[:,4].mean()),4)},
        "gyr_z": {"min": round(float(arr[:,5].min()),4), "max": round(float(arr[:,5].max()),4), "mean": round(float(arr[:,5].mean()),4)},
        "first_sample": [round(float(v),4) for v in arr[0]],
    }
    print(f"\nDEBUG: {debug}")

    feats    = extract_features(raw).reshape(1, -1)
    feats_sc = scaler.transform(feats)
    probs    = model.predict_proba(feats_sc)[0]
    class_id = int(np.argmax(probs))
    label, emoji = ACTIVITIES[class_id]

    return jsonify({
        "activity_id": class_id,
        "activity":    label,
        "emoji":       emoji,
        "confidence":  round(float(probs[class_id]) * 100, 1),
        "debug":       debug,
        "all_probs": [
            {"label": ACTIVITIES[i][0], "emoji": ACTIVITIES[i][1],
             "prob":  round(float(p) * 100, 1)}
            for i, p in enumerate(probs)
        ]
    })

# ── Démarrage ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ip = socket.gethostbyname(socket.gethostname())
    print(f"\n{'='*50}")
    print(f"  Sur ce PC        : http://localhost:5001")
    print(f"  Sur le telephone : http://{ip}:5001")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=5001, debug=False)
