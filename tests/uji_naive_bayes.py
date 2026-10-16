"""Pengujian bagian Cindy. Jalankan dari root proyek: python -m tests.uji_naive_bayes

1. Data uji asli: 2.000 baris, 68 gagal (tidak di-resampling), split 80:20.
2. Resampling hanya pada data training: ukuran hasil over/under sesuai harapan.
3. nb_over / nb_under: file json, joblib, png ada.
4. Metrik di json = hitung ulang dari model .joblib pada X_test, dan konsisten dengan
   perhitungan manual dari TN/FP/FN/TP (kelas positif = 1).
"""
import json
import sys

import joblib
from sklearn.naive_bayes import GaussianNB

from src import config
from src.resampling import oversample, undersample
from src.split_data import load_baseline
from src.evaluate import MODELS_DIR, RESULTS_DIR
from sklearn.metrics import confusion_matrix

gagal = []


def cek(nama, ok, info=""):
    print(f"[{'OK' if ok else 'GAGAL'}] {nama} {info}")
    if not ok:
        gagal.append(nama)


X_train, X_test, y_train, y_test = load_baseline()
cek("Split 80:20", len(X_train) == 8000 and len(X_test) == 2000, f"({len(X_train)}/{len(X_test)})")
cek("Data uji asli (68 gagal, tidak di-resampling)", int(y_test.sum()) == 68)

Xo, yo = oversample(X_train, y_train)
Xu, yu = undersample(X_train, y_train)
cek("Oversampling: rasio gagal:normal = 0,5", abs((yo == 1).sum() / (yo == 0).sum() - 0.5) < 0.01,
    f"({int((yo == 0).sum())} normal, {int((yo == 1).sum())} gagal)")
cek("Undersampling: kelas seimbang", (yu == 1).sum() == (yu == 0).sum(), f"({len(yu)} baris)")
cek("Data uji tidak berubah setelah resampling", len(X_test) == 2000 and int(y_test.sum()) == 68)

for nama, (Xr, yr) in {"nb_over": (Xo, yo), "nb_under": (Xu, yu)}.items():
    ada = all(p.exists() for p in [RESULTS_DIR / f"{nama}.json", MODELS_DIR / f"{nama}.joblib",
                                    config.FIGURES_DIR / f"cm_{nama}.png"])
    cek(f"{nama}: json, joblib, png tersimpan", ada)
    saved = json.loads((RESULTS_DIR / f"{nama}.json").read_text())
    model = joblib.load(MODELS_DIR / f"{nama}.joblib")
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_test, model.predict(X_test), labels=[0, 1]).ravel())
    cek(f"{nama}: json = hitung ulang dari model", (saved["tn"], saved["fp"], saved["fn"], saved["tp"]) == (tn, fp, fn, tp))
    manual = {"accuracy": (tp + tn) / (tp + tn + fp + fn), "precision": tp / (tp + fp),
              "recall": tp / (tp + fn)}
    manual["f1"] = 2 * manual["precision"] * manual["recall"] / (manual["precision"] + manual["recall"])
    cek(f"{nama}: metrik = rumus manual dari TN/FP/FN/TP",
        all(abs(saved[k] - v) < 1e-9 for k, v in manual.items()))
    # model yang disimpan sama dengan model yang dilatih ulang (deterministik)
    ulang = GaussianNB().fit(Xr, yr)
    cek(f"{nama}: model tersimpan = hasil latih ulang",
        (ulang.predict(X_test) == model.predict(X_test)).all())

print("\nSEMUA PEMERIKSAAN LULUS" if not gagal else f"\nADA YANG GAGAL: {gagal}")
sys.exit(1 if gagal else 0)