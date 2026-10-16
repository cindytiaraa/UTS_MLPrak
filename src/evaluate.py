"""Evaluasi model + penyimpanan keluaran sesuai kontrak (dipakai Cindy, Alisya, Nadine).

Setiap model memanggil evaluate_and_save(...) sehingga hasilnya seragam:
    output/results/<name>.json      -> accuracy, precision, recall, f1, tn, fp, fn, tp
    output/models/<name>.joblib     -> model hasil training
    output/figures/cm_<name>.png    -> gambar confusion matrix
<name> : nb_over | nb_under | svm_over | svm_under | dt_over | dt_under
"""
import json

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score)

from src import config
from src.utils import save_fig, subbanner

RESULTS_DIR = config.RESULTS_DIR
MODELS_DIR = config.MODELS_DIR

LABELS = ["Normal (0)", "Gagal (1)"]


def evaluate_and_save(model, X_test, y_test, name: str, verbose: bool = True) -> dict:
    """Evaluasi model pada data uji asli, simpan json + model + gambar confusion matrix.

    Label kelas positif = 1 (gagal). Confusion matrix dibaca dengan cm.ravel()
    -> (tn, fp, fn, tp), BUKAN dengan indeks manual seperti kode contoh modul
    (di modul TP/TN tertukar sehingga "sensitivity" sebenarnya precision).
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    metrics = {
        "name": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    save_fig(config.FIGURES_DIR / f"cm_{name}.png")

    if verbose:
        print_report(y_test, y_pred, metrics)
    return metrics


def print_report(y_test, y_pred, metrics: dict) -> None:
    """Cetak classification report, confusion matrix, dan ringkasan metrik (untuk screenshot)."""
    subbanner(f"Classification report - {metrics['name']}")
    print(classification_report(y_test, y_pred, target_names=LABELS,
                                digits=4, zero_division=0))

    subbanner(f"Confusion matrix - {metrics['name']}")
    print("                 Prediksi Normal  Prediksi Gagal")
    print(f"Aktual Normal    {metrics['tn']:>15}  {metrics['fp']:>14}   (TN, FP)")
    print(f"Aktual Gagal     {metrics['fn']:>15}  {metrics['tp']:>14}   (FN, TP)")

    subbanner(f"Ringkasan metrik - {metrics['name']} (kelas positif = Gagal)")
    print(f"Accuracy : {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall   : {metrics['recall']:.2%}")
    print(f"F1-score : {metrics['f1']:.2%}")
