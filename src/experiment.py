"""Alur latih-uji yang sama untuk semua model (NB, SVM, DT).

Per skenario: resampling data training -> fit model -> evaluasi pada data uji ASLI.
Tiap file model cukup menyediakan pabrik model (make_model).
"""
from src.evaluate import evaluate_and_save
from src.resampling import SCENARIOS
from src.split_data import load_baseline
from src.utils import banner, subbanner


def run_model(prefix: str, title: str, make_model) -> dict:
    """prefix: 'nb' | 'svm' | 'dt'. Mengembalikan {nama_eksperimen: metrik}."""
    banner(title)
    X_train, X_test, y_train, y_test = load_baseline()
    print(f"Data training asli: {X_train.shape}, gagal = {int(y_train.sum())}")
    print(f"Data uji          : {X_test.shape}, gagal = {int(y_test.sum())} (tidak di-resampling)")

    results = {}
    for suffix, (judul, resample) in SCENARIOS.items():
        name = f"{prefix}_{suffix}"
        subbanner(f"{name} - {judul}")
        X_res, y_res = resample(X_train, y_train)          # hanya data training
        print(f"Setelah resampling: {X_res.shape} | normal = {int((y_res == 0).sum())}, "
              f"gagal = {int((y_res == 1).sum())}")
        model = make_model()
        model.fit(X_res, y_res)
        results[name] = evaluate_and_save(model, X_test, y_test, name)
    return results
