import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import BorderlineSMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from src import config
from src.split_data import load_baseline
from src.utils import banner, subbanner, save_fig


def jalankan_eksperimen_oversampling():
    banner("5a. OVERSAMPLING PADA TRAINING DATA")

    # =========================================================
    # 1. LOAD BASELINE DATA
    # =========================================================
    subbanner("Memuat data baseline")
    X_train, X_test, y_train, y_test = load_baseline()

    print("Data baseline berhasil dimuat.")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # =========================================================
    # 2. DISTRIBUSI KELAS SEBELUM OVERSAMPLING
    # =========================================================
    subbanner("Distribusi kelas sebelum oversampling")
    print(y_train.value_counts())

    # =========================================================
    # 3. OVERSAMPLING DENGAN BORDERLINE-SMOTE
    # =========================================================
    subbanner("Menerapkan BorderlineSMOTE")

    smote = BorderlineSMOTE(
        sampling_strategy=config.OVERSAMPLING_STRATEGY,
        random_state=config.RANDOM_STATE,
    )

    X_train_res, y_train_res = smote.fit_resample(
        X_train,
        y_train,
    )

    print("Distribusi kelas setelah oversampling:")
    print(y_train_res.value_counts())

    # =========================================================
    # 4. VISUALISASI DISTRIBUSI
    # =========================================================
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    sns.countplot(x=y_train, ax=ax[0])
    ax[0].set_title("Sebelum Oversampling")
    ax[0].set_xlabel("Target")
    ax[0].set_ylabel("Jumlah")

    sns.countplot(x=y_train_res, ax=ax[1])
    ax[1].set_title("Sesudah BorderlineSMOTE")
    ax[1].set_xlabel("Target")
    ax[1].set_ylabel("Jumlah")

    plt.tight_layout()

    save_fig(
        config.FIGURES_DIR / "distribusi_oversampling.png"
    )

    # =========================================================
    # 5. KLASIFIKASI DECISION TREE
    # =========================================================
    banner("6. KLASIFIKASI DECISION TREE")

    dt = DecisionTreeClassifier(
        criterion=config.TREE_CRITERION,
        random_state=config.RANDOM_STATE,
    )

    print(f"Konfigurasi Model: {dt.get_params()}")

    dt.fit(X_train_res, y_train_res)

    # =========================================================
    # 6. TESTING
    # =========================================================
    subbanner("Testing pada data testing")

    y_pred = dt.predict(X_test)

    # =========================================================
    # 7. EVALUASI
    # =========================================================
    banner("7. EVALUASI (PADA DATA TESTING)")

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Normal (0)", "Gagal (1)"],
        )
    )

    # =========================================================
    # 8. CONFUSION MATRIX
    # =========================================================
    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Gagal"],
    )

    disp.plot(
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title("Confusion Matrix - Oversampling")

    plt.tight_layout()

    save_fig(
        config.FIGURES_DIR / "confusion_matrix_oversampling.png"
    )

    print(
        "-> Confusion matrix disimpan di "
        "output/figures/confusion_matrix_oversampling.png"
    )

    return dt, X_train_res, y_train_res, y_test, y_pred


if __name__ == "__main__":
    jalankan_eksperimen_oversampling()