import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from src import config
from src.split_data import load_baseline
from src.utils import banner, subbanner, save_fig


def run():
    banner("5b. UNDERSAMPLING PADA TRAINING DATA")

    # =========================================================
    # 1. LOAD BASELINE DATA
    # =========================================================
    X_train, X_test, y_train, y_test = load_baseline()

    print("Data baseline berhasil dimuat.")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # =========================================================
    # 2. DISTRIBUSI KELAS SEBELUM UNDERSAMPLING
    # =========================================================
    subbanner("Distribusi kelas sebelum undersampling")
    print(y_train.value_counts())

    # =========================================================
    # 3. UNDERSAMPLING
    # =========================================================
    subbanner("Menerapkan RandomUnderSampler")

    rus = RandomUnderSampler(
        sampling_strategy=config.UNDERSAMPLING_STRATEGY,
        random_state=config.RANDOM_STATE,
    )

    X_train_under, y_train_under = rus.fit_resample(
        X_train,
        y_train,
    )

    print("Distribusi kelas setelah undersampling:")
    print(y_train_under.value_counts())

    # =========================================================
    # 4. VISUALISASI DISTRIBUSI
    # =========================================================
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (title, data) in zip(
        axes,
        [
            ("Sebelum Undersampling", y_train),
            ("Sesudah Undersampling", y_train_under),
        ],
    ):
        sns.countplot(x=data, ax=ax)
        ax.bar_label(ax.containers[0])
        ax.set_title(title)
        ax.set_xlabel("Target")
        ax.set_ylabel("Jumlah")

    plt.tight_layout()

    save_fig(
        config.FIGURES_DIR / "distribusi_undersampling.png"
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

    dt.fit(
        X_train_under,
        y_train_under,
    )

    # =========================================================
    # 6. TESTING
    # =========================================================
    subbanner("Testing pada data testing")

    y_pred = dt.predict(X_test)

    # =========================================================
    # 7. EVALUASI
    # =========================================================
    banner("7. EVALUASI (PADA DATA TESTING)")

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

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

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Gagal"],
    ).plot(
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )

    ax.set_title("Confusion Matrix - Undersampling")

    plt.tight_layout()

    save_fig(
        config.FIGURES_DIR / "confusion_matrix_undersampling.png"
    )

    print(
        "-> Confusion matrix disimpan di "
        "output/figures/confusion_matrix_undersampling.png"
    )

    return (
        dt,
        X_train_under,
        y_train_under,
        y_test,
        y_pred,
    )

if __name__ == "__main__":
    run()