import argparse

# 1. TAMBAHKAN 'undersampling' PADA BARIS IMPORT INI
from src import init_dataset, preprocessing, transformation, split_data, undersampling
from src.oversampling import jalankan_eksperimen_oversampling
from src.utils import quiet


STAGES = [
    "init",
    "preprocessing",
    "transformation",
    "split",
]


def main(stage: str = "all") -> None:
    """Menjalankan pipeline sesuai stage yang dipilih."""

    # Menentukan tahap terakhir yang perlu dijalankan.
    if stage == "all":
        last = len(STAGES)
    else:
        last = STAGES.index(stage) + 1

    # Menentukan apakah output suatu tahap perlu ditampilkan.
    def show(name: str) -> bool:
        return stage == "all" or stage == name

    # Menjalankan tahap dengan atau tanpa output.
    def call_stage(name: str, fn, *args):
        if show(name):
            return fn(*args)

        return quiet(fn, *args)

    # =========================================================
    # 1. INIT DATASET
    # =========================================================
    df_raw = call_stage("init", init_dataset.run)

    if last == 1:
        return

    # =========================================================
    # 2. PREPROCESSING
    # =========================================================
    df_clean = call_stage(
        "preprocessing",
        preprocessing.run,
        df_raw,
    )

    if last == 2:
        return

    # =========================================================
    # 3. TRANSFORMATION
    # =========================================================
    X, y, _scaler = call_stage(
        "transformation",
        transformation.run,
        df_clean,
    )

    if last == 3:
        return

    # =========================================================
    # 4. SPLIT DATA
    # =========================================================
    call_stage(
        "split",
        split_data.run,
        X,
        y,
    )

    # =========================================================
    # SELESAI
    # =========================================================
    if stage == "all":
        print(
            "\nBaseline selesai: "
            "X_train, X_test, y_train, y_test siap dipakai."
        )
        
        jalankan_eksperimen_oversampling()
        

        undersampling.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline baseline UTS ML Soal C - Kelompok 3."
    )

    parser.add_argument(
        "--stage",
        choices=STAGES + ["all"],
        default="all",
        help="Tahap pipeline yang ingin ditampilkan.",
    )

    args = parser.parse_args()

    main(args.stage)