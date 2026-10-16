import argparse

from src import init_dataset, preprocessing, transformation, split_data, undersampling
from src.oversampling import jalankan_eksperimen_oversampling
from src import naive_bayes
from src.utils import quiet

# Tahap baseline (dari UTS) lalu Naive Bayes (bagian Cindy).
STAGES = ["init", "preprocessing", "transformation", "split", "naive_bayes"]


def main(stage: str = "all") -> None:
    """Menjalankan pipeline sesuai stage. Pada mode satu tahap, tahap sebelumnya tetap
    berjalan diam-diam sehingga layar hanya berisi output tahap yang dipilih."""
    last = len(STAGES) if stage == "all" else STAGES.index(stage) + 1

    def call_stage(name, fn, *args):
        return fn(*args) if stage in ("all", name) else quiet(fn, *args)

    df_raw = call_stage("init", init_dataset.run)
    if last == 1:
        return
    df_clean = call_stage("preprocessing", preprocessing.run, df_raw)
    if last == 2:
        return
    X, y, _scaler = call_stage("transformation", transformation.run, df_clean)
    if last == 3:
        return
    call_stage("split", split_data.run, X, y)
    if last == 4:
        return
    naive_bayes.run()      # selalu ditampilkan: nb_over dan nb_under

    if stage == "all":
        print(
            "\nBaseline selesai: "
            "X_train, X_test, y_train, y_test siap dipakai."
        )

        jalankan_eksperimen_oversampling()

        undersampling.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline UTS + Naive Bayes (bagian Cindy)."
    )

    parser.add_argument(
        "--stage",
        choices=STAGES + ["all"],
        default="all",
        help="Tahap pipeline yang ingin ditampilkan.",
    )

    args = parser.parse_args()

    main(args.stage)
