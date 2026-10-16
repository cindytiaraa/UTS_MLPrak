"""Naive Bayes (GaussianNB): nb_over dan nb_under.   Jalankan: python -m src.naive_bayes"""
from sklearn.naive_bayes import GaussianNB

from src.experiment import run_model


def run() -> dict:
    return run_model("nb", "5. NAIVE BAYES (GaussianNB)", GaussianNB)


if __name__ == "__main__":
    run()
