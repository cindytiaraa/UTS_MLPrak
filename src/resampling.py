"""Resampling data training (fungsi murni; tidak melatih model).

Parameter sama dengan UTS dan diambil dari config.py.
HANYA dipanggil pada data training. Data uji tidak boleh di-resampling.
"""
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

from src import config


def oversample(X_train, y_train):
    """BorderlineSMOTE: menambah sampel sintetis kelas gagal (rasio gagal:normal = 0,5)."""
    return BorderlineSMOTE(sampling_strategy=config.OVERSAMPLING_STRATEGY,
                           random_state=config.RANDOM_STATE).fit_resample(X_train, y_train)


def undersample(X_train, y_train):
    """RandomUnderSampler: mengurangi kelas normal sampai sama banyak dengan gagal."""
    return RandomUnderSampler(sampling_strategy=config.UNDERSAMPLING_STRATEGY,
                              random_state=config.RANDOM_STATE).fit_resample(X_train, y_train)


SCENARIOS = {
    "over": ("Oversampling (BorderlineSMOTE)", oversample),
    "under": ("Undersampling (RandomUnderSampler)", undersample),
}
