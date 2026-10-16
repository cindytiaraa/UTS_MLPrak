from pathlib import Path

# --- Path ---
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "predictive_maintenance.csv"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
BASELINE_DIR = OUTPUT_DIR / "baseline"

# --- Parameter split ---
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80% training : 20% testing

# --- Parameter resampling ---
OVERSAMPLING_STRATEGY = 0.5
UNDERSAMPLING_STRATEGY = 1.0

# --- Parameter model ---
TREE_CRITERION = "entropy"

# --- Kolom ---
TARGET = "Target"
TYPE_COLUMN = "Type"
ID_COLUMNS = ["UDI", "Product ID"]          # hanya identitas
LEAKAGE_COLUMNS = ["Failure Type"]          # bocoran label
DROP_COLUMNS = ID_COLUMNS + LEAKAGE_COLUMNS

NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# --- Transformasi ---
TYPE_MAPPING = {"L": 0, "M": 1, "H": 2}

# --- Outlier ---
IQR_FACTOR = 1.5

# --- Output tugas klasifikasi ---
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
PREPROCESS_PATH = MODELS_DIR / "preprocess.joblib"
BEST_MODEL_PATH = RESULTS_DIR / "best_model.txt"
COMPARISON_CSV = RESULTS_DIR / "comparison.csv"

# --- Resampling (sama dengan UTS, hanya untuk data training) ---
OVERSAMPLING_STRATEGY = 0.5     # BorderlineSMOTE
UNDERSAMPLING_STRATEGY = 1.0    # RandomUnderSampler

# --- Parameter model ---
SVM_PARAMS = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}   # default SVC, dituliskan eksplisit
DT_PARAMS = {"criterion": "entropy"}   # cocok dengan hasil UTS (lihat src/decision_tree.py)

# --- Enam eksperimen: nama -> (model, skenario) ---
EXPERIMENTS = ["nb_over", "nb_under", "svm_over", "svm_under", "dt_over", "dt_under"]

# --- Hasil Decision Tree pada laporan UTS (untuk verifikasi) : tn, fp, fn, tp ---
UTS_DT_REFERENCE = {
    "dt_over": (1884, 48, 13, 55),
    "dt_under": (1716, 216, 4, 64),
}
