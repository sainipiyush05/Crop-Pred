import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from dataclasses import dataclass
from typing import Tuple


# ── Constants ─────────────────────────────────────────────────────────────────

# Exact column order the ML model expects — must never change after training
FEATURE_COLUMNS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
LABEL_COLUMN    = 'label'

# Paths
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DATASET_PATH = os.path.join(DATA_DIR, 'agroware_dataset_200_crops_improved.csv')
SCALER_PATH  = os.path.join(MODELS_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')


@dataclass
class PipelineOutput:
    X_train:        np.ndarray
    X_test:         np.ndarray
    y_train:        np.ndarray
    y_test:         np.ndarray
    scaler:         StandardScaler
    label_encoder:  LabelEncoder
    feature_columns: list
    class_names:    list      # decoded crop names in label order
    n_classes:      int
    n_features:     int
    train_size:     int
    test_size:      int


class DataPipeline:
    """
    Loads the crop dataset and prepares it for model training.

    Steps:
      1. Load CSV
      2. Validate — check columns, nulls, data types
      3. Clean    — remove duplicates, clamp outliers
      4. Encode   — LabelEncoder on crop names → integers
      5. Scale    — StandardScaler on feature columns
      6. Split    — stratified train/test split (80/20)
      7. Save     — persist scaler and label encoder as .pkl files

    The scaler and label encoder MUST be saved and reused at inference time.
    If you retrain, the saved .pkl files are automatically overwritten.
    """

    def run(self, test_size: float = 0.2, random_state: int = 42) -> PipelineOutput:
        """
        Full pipeline. Returns a PipelineOutput ready for model training.

        Args:
            test_size    : fraction of data for test set (default 0.2 = 20%)
            random_state : seed for reproducibility
        """
        print("=" * 55)
        print("  AgroWare Data Pipeline")
        print("=" * 55)

        # Step 1 — Load
        df = self._load(DATASET_PATH)

        # Step 2 — Validate
        self._validate(df)

        # Step 3 — Clean
        df = self._clean(df)

        # Step 4 — Encode labels
        df, label_encoder = self._encode_labels(df)

        # Step 5 — Split features and target
        X = df[FEATURE_COLUMNS].values.astype(np.float32)
        y = df[LABEL_COLUMN].values.astype(np.int32)

        # Step 6 — Stratified train/test split
        # Stratified ensures each crop class is proportionally represented
        # in both train and test sets — critical with only 30 samples per class
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,            # keeps class distribution equal in both sets
        )

        print(f"\n[Split] Train: {len(X_train)} rows | Test: {len(X_test)} rows")
        print(f"[Split] Each class → ~{int(len(X_train) / len(label_encoder.classes_))} train, "
              f"~{int(len(X_test) / len(label_encoder.classes_))} test samples")

        # Step 7 — Scale features
        # Fit scaler ONLY on training data — never on test data (data leakage)
        X_train, X_test, scaler = self._scale(X_train, X_test)

        # Step 8 — Save scaler and encoder
        os.makedirs(MODELS_DIR, exist_ok=True)
        self._save_artifacts(scaler, label_encoder)

        class_names = list(label_encoder.classes_)

        print(f"\n[Pipeline] Done.")
        print(f"  Features  : {FEATURE_COLUMNS}")
        print(f"  Classes   : {len(class_names)} crops")
        print(f"  Train set : {len(X_train)} samples")
        print(f"  Test set  : {len(X_test)} samples")
        print("=" * 55)

        return PipelineOutput(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            label_encoder=label_encoder,
            feature_columns=FEATURE_COLUMNS,
            class_names=class_names,
            n_classes=len(class_names),
            n_features=len(FEATURE_COLUMNS),
            train_size=len(X_train),
            test_size=len(X_test),
        )

    # ── Step 1: Load ──────────────────────────────────────────────────────────

    def _load(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Dataset not found at: {path}\n"
                f"Make sure the CSV is in the data/ folder."
            )
        df = pd.read_csv(path)
        print(f"\n[Load] {len(df)} rows loaded from {os.path.basename(path)}")
        return df

    # ── Step 2: Validate ──────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame):
        print(f"[Validate] Checking dataset...")

        # Check required columns exist
        required = FEATURE_COLUMNS + [LABEL_COLUMN]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        # Check for nulls
        null_counts = df[required].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Null values found:\n{null_counts[null_counts > 0]}")

        # Check label column is string
        if df[LABEL_COLUMN].dtype not in ['object', 'string']:
            raise ValueError(f"Label column must be string type, got {df[LABEL_COLUMN].dtype}")

        # Check class balance
        class_counts = df[LABEL_COLUMN].value_counts()
        min_count = class_counts.min()
        max_count = class_counts.max()
        if min_count < 10:
            print(f"  [WARNING] Some classes have fewer than 10 samples: "
                  f"{class_counts[class_counts < 10].to_dict()}")

        print(f"  Classes   : {df[LABEL_COLUMN].nunique()}")
        print(f"  Rows      : {len(df)}")
        print(f"  Nulls     : 0 ✓")
        print(f"  Per class : min={min_count}, max={max_count} ✓")

    # ── Step 3: Clean ─────────────────────────────────────────────────────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        original_len = len(df)

        # Remove exact duplicate rows
        df = df.drop_duplicates()
        dupes_removed = original_len - len(df)
        if dupes_removed > 0:
            print(f"[Clean] Removed {dupes_removed} duplicate rows")

        # Strip whitespace from label names
        df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip().str.lower()

        # Clamp feature values to valid agronomic ranges
        # These ranges are based on the dataset min/max + small buffer
        clamp_ranges = {
            'N':           (0,   200),
            'P':           (0,   150),
            'K':           (0,   150),
            'temperature': (0,   50),
            'humidity':    (0,   100),
            'ph':          (0,   14),
            'rainfall':    (0,   500),
        }
        for col, (low, high) in clamp_ranges.items():
            before = len(df[(df[col] < low) | (df[col] > high)])
            df[col] = df[col].clip(lower=low, upper=high)
            if before > 0:
                print(f"[Clean] Clamped {before} out-of-range values in '{col}'")

        print(f"[Clean] {len(df)} rows after cleaning")
        return df

    # ── Step 4: Encode labels ─────────────────────────────────────────────────

    def _encode_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
        """
        Converts crop name strings to integers.
        e.g. 'wheat' → 185, 'rice' → 134
        The encoder is saved so we can decode predictions back to names.
        """
        encoder = LabelEncoder()
        df = df.copy()
        df[LABEL_COLUMN] = encoder.fit_transform(df[LABEL_COLUMN])

        print(f"\n[Encode] {len(encoder.classes_)} crop classes encoded")
        print(f"  Example: 'wheat' → {encoder.transform(['wheat'])[0]}, "
              f"'rice' → {encoder.transform(['rice'])[0]}")

        return df, encoder

    # ── Step 5: Scale ─────────────────────────────────────────────────────────

    def _scale(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        StandardScaler: removes mean, scales to unit variance.
        Formula: z = (x - mean) / std

        IMPORTANT: fit only on X_train, then transform both train and test.
        Fitting on X_test would leak test distribution into training — data leakage.
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
        X_test_scaled  = scaler.transform(X_test)         # transform only on test

        print(f"\n[Scale] StandardScaler applied")
        print(f"  Feature means  : {scaler.mean_.round(2)}")
        print(f"  Feature stds   : {scaler.scale_.round(2)}")

        return X_train_scaled, X_test_scaled, scaler

    # ── Step 6: Save artifacts ────────────────────────────────────────────────

    def _save_artifacts(self, scaler: StandardScaler, encoder: LabelEncoder):
        """
        Saves scaler and label encoder as .pkl files.
        These MUST be loaded at inference time to transform new inputs
        the same way the training data was transformed.
        """
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        with open(ENCODER_PATH, 'wb') as f:
            pickle.dump(encoder, f)

        print(f"\n[Save] scaler.pkl       → {SCALER_PATH}")
        print(f"[Save] label_encoder.pkl → {ENCODER_PATH}")


# ── Utility: load saved artifacts ─────────────────────────────────────────────

def load_scaler() -> StandardScaler:
    """Load the saved scaler for use at inference time."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "scaler.pkl not found. Run DataPipeline().run() first to train the model."
        )
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)


def load_label_encoder() -> LabelEncoder:
    """Load the saved label encoder for use at inference time."""
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            "label_encoder.pkl not found. Run DataPipeline().run() first to train the model."
        )
    with open(ENCODER_PATH, 'rb') as f:
        return pickle.load(f)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = DataPipeline()
    output   = pipeline.run()

    print(f"\nX_train shape : {output.X_train.shape}")
    print(f"X_test shape  : {output.X_test.shape}")
    print(f"y_train shape : {output.y_train.shape}")
    print(f"Classes       : {output.class_names[:5]} ... ({output.n_classes} total)")

    # Verify scaler is working — mean of scaled train should be ~0
    print(f"\nScaled X_train column means (should be ~0): "
          f"{output.X_train.mean(axis=0).round(3)}")
    print(f"Scaled X_train column stds  (should be ~1): "
          f"{output.X_train.std(axis=0).round(3)}")