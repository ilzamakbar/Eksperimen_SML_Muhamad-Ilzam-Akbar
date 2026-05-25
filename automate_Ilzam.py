import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')


# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
def load_data(source: str = "seaborn") -> pd.DataFrame:
    """
    Muat dataset Titanic.
    source = 'seaborn'  → langsung dari sns.load_dataset
    source = 'csv'      → dari file titanic_raw/titanic.csv
    """
    if source == "seaborn":
        df = sns.load_dataset('titanic')
        print(f"[LOAD] Dataset berhasil dimuat dari seaborn: {df.shape}")
    else:
        csv_path = os.path.join("titanic_raw", "titanic.csv")
        df = pd.read_csv(csv_path)
        print(f"[LOAD] Dataset berhasil dimuat dari CSV: {df.shape}")
    return df


# ─── 2. SELEKSI KOLOM ─────────────────────────────────────────────────────────
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop kolom yang tidak relevan untuk model.
    Kolom yang dipertahankan sesuai eksperimen notebook.
    """
    cols_to_keep = ['survived', 'pclass', 'sex', 'age',
                    'sibsp', 'parch', 'fare', 'embarked', 'alone']
    df = df[cols_to_keep].copy()
    print(f"[SELECT] Kolom setelah seleksi: {list(df.columns)}")
    return df


# ─── 3. HANDLE MISSING VALUES ─────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    - age  → imputasi median
    - embarked → imputasi modus
    """
    # Age: median imputation
    age_imputer = SimpleImputer(strategy='median')
    df['age'] = age_imputer.fit_transform(df[['age']])

    # Embarked: mode imputation
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

    print(f"[MISSING] Sisa missing values: {df.isnull().sum().sum()}")
    return df


# ─── 4. ENCODING ──────────────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - sex      → binary encoding (male=1, female=0)
    - embarked → one-hot encoding, drop_first=True
    - alone    → cast ke int
    """
    # Binary encoding sex
    df['sex'] = df['sex'].map({'male': 1, 'female': 0})

    # One-hot encoding embarked
    embarked_dummies = pd.get_dummies(df['embarked'], prefix='embarked', drop_first=True)
    df = pd.concat([df.drop(columns=['embarked']), embarked_dummies], axis=1)

    # Bool → int
    df['alone'] = df['alone'].astype(int)

    print(f"[ENCODE] Kolom setelah encoding: {list(df.columns)}")
    return df


# ─── 5. OUTLIER HANDLING ──────────────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    fare: capping pada percentile 99, lalu log1p transform.
    """
    cap = df['fare'].quantile(0.99)
    df['fare'] = df['fare'].clip(upper=cap)
    df['fare'] = np.log1p(df['fare'])
    print(f"[OUTLIER] Fare setelah capping+log transform — max: {df['fare'].max():.4f}")
    return df


# ─── 6. FEATURE ENGINEERING ───────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tambah fitur baru sesuai eksperimen:
    - family_size = sibsp + parch + 1
    - is_child    = 1 jika age < 16
    """
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_child']    = (df['age'] < 16).astype(int)
    print(f"[FEAT_ENG] family_size & is_child ditambahkan.")
    return df


# ─── 7. SPLIT & SCALE ─────────────────────────────────────────────────────────
def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train-test split 80:20, lalu StandardScaler.
    Return: X_train, X_test, y_train, y_test (semua numpy array),
            dan df_train, df_test (DataFrame dengan kolom asli untuk disimpan).
    """
    X = df.drop(columns=['survived'])
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Kembalikan juga sebagai DataFrame agar mudah disimpan
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_df  = pd.DataFrame(X_test_scaled,  columns=X.columns)

    print(f"[SPLIT]  Train: {X_train_df.shape}, Test: {X_test_df.shape}")
    return X_train_df, X_test_df, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


# ─── 8. SIMPAN HASIL ──────────────────────────────────────────────────────────
def save_preprocessed(X_train, X_test, y_train, y_test,
                       output_dir: str = "titanic_preprocessing"):
    """
    Simpan hasil preprocessing ke folder output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df = X_train.copy()
    train_df['survived'] = y_train.values
    test_df  = X_test.copy()
    test_df['survived']  = y_test.values

    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"[SAVE] Train disimpan  → {train_path} ({train_df.shape})")
    print(f"[SAVE] Test disimpan   → {test_path}  ({test_df.shape})")
    return train_path, test_path


# ─── PIPELINE UTAMA ───────────────────────────────────────────────────────────
def run_preprocessing(source: str = "seaborn",
                      output_dir: str = "titanic_preprocessing") -> dict:
    """
    Jalankan seluruh pipeline preprocessing dari awal hingga simpan CSV.
    Return dict berisi path file hasil.
    """
    print("=" * 55)
    print("   Automate Preprocessing — Titanic Dataset")
    print("=" * 55)

    df = load_data(source)
    df = select_features(df)
    df = handle_missing(df)
    df = encode_features(df)
    df = handle_outliers(df)
    df = feature_engineering(df)

    X_train, X_test, y_train, y_test = split_and_scale(df)
    train_path, test_path = save_preprocessed(X_train, X_test, y_train, y_test, output_dir)

    print("=" * 55)
    print("   Preprocessing selesai!")
    print("=" * 55)

    return {
        "train": train_path,
        "test":  test_path,
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
    }


if __name__ == "__main__":
    result = run_preprocessing()
