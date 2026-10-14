import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def dataset_choice():
    while True:
        choice = input(
            "Choose a dataset (Iris, Wine, Breast Cancer): "
        ).strip()

        if choice in ["Iris", "Wine", "Breast Cancer"]:
            return choice
        else:
            print("Invalid choice. Please try again.")


def load_dataset(choice):
    paths = {
        "Iris": "data/iris.csv",
        "Wine": "data/wine.csv",
        "Breast Cancer": "data/breast_cancer.csv"
    }

    file_path = paths[choice]

    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("Dataset not found.")
        return None


def summarize_dataset(dataset):
    print("\n--- Dataset Summary ---")
    print("Number of rows:", len(dataset))
    print("Number of columns:", len(dataset.columns))
    print("\nColumns:", dataset.columns.tolist())
    print("\nMissing values:")
    print(dataset.isna().sum())
    print("\nFirst 5 rows:")
    print(dataset.head())


def prepare_features(dataset):
    X = dataset.drop(columns=["target"])
    y = dataset["target"]
    return X, y


def split_and_clean_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("\nTraining rows:", len(X_train))
    print("Test rows:", len(X_test))

    return X_train, X_test, y_train, y_test