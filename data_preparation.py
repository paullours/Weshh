from data_utils import display_dataset_summary, load_dataset as load_supplied_dataset
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


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
    dataset_names = {
        "Iris": "iris",
        "Wine": "wine",
        "Breast Cancer": "breast_cancer"
    }

    return load_supplied_dataset(dataset_names.get(choice, choice))


def summarize_dataset(dataset, dataset_name="Selected dataset"):
    display_dataset_summary(dataset_name, dataset)
    print("\nColumn names:", dataset.columns.tolist())
    print("\nFirst 5 rows:")
    print(dataset.head())


def prepare_features(dataset):
    X = dataset.drop(columns=["target"])
    y = dataset["target"]
    return X, y


def split_and_clean_data(X, y, model):
    """Split and impute features, scaling for the selected classifier object."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    if isinstance(model, (LogisticRegression, KNeighborsClassifier)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    print("\nTraining rows:", len(X_train))
    print("Test rows:", len(X_test))

    return X_train, X_test, y_train, y_test
