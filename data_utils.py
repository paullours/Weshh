from pathlib import Path

import pandas as pd


DATASETS = {
    "iris": "iris.csv",
    "wine": "wine.csv",
    "breast_cancer": "breast_cancer.csv",
}


def load_dataset(dataset_name, data_folder="data"):
    """Return a supplied dataset, or None after explaining a loading problem."""
    if dataset_name not in DATASETS:
        print("Unknown dataset. Choose iris, wine, or breast_cancer.")
        return None

    dataset_path = Path(data_folder) / DATASETS[dataset_name]

    try:
        dataframe = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Dataset file not found: {dataset_path}. Check the data folder.")
        return None
    except OSError as error:
        print(f"Cannot read dataset file {dataset_path}: {error}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Dataset file is empty or has no columns: {dataset_path}.")
        return None
    except pd.errors.ParserError as error:
        print(f"Invalid CSV format in {dataset_path}: {error}")
        return None
    except UnicodeDecodeError:
        print(f"Cannot read text in {dataset_path}. Expected a UTF-8 CSV file.")
        return None

    if dataframe.shape[0] == 0:
        print(f"Dataset has no observations: {dataset_path}.")
        return None

    if "target" not in dataframe.columns:
        print(f"Dataset {dataset_path} must contain a column named target.")
        return None

    if dataframe["target"].isna().sum() > 0:
        print(f"Dataset {dataset_path} contains missing target labels.")
        return None

    return dataframe


def display_dataset_summary(dataset_name, dataframe):
    """Print the summary of a DataFrame returned by load_dataset; return None."""
    print(f"\nDataset: {dataset_name}")
    print("Rows:", dataframe.shape[0])
    print("Columns:", dataframe.shape[1])
    print("Target column: target")
    print("Observations per target class:")
    print(dataframe["target"].value_counts())
    print("Missing values per column:")
    print(dataframe.isna().sum())
