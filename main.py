from data_utils import DATASETS, display_dataset_summary, load_dataset


print("PP4AI project by Paul and Lorenzo.")
print("Dataset inspection demonstration (before cleaning).")

for dataset_name in DATASETS:
    dataframe = load_dataset(dataset_name)
    if dataframe is not None:
        display_dataset_summary(dataset_name, dataframe)
