from data_utils import DATASETS, display_dataset_summary, load_dataset


def ask_run_again():
    while True:
        answer = input("Run another experiment? (yes/no): ").strip().lower()

        if answer == "yes":
            return True
        elif answer == "no":
            return False
        else:
            print("Invalid answer. Please enter yes or no.")


if __name__ == "__main__":
    print("PP4AI project by Paul and Lorenzo.")
    print("Dataset inspection demonstration (before cleaning).")

    for dataset_name in DATASETS:
        dataframe = load_dataset(dataset_name)
        if dataframe is not None:
            display_dataset_summary(dataset_name, dataframe)
