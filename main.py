from data_utils import display_dataset_summary, load_dataset
from data_preparation import dataset_choice, prepare_features, split_and_clean_data
from model_selection import model_choice
from model_training import train_and_evaluate


def ask_run_again():
    while True:
        answer = input("Run another experiment? (yes/no): ").strip().lower()

        if answer == "yes":
            return True
        elif answer == "no":
            return False
        else:
            print("Invalid answer. Please enter yes or no.")


def main():
    print("Welcome to the PP4AI Classification Application by Paul and Lorenzo.")

    while True:
        dataset_name = dataset_choice()
        dataframe = load_dataset(dataset_name)
        if dataframe is not None:
            display_dataset_summary(dataset_name, dataframe)
            X, y = prepare_features(dataframe)
            model = model_choice()

            try:
                X_train, X_test, y_train, y_test = split_and_clean_data(X, y, model)
            except ValueError as error:
                print(f"Cannot prepare the selected data: {error}")
            else:
                train_and_evaluate(model, X_train, X_test, y_train, y_test, dataset_name)

        if not ask_run_again():
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
