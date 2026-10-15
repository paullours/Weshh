from sklearn.metrics import accuracy_score


def train_and_evaluate(model, X_train, X_test, y_train, y_test, dataset_name):
    """Train on prepared training data; return test accuracy, or None on invalid data."""
    if len(X_train) == 0 or len(X_test) == 0:
        print("Cannot train and evaluate: training and test sets must not be empty.")
        return None

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        print("Cannot train and evaluate: each feature row needs one target label.")
        return None

    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, predictions)
    except ValueError as error:
        print(f"Cannot train and evaluate the supplied data: {error}")
        return None

    model_name = type(model).__name__
    if model_name == "LogisticRegression":
        model_name = "Logistic Regression"
    elif model_name == "DecisionTreeClassifier":
        model_name = "Decision Tree"
    elif model_name == "KNeighborsClassifier":
        model_name = "K-Nearest Neighbors"

    print(f"\nDataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    return test_accuracy
