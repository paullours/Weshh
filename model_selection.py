from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def model_choice():
    while True:
        print("\n--- Choose a Classification Model ---")
        print("1. Logistic Regression")
        print("2. Decision Tree")
        print("3. K-Nearest Neighbors")

        choice = input("Enter your choice (1-3): ").strip()

        if choice == "1":
            return LogisticRegression(max_iter=3000)
        elif choice == "2":
            return DecisionTreeClassifier(random_state=42)
        elif choice == "3":
            return KNeighborsClassifier()
        else:
            print("Invalid choice. Please insert 1, 2 or 3.")


if __name__ == "__main__":
    model = model_choice()
    print("Selected model:", model)