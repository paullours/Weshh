# Weshh — Our PP4AI Project

## 1. About our project

We are Paul and Lorenzo, two Master's students. We built Weshh for our Programming Principles for AI course.

We wanted to make a simple terminal app where we could choose a dataset, explore it, and train a classifier. The app then shows how well the model predicts data it did not see during training.

## 2. How to run it

Open a terminal in the project folder and run:

```text
python -m pip install -r requirements.txt
python main.py
```

Choose a dataset and a model by entering `1`, `2`, or `3`. After each experiment, enter `yes` to try again or `no` to exit. If you enter an invalid answer, the app asks again.

We used pandas and scikit-learn. After installation, the terminal app works offline without an API key or Streamlit.

## 3. Datasets and models

We used the professor's three original CSV files:

- `data/iris.csv`
- `data/wine.csv`
- `data/breast_cancer.csv`

Each file contains numeric features and a `target` column with the class to predict. We kept the original files unchanged.

We used three classifiers: **Logistic Regression**, **Decision Tree**, and **K-Nearest Neighbors (KNN)**. Each can run on all three datasets.

## 4. How our code works

First, the app shows the dataset's size, class counts, and missing values. Then it separates the input features from `target`.

We split our data into two parts: 80% to train the model and 20% to test it. We kept similar class proportions in both parts and used `random_state=42` to get the same split each time.

Next, we fill missing feature values with averages calculated from the training data. No rows are removed.

For Logistic Regression and KNN, we also use `StandardScaler` to put features on similar scales. Decision Tree does not need scaling. Both the replacement averages and the scaling are learned from training data only, then applied to the test data.

Finally, the model makes predictions on the test data. Accuracy tells us what percentage of those predictions were correct.

## 5. Project files

| File | What it does |
| --- | --- |
| `main.py` | Runs the app and handles repeating or exiting. |
| `data_utils.py` | Loads and checks CSVs, then shows summaries. |
| `data_preparation.py` | Handles dataset selection, splitting, missing values, and scaling. |
| `model_selection.py` | Handles model selection and creates the classifier. |
| `model_training.py` | Trains the model and calculates test accuracy. |
| `data/` | Contains the three original datasets. |
| `requirements.txt` | Lists the terminal app's dependencies. |
| `streamlit_app.py` | Runs the optional dashboard. |

## 6. Our results

The recorded automated checks produced these nine accuracy results:

| Dataset | Training/test rows | Logistic Regression | Decision Tree | KNN |
| --- | --- | ---: | ---: | ---: |
| Iris | 120/30 | 96.67% | 90.00% | 93.33% |
| Wine | 142/36 | 97.22% | 94.44% | 97.22% |
| Breast Cancer | 455/114 | 98.25% | 92.11% | 95.61% |

## 7. Tests and limitations

We used Codex to help run automated tests for all nine dataset and model combinations. Each combination was tested twice, with the same results both times. Additional checks confirmed the accuracy calculations, training-only data preparation, input handling, and recovery from file errors. The CSVs also matched the supplied originals.

We also manually tested the terminal application ourselves. A fresh installation and another visual browser check were not completed.

The terminal app must run from the project folder and supports only the supplied numeric datasets. We used one fixed split, without cross-validation or model tuning. We did not require specific library versions, so results may vary between versions. The Breast Cancer predictions are only a classroom exercise.

## 8. AI assistance

We used ChatGPT and Codex for help with planning, coding, debugging, reviewing, testing, and documentation. The work was checked against the assignment and course examples.

## 9. Optional Streamlit dashboard

We also added a dashboard to explore datasets and compare models using the same backend.

From the project folder, run:

```text
python -m pip install streamlit
python -m streamlit run streamlit_app.py
```

Automated checks passed all nine individual runs and all three dataset comparisons.
