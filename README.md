# Weshh: PP4AI Classification Application

University project by **Paul and Lorenzo**.

The finished application will let a user select one of three supplied datasets
and one of three classification models, then train and evaluate the model.
Issues #2–#4 provide dataset inspection, preparation and classifier selection. Running
`main.py` demonstrates the summaries for all three supplied datasets before
cleaning. The complete interactive training and evaluation workflow is still pending.

## Install and run

Use an installed Python interpreter in your terminal and select that same
interpreter in VS Code. From the project folder, run:

```text
python --version
python -m pip install -r requirements.txt
python main.py
```

`requirements.txt` lists the two external libraries required for the application:
**pandas** for CSV data and **scikit-learn** for classification.
`python -m pip` runs Python's package installer for the selected interpreter.

The program displays the dataset name, rows, columns, target column, observations
per target class, and missing values in every column (including `target`).
These results are calculated from the CSV contents:

| Dataset | Rows | Columns including target | Target class counts | Missing feature values |
| --- | ---: | ---: | --- | ---: |
| iris | 150 | 5 | 0: 50, 1: 50, 2: 50 | 15 |
| wine | 178 | 14 | 0: 59, 1: 71, 2: 48 | 18 |
| breast_cancer | 569 | 31 | 0: 212, 1: 357 | 57 |

`value_counts()` normally displays the most frequent class first, so Wine prints
class 1 before class 0. There are no missing targets. Iris has 4, 4, 4 and 3 missing
values in its four feature columns. Wine's first five features have 2 each and
the remaining eight have 1 each. Breast Cancer's first 27 features have 2 each
and the last three have 1 each. The program prints each column's name and count;
the total in this table is a verification result, not an extra program output.

## Files

```text
Weshh/
    .gitattributes
    main.py
    data_utils.py
    data_preparation.py
    model_selection.py
    model_training.py
    requirements.txt
    README.md
    data/
        iris.csv
        wine.csv
        breast_cancer.csv
```

The CSV files are unchanged copies from the supplied PP4AI project archive.
Their final column is `target`; missing feature values are empty fields.
Keep the original CSVs unchanged. Data preparation cleans data in memory.
`.gitattributes` disables Git's line-ending conversion for the supplied CSVs,
so a Windows checkout preserves their original bytes.
Run commands from the project folder; no personal absolute paths are required.

## Course style and collaboration

Our primary reference is Paul's personal `Copie de PP4AI_L1.ipynb` through
`Copie de PP4AI_L4.ipynb`, including the completed exercises and lesson examples.
Use short functions, variables, lists, dictionaries, conditions, loops,
return values, basic exception handling, pandas and the required scikit-learn
operations. Explain necessary unfamiliar techniques before implementing them.
Avoid unnecessary frameworks, custom class architectures and abstractions.

Each functionality has one Issue, one feature branch and one Pull Request.
Never push directly to `main`. The other student must review and approve before
merging: Lorenzo (`gangio25`) reviews Paul's PRs; Paul (`paullours`) reviews
Lorenzo's PRs. Both students must understand and explain the whole application.

`main.py` imports the dictionary and functions from `data_utils.py`, loops over
the three names, and displays a summary only if loading returned a DataFrame.
This is a temporary demonstration, without an interactive menu.

Issue #6 adds standalone menu helpers: `data_preparation.dataset_choice()` accepts
1-3 and returns `iris`, `wine` or `breast_cancer`; `main.ask_run_again()` accepts
yes/no (ignoring case and surrounding whitespace) and returns `True`/`False`.
Invalid answers retry. Importing `main` does not run the demonstration; running
`python main.py` still does. The helpers are not connected to an application loop.

## Dataset functions and course examples

`DATASETS` maps the supported names `iris`, `wine`, and `breast_cancer` to their
supplied CSV filenames. It follows L1's product dictionary and key lookup
(cells 69, 72 and 78). Arbitrary dataset names and uploads are not supported.

- `load_dataset(dataset_name, data_folder="data")` receives a supported name and
  an optional folder. It reads the CSV, checks that it has observations and a
  `target` column with no missing labels, then returns the DataFrame. If a known
  file or validation problem occurs, it prints an explanation and returns `None`.
  The optional folder lets checks use temporary copies with the same filenames.
- `display_dataset_summary(dataset_name, dataframe)` receives a name and the
  validated DataFrame returned by the loader. It prints the summary and returns
  `None` implicitly. It does not change the DataFrame or save anything to disk.

`data_preparation.load_dataset(choice)` reuses this loader and accepts both
`Iris` / `Wine` / `Breast Cancer` and `iris` / `wine` / `breast_cancer`.
Its `summarize_dataset(dataset, dataset_name="Selected dataset")` reuses the
required summary and adds column names and the first five rows.
`prepare_features()` separates features from `target`; `split_and_clean_data(X, y, model)`
uses a stratified 80/20 split with seed 42, fills missing features with training
means, and scales features only for Logistic Regression and KNN. StandardScaler
puts features on comparable scales for Logistic Regression's optimization and
regularization and KNN's distance calculations. Decision Trees use imputed,
unscaled features. The imputer and, when used, scaler learn from training features
only; test features use the same fitted objects. No rows are removed.
For Issue #7, call `model = model_choice()` before
`split_and_clean_data(X, y, model)`: the new required argument is the classifier
object, not a name or menu number. The four return values remain
`X_train, X_test, y_train, y_test`. `model_selection.model_choice()` returns
Logistic Regression, Decision Tree or KNN through its existing terminal prompt.

Issue #7 adds `model_training.train_and_evaluate(model, X_train, X_test,
y_train, y_test, dataset_name)`. Pass the selected classifier and the four arrays
from `split_and_clean_data(X, y, model)`. It fits only on training data, predicts
test labels, and calculates `accuracy_score(y_test, predictions)`: the fraction
of test observations classified correctly. It prints the dataset, readable model
name and percentage, and returns the numerical accuracy. Preparation already
prints training/test counts, so the new function does not repeat them. Empty or
misaligned sets and scikit-learn `ValueError` failures print an explanation and
return `None` without reporting an accuracy. Other errors propagate.

Issue #7 verification (Python 3.13.0, pandas 3.0.6, scikit-learn 1.9.1), using
the supplied CSVs and existing classifier settings:

| Dataset | Training/test rows | Logistic Regression | Decision Tree | KNN |
| --- | --- | ---: | ---: | ---: |
| Iris | 120/30 | 96.67% | 90.00% | 93.33% |
| Wine | 142/36 | 97.22% | 94.44% | 97.22% |
| Breast Cancer | 455/114 | 98.25% | 92.11% | 95.61% |

All nine experiments passed twice without warnings. Returned scores matched
independent test accuracy, stayed within 0-1, and repeated exactly. Checks verified
training-only fitting and preprocessing, invalid-data handling, module imports,
existing menus and the unchanged `main.py` demonstration. Existing preprocessing
and classifier code remained unchanged; CSV bytes still matched the supplied ZIP.
The verification script stayed outside the repository. Full interactive
integration and handling preparation failures remain for Issue #8.

Cell numbers below are zero-based positions in Paul's personal notebooks, not
execution numbers. Examples included in a notebook do not prove that Paul
completed the corresponding exercises.

| Operation | Simple explanation | Personal notebook reference |
| --- | --- | --- |
| `pd.read_csv(dataset_path)` | Read a CSV into a table called a DataFrame; empty fields become missing values | L3 cell 142; L4 cell 62 has a commented local CSV alternative |
| `dataframe.shape[0]` / `[1]` | Read the row / column count; `shape` is an attribute, so it has no parentheses | L3 cell 144, with L1 indexing |
| `dataframe.columns` | Read column names to check that `target` exists | L3 cell 145 |
| `dataframe["target"]` | Select the target labels | L3 cell 149 selects `species` |
| `.value_counts()` | Count observations for each distinct label | L3 cell 176; L4 cell 63 |
| `.isna().sum()` | Mark missing values with `True`, then count them per column | L3 cells 161-162; L4 cell 63 |
| `try` / `except` | Respond to a particular expected error with a useful message | L3 cell 125 |
| `Path(...)` | Represent a file path | L4 cells 92 and 94 |

L3 cells 37-38 contain Paul's executed discounted-price exercise using parameters
and a return value; cells 59-60 contain his inventory-value function and call.
The pandas operations above are lesson examples. L3 Penguins exercise cells
203-208 are blank; cells 212-218 are supplied solutions after the instruction to
read them after finishing or when stuck. L4's inspection cell 63 has no recorded
execution count. We reuse the examples without claiming those exercises were
completed.

The new local import `from data_utils import ...` uses our own Python file as a
module. It extends L3's `from math import sqrt` (cell 138): the functions are
defined once and reused by `main.py`. `Path(data_folder) / filename` joins the
folder and filename using the operating system's path rules. The default
parameter supplies `"data"` when no folder is passed; L4 cell 53 also demonstrates
function parameters with defaults.

The specific file exceptions extend L3's `try/except` example:
`FileNotFoundError` means the file is missing; `OSError` covers permission and
other file-reading errors; `pd.errors.EmptyDataError` means pandas found no
columns; `pd.errors.ParserError` means it could not parse the CSV; and
`UnicodeDecodeError` means the text could not be read as UTF-8. `as error` stores
the error details so we can include them in the message, as in L4 cell 50.
There is no catch-all exception handler.

When a CSV is missing, the loader names its path, explains that it was not found,
and returns `None`. The demonstration skips its summary, continues to the other
datasets, and finishes normally. A header-only CSV and missing target labels
also produce clear messages rather than an invalid summary.

## Validation and AI assistance

Setup checks performed on Windows using Python 3.13.0 in a temporary environment:

- `python -m pip install -r requirements.txt`: completed successfully with
  pandas 3.0.6 and scikit-learn 1.9.1.
- `python main.py`: printed the two expected setup messages and exited normally.
- `python -m pip check`: reported no broken requirements.
- SHA-256 checks: all three copied CSVs match the files in the supplied archive.

Issue #2 checks executed on Windows with Python 3.13.0 and pandas 3.0.6:

- `python main.py` using the existing temporary Python environment: all three
  complete summaries printed and the process exited with code 0.
- A separate temporary verification script ran 17 check groups successfully:
  three real dataset checks (dimensions, target validity, class counts, missing
  totals, printed summaries, and unchanged DataFrames); eight file/validation
  cases (missing, empty, header-only, absent target, entirely missing target,
  partly missing target, malformed CSV, invalid UTF-8); an unreadable path
  check; an unsupported name check; a demonstration run with a temporarily
  renamed Iris copy; and three original CSV byte comparisons.
- The unreadable path was a directory named `iris.csv`, which produced an actual
  Windows permission error. A regular CSV with restricted permissions was not
  separately tested. No mocked file-reading errors were used.
- The rename check used a temporary copy of `data/`: the missing Iris message
  appeared, Wine and Breast Cancer still printed, exit code was 0, stderr was
  empty, and the copied filename was restored. Original filenames were untouched.
- After verification, all three original CSVs matched both their pre-test bytes
  and the supplied ZIP bytes. SHA-256 values were Iris
  `6feed722b6046b4e3efc135dfcf354df0bd8bf68b89a84a167d02bf124969733`, Wine
  `f77e0a47ccdefd5d15e83039cb2b64f3c3f394ba81d8cc2c9dd9690455f924ef`, and Breast Cancer
  `7540c4261e92627b4d78a4fe3cfcb20ea516fa70779a58c61aa627b15b86040c`.

The verification script stayed outside the repository; it used standard-library
assertions, output capture, temporary fixtures, a subprocess run, and byte/hash
comparisons. Those testing techniques are separate from the student application.
For a quick manual missing-file check, run this from the project root:

```text
python -c "from data_utils import load_dataset; load_dataset('iris', 'folder_that_does_not_exist')"
```

Integration cleanup checks (Python 3.13.0, pandas 3.0.6, scikit-learn 1.9.1):
34 temporary check groups passed, including both dataset-name styles, invalid
names, file errors, complete summaries, reproducible training-only preprocessing,
existing prompt retries and `main.py`. All nine dataset/model smoke tests trained
and predicted without warnings. The CSVs still match the original ZIP. These
checks exercise functions independently; the final interactive workflow is pending.

Issue #5 checks (same Python and library versions): all nine dataset/model
combinations trained and predicted without warnings. Independent numerical
checks verified training-mean imputation, model-specific scaling, reproducible
splits, finite features, dimensions and row/label alignment. Fit-call inspection
and changes to test features confirmed training-only learning and reuse of the
training settings on test features. Existing functions and `main.py` passed;
all three CSVs remained byte-for-byte unchanged and matched the supplied ZIP.
These were independent smoke tests, not the final interactive workflow.

Historical Issue #2 status of the official Section 11 checklist (all 14 checks):

| Check | Status for Issue #2 |
| --- | --- |
| 1. Runs using README commands | Current demonstration verified; final application still pending |
| 2. All three CSVs load | Verified with supplied files |
| 3. All nine dataset/model combinations | Later Issues |
| 4. Summary identifies and program handles missing values | Inspection verified; missing-feature preparation remains for later Issues |
| 5. Target excluded from input features | Later Issues; no feature matrix created yet |
| 6. Learned preprocessing uses training data only | Later Issues; no preprocessing yet |
| 7. Predictions and accuracy use test data | Later Issues |
| 8. Invalid numbers, text and empty selections do not crash | Later Issues; unsupported loader names checked, no menu yet |
| 9. Invalid repeat/exit answers handled | Later Issues |
| 10. Renamed CSV gives helpful error, then restored | Verified using a temporary supplied CSV copy |
| 11. Second experiment without restart | Later Issues; demonstration is not an experiment loop |
| 12. Clean exit when requested | Later Issues; demonstration exits normally, no exit prompt yet |
| 13. Submission archive and README limitations | Current limitations documented; final archive not built or verified |
| 14. Students can explain code and results | Explanation provided; Paul and Lorenzo must demonstrate understanding |

Limitations: run from the repository root; the summary function expects a
DataFrame validated by the loader. `main.py` only inspects the datasets; it does
not call preparation or classifier selection. The full training/evaluation
workflow and integration of repeat/exit handling remain for later Issues. Preparation expects
usable numeric features and sufficient data for stratification. The loader
checks target presence and missingness; it does not perform a general CSV schema
audit. The three original supplied files remain the only supported datasets.

Codex helped prepare Issues, the project setup, Issue #2's loading functions,
and the integration adapters, explanations and checks. Its suggestions were
compared with all 14 sections of the official assignment, the dataset README,
and Paul's personal L1-L4 copies.
The verification notes describe executed checks; both students still need to
review and explain these changes before merging.

## Submission

Project ZIP due **October 16, 2026**; practical/oral evaluation **October 19, 2026**.
The final ZIP must contain source files, the three supplied CSVs,
`requirements.txt` and this README. Exclude virtual environments, caches and
credentials. Streamlit and other optional features are deferred until the
required terminal application is finished and an extension is explicitly approved.
