# Weshh: PP4AI Classification Application

University project by **Paul and Lorenzo**.

The finished application will let a user select one of three supplied datasets
and one of three classification models, then train and evaluate the model.
This branch implements **Issue #1: project setup only**. Menus, data loading,
cleaning, training and evaluation are not implemented yet.

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

Expected output at this stage:

```text
PP4AI project by Paul and Lorenzo.
Project setup only: application features are not implemented yet.
```

## Files

```text
Weshh/
    main.py
    requirements.txt
    README.md
    data/
        iris.csv
        wine.csv
        breast_cancer.csv
```

The CSV files are unchanged copies from the supplied PP4AI project archive.
Their final column is `target`; missing feature values are empty fields.
Keep the original CSVs unchanged. Later features will clean data in memory.
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

This setup uses L1's `print()` example and L4's instructions for moving notebook
code into Python scripts and installing libraries from the terminal.
`main.py` currently has no functions or machine learning logic.

## Validation and AI assistance

Setup checks performed on Windows using Python 3.13.0 in a temporary environment:

- `python -m pip install -r requirements.txt`: completed successfully with
  pandas 3.0.6 and scikit-learn 1.9.1.
- `python main.py`: printed the two expected setup messages and exited normally.
- `python -m pip check`: reported no broken requirements.
- SHA-256 checks: all three copied CSVs match the files in the supplied archive.

The nine dataset/model combinations and input validation belong to later PRs
and have not been tested at this stage.

Codex helped prepare Issues and the project setup. Its suggestions were compared
with the assignment and personal notebooks. Verification results below describe
checks actually performed; both students still need to review and explain these
changes before merging.

## Submission

Project ZIP due **October 16, 2026**; practical/oral evaluation **October 19, 2026**.
The final ZIP must contain source files, the three supplied CSVs,
`requirements.txt` and this README. Exclude virtual environments, caches and
credentials. Streamlit and other optional features are deferred until the
required terminal application is finished and an extension is explicitly approved.
