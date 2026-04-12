# Do Stopwords and Punctuation Matter?
## A Systematic Evaluation for Multi-Class Sentiment Classification of Movie Reviews

**Module:** CS7IS4 — Text Analytics, Trinity College Dublin
**Authors:** Drishya Dinesh, Parth Deshmukh, Mukul Ghare, Tanmay Ghosh, Raghav Manish Gupta, Yilin Wen

---

## What This Project Is About

When people analyse text with a computer (for example, to find out if a movie review is
positive, negative, or neutral), a very common first step is to remove "stopwords" (common
words like *the*, *is*, *not*, *but*) and punctuation marks (!?.,). The assumption is that
these words and symbols carry little useful information and removing them speeds up the
computer.

This project **challenges that assumption**. We test whether removing these items actually
hurts the computer's ability to correctly classify the sentiment of a movie review. We also
test a gentler alternative called **frequency discounting**, where instead of deleting common
words we simply reduce how much weight they carry in the analysis.

### The Dataset

We use **IMDB62** — a publicly available collection of **61,987 movie reviews** from IMDb,
each rated by the reviewer on a scale of 1–10. We map these ratings into three sentiment
categories:

| Rating | Sentiment label |
|--------|-----------------|
| 1–3    | Negative        |
| 4–6    | Neutral         |
| 7–10   | Positive        |

### The Two Models Tested

We test two different machine learning classifiers:

1. **LinearSVC (Linear Support Vector Machine)** — Finds the best mathematical boundary
   between the three sentiment classes. It is one of the most reliable classifiers for
   text data.

2. **ComplementNB (Complement Naive Bayes)** — A probability-based classifier that is
   especially good when one class has many more examples than another.

### The Five Preprocessing Scenarios

Each model is tested under five different text-preparation conditions:

| Scenario | Dataset file used | What is done to the text |
|----------|-------------------|--------------------------|
| S1 — All retained | `imdb62_A.csv` | Nothing removed. Full text used as-is (baseline). |
| S2 — Stopwords removed | `imdb62_B.csv` | Common English words (*the*, *not*, *very*, etc.) removed using the NLTK library. |
| S3 — Punctuation removed | `imdb62_C.csv` | Symbols like `! ? . , ' "` removed. |
| S4 — Both removed | `imdb62_D.csv` | Both stopwords and punctuation removed together. |
| S5 — Frequency discounting | `imdb62_A.csv` (reused) | No words removed. The **same full-text file as S1** is used, but the word-count numbers are mathematically reduced using a square-root formula at runtime inside the model code. No new dataset file is needed. |

> **Why only 4 dataset files for 5 scenarios?**
> Scenarios S1–S4 each require a differently pre-processed version of the text, so each
> has its own file. Scenario S5 (frequency discounting) does **not** change or remove any
> words — it only adjusts the numerical weights assigned to words during model training.
> It therefore reuses the full-text file (`imdb62_A.csv`) and applies the square-root
> transformation on the fly inside the pipeline code.

### How We Measured Performance

Each scenario is evaluated using **10-fold cross-validation**: the dataset is split into
10 equal chunks; the model trains on 9 chunks and tests on the remaining 1, repeating this
10 times so every review gets tested exactly once.

We measure **Accuracy**, **Precision**, **Recall**, and **F1-Score** (all macro-averaged
across the three sentiment classes). To check whether differences between scenarios are
statistically real (not just random variation), we apply the **Friedman test** followed
by **Holm-corrected Wilcoxon signed-rank tests** — a standard statistical framework for
comparing multiple conditions.

### Key Findings (Summary)

- Removing stopwords or punctuation **consistently hurts** LinearSVC. The performance drop
  is statistically significant.
- **Frequency discounting performs as well as keeping everything**, and is significantly
  better than any deletion strategy.
- ComplementNB is more robust — the differences between scenarios are smaller — but
  frequency discounting is still the best-performing condition for that model too.
- **Conclusion:** Do not blindly remove stopwords and punctuation. Frequency discounting
  is a safer, better-performing alternative.

---

## Folder Structure

```
TA-Submission/
│
├── README.md                         ← This file
├── requirements.txt                  ← List of Python packages needed
│
├── Dataset/
│   ├── imdb62.csv                    ← Raw, unmodified IMDB62 dataset (source file)
│   │                                    Used as input by dataset_preprocessing.ipynb
│   ├── imdb62_A.csv                  ← S1 & S5: Full text (all tokens retained)
│   │                                    Note: S5 reuses this file — no words are removed
│   │                                    for frequency discounting; the weighting is applied
│   │                                    inside the model code at runtime.
│   ├── imdb62_B.csv                  ← S2: Stopwords removed
│   ├── imdb62_C.csv                  ← S3: Punctuation removed
│   └── imdb62_D.csv                  ← S4: Both stopwords and punctuation removed
│                                        (4 derived files cover 5 scenarios — see explanation above)
│
├── code/
│   ├── dataset_preprocessing.ipynb   ← Jupyter notebook: creates the 4 datasets above
│   │                                    from the raw imdb62.csv file
│   ├── crossvalidation_pipeline.py   ← Shared engine: handles cross-validation and
│   │                                    statistical testing (used by both models below)
│   ├── linearsvc_pipeline.py         ← Runs the LinearSVC model across all 5 scenarios
│   └── complementnb_pipeline.py      ← Runs the ComplementNB model across all 5 scenarios
│
├── results/
│   ├── linearsvc_results_summary.csv      ← LinearSVC: average scores per scenario
│   ├── linearsvc_results_per_fold.csv     ← LinearSVC: scores for each of the 10 folds
│   ├── linearsvc_statistical_tests.csv    ← LinearSVC: statistical significance results
│   ├── complementnb_results_summary.csv   ← ComplementNB: average scores per scenario
│   ├── complementnb_results_per_fold.csv  ← ComplementNB: scores for each of the 10 folds
│   └── complementnb_statistical_tests.csv ← ComplementNB: statistical significance results
│
└── AdditionalMaterials/
    │
    ├── Final Research Paper/
    │   └── TA_Group2_Paper_Frequency.pdf       ← Final submitted research paper (PDF)
    │
    ├── AccountantDocs-Parth/                   ← Weekly time-tracking reports (Parth Deshmukh, Accountant)
    │   ├── WEEK_2_REPORT_ACCOUNTANT_GROUP2.pdf
    │   ├── WEEK_3_REPORT_ACCOUNTANT_GROUP2.pdf
    │   ├── WEEK_4_REPORT_ACCOUNTANT_GROUP2.pdf
    │   ├── WEEK_5_REPORT_ACCOUNTANT_GROUP2.pdf
    │   ├── WEEK_8_REPORT_ACCOUNTANT_GROUP2.pdf
    │   ├── WEEK_9_REPORT_ACCOUNTANT_GROUP2.pdf
    │   ├── WEEK_10_REPORT_ACCOUNTANT_GROUP2.pdf
    │   ├── WEEK_11_REPORT_ACCOUNTANT_GROUP2.pdf
    │   └── WEEK_12_REPORT_ACCOUNTANT_GROUP2.pdf
    │
    ├── ChairMeetingAgendas-Drishya/            ← Meeting agendas for all group meetings (Drishya Dinesh, Chair)
    │   ├── Group 2 Meeting Agenda - Week 2.docx
    │   ├── Group 2 Meeting Agenda - Week 3.docx
    │   ├── Group 2 Meeting Agenda - Week 4 Meeting 1.docx
    │   ├── Group 2 Meeting Agenda - Week 4 Meeting 2.docx
    │   ├── Group 2 Meeting Agenda - Week 5 Meeting 1.docx
    │   ├── Group 2 Meeting Agenda - Week 6 Meeting 1.docx
    │   ├── Group 2 Meeting Agenda - Week 8 Meeting 1.docx
    │   ├── Group 2 Meeting Agenda - Week 9 Meeting 1.docx
    │   └── Group 2 Meeting Agenda - Week 10 Meeting 1.docx
    │
    ├── MonitorSheet-Mukul/                     ← Article summary tracking sheet (Mukul Ghare, Monitor)
    │   └── TA Monitor Sheet.xlsx
    │
    ├── AmbassadorReport-Yilin/                 ← Inter-group communication report (Yilin Wen, Ambassador)
    │   └── WEN-YILIN-25346690-CS7IS4-GROUP2-AMBASSADOR.docx
    │
    └── VerifierReport-Tanmay/                  ← Weekly verification records (Tanmay Ghosh, Verifier)
        └── Verifier's Report - Tanmay.pdf
```

> **The `results/` folder already contains our pre-computed outputs.**
> You only need to re-run the code if you want to independently reproduce the numbers.

---

## Additional Materials

The `AdditionalMaterials/` folder contains the supplementary group process documents
submitted as part of the replicability archive (requirement (d) per module guidelines).
These are organised by role:

| Folder | Role | Contents |
|--------|------|----------|
| `Final Research Paper/` | — | `TA_Group2_Paper_Frequency.pdf` — the final submitted research paper |
| `AccountantDocs-Parth/` | Accountant (Parth Deshmukh) | 9 weekly time-tracking reports (Weeks 2–5, 8–12) recording each member's hours contributed per week |
| `ChairMeetingAgendas-Drishya/` | Chair (Drishya Dinesh) | 9 meeting agendas covering all group meetings from Week 2 through Week 10 |
| `MonitorSheet-Mukul/` | Monitor (Mukul Ghare) | Spreadsheet tracking which research article each group member read and summarised each week |
| `AmbassadorReport-Yilin/` | Ambassador (Yilin Wen) | Signed report documenting ideas exchanged with other groups |
| `VerifierReport-Tanmay/` | Verifier (Tanmay Ghosh) | Report verifying that weekly responsibilities of all role-holders were met throughout the project |

### Raw Dataset

`Dataset/imdb62.csv` is the raw, unmodified source file downloaded from
[HuggingFace tasksource/imdb62](https://huggingface.co/datasets/tasksource/imdb62).
It is included here so the preprocessing step (Step A) can be reproduced without
needing an external download. The four derived files (`imdb62_A/B/C/D.csv`) were
produced from this file by `code/dataset_preprocessing.ipynb`.

---

## Setup: Getting Your Computer Ready

Follow these steps in order. You only need to do this once.

### Step 1 — Check that Python is installed

Open a terminal (on Mac: search for "Terminal"; on Windows: search for "Command Prompt")
and type:

```
python3 --version
```

You should see something like `Python 3.10.x`. If you get an error, install Python from
https://www.python.org/downloads/ (version 3.9 or higher).

### Step 2 — Navigate to the submission folder

In the terminal, type `cd` followed by the path to the `TA-Submission` folder.
For example:

```
cd /path/to/TA-Submission
```

All commands from this point onwards assume you are inside `TA-Submission/`.

### Step 3 — Create a virtual environment

A virtual environment is an isolated space for this project's packages, so they do not
interfere with anything else on your computer.

```
python3 -m venv venv
```

### Step 4 — Activate the virtual environment

**Mac / Linux:**
```
source venv/bin/activate
```

**Windows:**
```
venv\Scripts\activate
```

Your terminal prompt should now start with `(venv)` to show the environment is active.
You must do this every time you open a new terminal window and want to run the code.

### Step 5 — Install required packages

```
pip install --upgrade pip
pip install -r requirements.txt
```

This installs everything needed: pandas (data handling), scikit-learn (machine learning
models), scipy (statistical tests), nltk (natural language processing), and jupyter
(for running the preprocessing notebook).

### Step 6 — Download NLTK language data

This is only needed if you plan to re-run the preprocessing notebook (Step A below).
If you just want to run the models on the data already provided, you can skip this.

```
python3 -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
"
```

---

## Running the Code

There are three independent things you can run. You do not need to run them in order —
the datasets and results are already provided. Run only what you need.

---

### Step A — (Optional) Re-create the preprocessed datasets

> **Skip this if you just want to run the models.** The four files in `Dataset/` are
> already provided and ready to use.

This step re-creates `imdb62_A/B/C/D.csv` from the original raw IMDB62 file.

The raw file `imdb62.csv` is already included in the `Dataset/` folder, so no
download is needed. Launch Jupyter directly from the `code/` directory:

```
cd code
jupyter notebook dataset_preprocessing.ipynb
```

A browser window will open. Click **Kernel → Restart & Run All** to run the whole
notebook from top to bottom.

When it finishes, four new CSV files will appear in `Dataset/`, one for each
preprocessing scenario (A = all retained, B = stopwords removed,
C = punctuation removed, D = both removed).

Go back to `TA-Submission/` when done:
```
cd ..
```

---

### Step B — Run the LinearSVC model

This trains and evaluates the Linear Support Vector Machine across all five scenarios
using 10-fold cross-validation, then runs the statistical significance tests.

```
python3 code/linearsvc_pipeline.py
```

**What to expect:** You will see progress printed for each fold of each scenario (50 lines
of output total). The full run takes roughly **15–25 minutes** on a standard laptop.

When it finishes, three files are written to `results/`:

| File | What it contains |
|------|-----------------|
| `linearsvc_results_summary.csv` | Average Accuracy, Precision, Recall, F1 ± standard deviation for each of the 5 scenarios |
| `linearsvc_results_per_fold.csv` | The individual score for every one of the 50 fold runs |
| `linearsvc_statistical_tests.csv` | Whether each pair of scenarios is statistically significantly different |

---

### Step C — Run the ComplementNB model

This does the same thing as Step B but uses the Complement Naive Bayes classifier.

```
python3 code/complementnb_pipeline.py
```

**What to expect:** Much faster than LinearSVC — roughly **3–5 minutes** on a standard
laptop.

When it finishes, three files are written to `results/`:

| File | What it contains |
|------|-----------------|
| `complementnb_results_summary.csv` | Average scores per scenario |
| `complementnb_results_per_fold.csv` | Individual fold-level scores |
| `complementnb_statistical_tests.csv` | Pairwise statistical significance results |

---

## Understanding the Result Files

### Summary files (`*_results_summary.csv`)

One row per preprocessing scenario. Columns:

```
Scenario, accuracy_mean, accuracy_std,
          precision_mean, precision_std,
          recall_mean, recall_std,
          f1_mean, f1_std
```

- **mean** = average of the metric across all 10 folds
- **std** = how much the result varied across the 10 folds (smaller is more stable)
- All metrics are **macro-averaged**: each of the three sentiment classes
  (Negative, Neutral, Positive) is weighted equally, regardless of how many reviews
  belong to each class.

### Per-fold files (`*_results_per_fold.csv`)

One row per fold per scenario (50 rows total). Columns:

```
Scenario, Fold, Accuracy, Precision, Recall, F1
```

These are the raw measurements that go into all statistical tests.

### Statistical test files (`*_statistical_tests.csv`)

One row per pair of scenarios compared (10 pairs total). Columns:

```
Comparison, Raw_p, Holm_p, Significant
```

- **Raw_p** — the raw p-value from the Wilcoxon signed-rank test
- **Holm_p** — the p-value after Holm correction (adjusts for the fact that we are
  running 10 tests at once, which increases the chance of a false positive)
- **Significant** — ✔ Yes means the two scenarios perform significantly differently
  (Holm_p < 0.05); ✘ No means any observed difference could be random chance

The Friedman test result (an overall test asking "are any of the 5 scenarios different
from each other?") is printed to the terminal when the scripts run.

---

## Code File Reference

### `dataset_preprocessing.ipynb`

A step-by-step Jupyter notebook that takes the raw `imdb62.csv` file and produces four
cleaned variants. Each section of the notebook applies a different combination of
stopword removal and punctuation removal, followed by:
- Removing non-standard characters (emojis, etc.)
- Converting all text to lowercase
- Splitting text into individual word tokens
- Lemmatisation (reducing each word to its base form, e.g. *running* → *run*)

### `crossvalidation_pipeline.py`

The shared engine used by both model pipelines. You do not need to modify or run this
file directly. It provides two functions:

- **`run_all_scenarios_cv`** — takes a list of scenarios and a model, runs 10-fold
  cross-validation for each, and returns per-fold and summary results
- **`run_statistical_tests`** — takes those results and runs the Friedman test followed
  by Holm-corrected Wilcoxon pairwise tests

### `linearsvc_pipeline.py`

The entry point for the LinearSVC experiment. Run this with `python3 code/linearsvc_pipeline.py`.
It loads the four dataset files, defines the five scenarios, calls the cross-validation
engine, and saves the three result files.

The model itself is defined in one line:
```python
LinearSVC(random_state=42, max_iter=2000, dual=True, class_weight='balanced')
```
- `class_weight='balanced'` corrects for the fact that positive reviews outnumber
  negative ones — the model is penalised more for misclassifying the rarer classes.
- `random_state=42` ensures the results are reproducible.

### `complementnb_pipeline.py`

The entry point for the ComplementNB experiment. Identical structure to the LinearSVC
pipeline. Run with `python3 code/complementnb_pipeline.py`.

The model:
```python
ComplementNB()
```
ComplementNB corrects for class imbalance automatically through its mathematical
formulation (it learns from the complement of each class rather than the class itself),
so no extra parameter is needed.
