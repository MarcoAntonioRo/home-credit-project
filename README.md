# Home Credit Default Risk

Marco Rodriguez · IS 6812 MSBA Capstone 1 · University of Utah

## Project overview

Home Credit lends to customers with little or no formal credit history. This project builds a repayment-risk score from data the company already holds, with the goal of reducing defaults among approved loans while approving more creditworthy thin-file applicants. The data comes from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) competition on Kaggle.

## Repository contents

| File | Purpose |
|------|---------|
| [`data_preparation.py`](data_preparation.py) | Reusable cleaning and feature engineering functions (described below) |
| [`notebooks/eda.qmd`](notebooks/eda.qmd) | Exploratory data analysis notebook (Quarto source) |
| [`notebooks/styles.css`](notebooks/styles.css) | Stylesheet used when rendering the EDA notebook to HTML |
| [`docs/feature_report.md`](docs/feature_report.md) | Research on the risk features lenders use, mapped to the data |
| `.gitignore` | Keeps data files and generated outputs out of the repository |

Data files are not committed. Download them from Kaggle into `data/raw/` (see [How to run](#how-to-run)).

---

## Data preparation script

### What it does

[`data_preparation.py`](data_preparation.py) turns the raw application tables into a model-ready dataset. It implements the data preparation decisions recorded in the Results section of the [EDA notebook](notebooks/eda.qmd). The script is organized as reusable functions that apply identical transformations to the training and test data.

The script runs in five steps:

1. **Clean** (`clean_application`): recode placeholders, undocumented categories and extreme values. No rows are ever removed.
2. **Flag missing data** (`add_missing_indicators`): record which values were missing *before* any blanks are filled, because missingness predicts default.
3. **Engineer features** (`add_engineered_features`): unit conversions, financial ratios, summary scores, bins and interaction terms.
4. **Impute and encode** (`impute_and_encode`): fill blanks with training medians and one-hot encode categories using training levels.
5. **Check** (`check_consistency`): verify the train and test outputs match.

### How each transformation maps to the EDA

**Cleaning**

| Transformation | EDA finding and decision |
|---|---|
| `DAYS_EMPLOYED = 365243` set to missing, plus flag `FLAG_EMPLOYED_PLACEHOLDER` | Placeholder on 55,374 rows, 99.96% pensioners, who default at only 5.4%. The flag keeps that low-risk signal. |
| `OWN_CAR_AGE` of 64 or 65 set to missing, plus flag `FLAG_CAR_AGE_PLACEHOLDER` | Spike of 3,334 cars "64 to 65 years old" owned by applicants with a median age of 37: a second placeholder. |
| `XNA` gender and `Unknown` family status set to missing | Levels not documented in the data dictionary. |
| Region rating of `-1` set to missing | Dictionary allows only 1, 2 and 3; one test row has -1. |
| Category levels seen fewer than 30 times in train set to missing | Train-only levels (e.g. `Maternity leave`, 5 rows) would create columns that exist in only one file. |
| Income, bureau inquiries, social circle counts, children and family size capped at the train 99.9th percentile | Entry error of 117,000,000 income; 261 inquiries in a quarter; social circle count of 348. |
| `FLAG_MOBIL` and 9 document flags dropped | Constant or nearly constant columns carry no information. |

**Engineered features**

| Feature(s) | EDA finding and decision |
|---|---|
| `AGE_YEARS`, `EMPLOYED_YEARS`, `YEARS_SINCE_REGISTRATION`, `YEARS_SINCE_ID_PUBLISH`, `YEARS_SINCE_PHONE_CHANGE` | Raw `DAYS_*` columns are negative day counts; converted to positive years. |
| `FLAG_PHONE_CHANGED_ON_APP_DAY` | Q9: 12% of applicants changed phone on application day and default at 9.7%; the zero behaves as its own category. |
| `PAYMENT_TO_INCOME` | Q4: the central ability-to-repay ratio. Kept as a candidate even though income looks self-reported. |
| `CREDIT_TO_INCOME` | Standard companion to payment-to-income: total debt relative to income. |
| `LOAN_TO_VALUE` and `LTV_BAND` | Q5: loans more than 30% above the goods price default at 12.8%, about twice the rate at the goods price. Bands keep the non-linear jump. |
| `CREDIT_TO_ANNUITY` | Q6: approximates loan term and carries the lender's past pricing. |
| `EMPLOYED_TO_AGE` | Q7: tenure is a strong signal; scaling by age separates long careers from young workers. |
| `EXT_SOURCE_MEAN` | Q3: the mean of available external scores (AUC 0.72) beats any single score. |
| `TOTAL_BUREAU_INQUIRIES` (hour, day and week counts dropped) | Q8: individual counts are weak and mostly zero; one total is kept. |
| `AGE_BAND` | Q11: default falls steadily with age; bands use fixed cutoffs. |
| `INCOME_QUINTILE` | Income groups with cutoffs learned from train. |
| `EXT_MEAN_X_THIN_FILE` (interaction) | Q2 and Q3: external scores may carry different weight when no bureau file exists. |
| `PTI_X_PENSIONER` (interaction) | Q4 and Q7: payment burden weighs differently on a pension than on a wage. |

**Missing-data indicators** (created before imputation)

| Feature | EDA finding and decision |
|---|---|
| `FLAG_THIN_FILE` | Q2: 13.5% of applicants have no bureau inquiry record and default at 10.3% vs 7.7%. |
| `EXT_SOURCE_1/2/3_MISSING`, `EXT_SOURCE_MISSING_COUNT` | Q3: default rises with each missing external score. |
| `BUILDING_INFO_MISSING_SHARE` | The 47 building columns go missing as a block; applicants with none default at 9.3% vs 7.0%. One share replaces 47 flags. |
| `FLAG_OCCUPATION_MISSING` | Missing occupation marks a *safer* group (57% pensioners, 6.5% default). |
| `FLAG_GOODS_PRICE_MISSING` | Goods price is missing only for revolving loans. |

**Redundant columns removed.** The building block records 14 measures three ways (`_AVG`, `_MODE`, `_MEDI`). Only the `_AVG` versions are kept, plus `TOTALAREA_MODE` (which has no `_AVG` version) and the four text `_MODE` categories.

### EDA decisions changed or not implemented

| EDA decision | What the script does instead | Reason |
|---|---|---|
| Drop the row with 117,000,000 income | Caps income at the train 99.9th percentile (900,000) | Capping fixes the error without removing a row, and handles any future extreme income the same way. |
| Drop the 11 document flags never used in test | Drops columns that are near-constant **in train** (10 columns) | Choosing columns by inspecting test would leak test information into preparation. |
| Exclude `CODE_GENDER` from the model | Kept in the prepared data | Needed later to test the model for disparate impact. **The modeling stage must exclude the `CODE_GENDER_*` columns from model features.** |
| Process supplementary tables (bureau, previous applications, installments) | Not implemented yet | Optional at this stage. `FLAG_THIN_FILE` is a proxy for a missing bureau file and will be confirmed against `bureau.csv` coverage when that table is added. |
| Validate performance on cash loans separately; request verified income, loss given default and margin | Not part of data preparation | These belong to the modeling and business stages. |

### Train/test consistency

Every value that depends on the data is learned **once, from the training data only**, by `fit_preparation()`. It is saved to `prep_params.json` and reused unchanged by `transform()` on both files. `transform()` never calculates a statistic from the data it is given.

| Learned from train | Reused on test for |
|---|---|
| 99.9th-percentile caps for 8 columns (e.g. income cap of 900,000) | Capping extreme values |
| Allowed category levels (seen 30+ times) | Mapping rare or unseen levels to missing |
| Near-constant columns to drop | Dropping the same 10 columns |
| Income quintile cutoffs (99,000; 135,000; 162,000; 225,000) | Assigning `INCOME_QUINTILE` |
| Medians for all 83 numeric features | Filling blanks |
| Category level lists for one-hot encoding | Creating identical dummy columns |
| Final column order | Returning columns in the same order |

**Check output** from `check_consistency()` on the full data:

```
Consistency checks:
  identical_columns_except_target: True
  one_row_per_id_train: True
  one_row_per_id_test: True
  no_rows_dropped_train: True
  no_rows_dropped_test: True
  no_missing_values_train: True
  no_missing_values_test: True
  target_only_in_train: True
  train_shape: (307511, 254)
  test_shape: (48744, 253)
```

In addition, transforming 500 test rows on their own produces exactly the same values as transforming the full test file. This confirms that `transform()` uses only the parameters learned from train.

### How to run

**Requirements:** Python 3.10 or later, with `pandas`, `numpy` and `pyarrow` (all included in Anaconda).

**1. Get the data.** Download from the [Kaggle competition data page](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place the files here:

```
home-credit-project/
├── data/
│   └── raw/
│       ├── application_train.csv
│       └── application_test.csv
└── data_preparation.py
```

**2. Run from the project root:**

```bash
python data_preparation.py
```

Custom paths are optional:

```bash
python data_preparation.py --train data/raw/application_train.csv \
    --test data/raw/application_test.csv --out data/processed
```

The full run takes about 25 seconds.

**3. Or call the functions from Python:**

```python
import pandas as pd
from data_preparation import fit_preparation, transform

train = pd.read_csv("data/raw/application_train.csv")
test = pd.read_csv("data/raw/application_test.csv")

params = fit_preparation(train)          # learn from train only
train_ready = transform(train, params)   # apply to train
test_ready = transform(test, params)     # apply the same values to test
```

### Inputs and outputs

| | File | Rows | Columns |
|---|---|---|---|
| **Input** | `data/raw/application_train.csv` | 307,511 | 122 |
| **Input** | `data/raw/application_test.csv` | 48,744 | 121 |
| **Output** | `data/processed/train_prepared.parquet` | 307,511 | 254 (`SK_ID_CURR` + 252 features + `TARGET`) |
| **Output** | `data/processed/test_prepared.parquet` | 48,744 | 253 (`SK_ID_CURR` + 252 features) |
| **Output** | `data/processed/prep_params.json` | n/a | All values learned from train |

All output files are excluded from the repository by `.gitignore`.

### Use of AI

I used Claude (Anthropic) to help write the script, working one stage at a time: the fit/transform structure, cleaning, feature engineering, then consistency checks. At each stage I checked the logic against my EDA decisions and the outputs against the data. The consistency checks earned their place on the first full run: they caught a test-only blank (the recoded `-1` region rating) that the first version could not fill, which led to learning a median for every numeric column.
