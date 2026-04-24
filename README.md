# Integration 5B — Model Comparison & Decision Memo

Module 5 Week B Integration Task for AISPIRE Applied AI & ML Systems.

This repository contains the full base integration task for comparing churn prediction models on the Petra Telecom dataset, along with optional challenge extensions for threshold optimization, permutation importance, and a configurable model selection framework.

## Project Goal

The goal of this project is to compare multiple model families for churn prediction and recommend a model based on both evaluation metrics and business tradeoffs. The workflow includes cross-validation, precision-recall analysis, calibration analysis, experiment logging, model saving, and a decision memo.

---

## Base Integration Task

The core task is implemented in `model_comparison.py`.

### Base workflow

The script completes these 9 required tasks:

1. `load_and_preprocess` — load dataset, select numeric features, and split 80/20 with stratification
2. `define_models` — define 6 sklearn Pipeline model configurations
3. `run_cv_comparison` — run 5-fold stratified cross-validation and compute mean ± std for:
   - accuracy
   - precision
   - recall
   - F1
   - PR-AUC
4. `save_comparison_table` — save the comparison table to CSV
5. `plot_pr_curves_top3` — save precision-recall curves for the top 3 models
6. `plot_calibration_top3` — save calibration curves for the top 3 models
7. `save_best_model` — save the best model using `joblib`
8. `log_experiment` — save experiment results with timestamps
9. `find_tree_vs_linear_disagreement` — identify one test sample where RF and LR disagree most

### Run the base script

```bash
pip install -r requirements.txt
python model_comparison.py
pytest tests/ -v
```

### Base outputs

The base script writes required outputs to the `results/` folder:

- `comparison_table.csv`
- `pr_curves.png`
- `calibration.png`
- `best_model.joblib`
- `experiment_log.csv`
- `tree_vs_linear_disagreement.md`

### Base results summary

Using 5-fold stratified cross-validation, the best base model was:

- **RF_default**
- **PR-AUC = 0.513682**

Other key results:

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy | 0.8364 | 0.0000 | 0.0000 | 0.0000 | 0.1636 |
| LR_default | 0.8542 | 0.7135 | 0.1952 | 0.3032 | 0.4656 |
| LR_balanced | 0.7075 | 0.3209 | 0.7028 | 0.4404 | 0.4630 |
| DT_depth5 | 0.8572 | 0.6444 | 0.2852 | 0.3935 | 0.4496 |
| RF_default | 0.8611 | 0.7341 | 0.2394 | 0.3593 | 0.5137 |
| RF_balanced | 0.8086 | 0.4267 | 0.4720 | 0.4475 | 0.4707 |

---

## Challenge Extensions

### Tier 1 — Threshold Optimization for Deployment

This extension evaluates threshold choices for the recommended model instead of using only the default 0.5 threshold.

It:
- sweeps thresholds from 0.10 to 0.90
- computes precision, recall, F1, and expected alerts
- selects the best threshold under Petra Telecom’s monthly contact capacity

#### Tier 1 result

The recommended deployment threshold was:

- **Threshold = 0.70**
- **Precision = 0.7778**
- **Recall = 0.0476**
- **F1 = 0.0897**
- **Alerts per 1,000 customers = 10**
- **Alerts per 10,000 customers = 100**

This was the best threshold that stayed within the business constraint of contacting at most 150 customers per 10,000.

Tier 1 output files:
- `results/threshold_sweep.csv`
- `results/threshold_sweep.png`

---

### Tier 2 — Permutation Importance and Model Explanation

This extension uses permutation importance as a model-agnostic way to compare which features matter most across the top 3 models.

Top 3 models used:
- `RF_default`
- `RF_balanced`
- `LR_default`

#### Tier 2 result summary

All 3 models agreed on the strongest churn drivers:

- `num_support_calls`
- `contract_months`
- `tenure`

Example permutation importance values:

| Feature | RF_default | RF_balanced | LR_default |
|---|---:|---:|---:|
| num_support_calls | 0.2053 | 0.1920 | 0.1880 |
| contract_months | 0.0977 | 0.1143 | 0.0724 |
| tenure | 0.0624 | 0.0417 | 0.0260 |

This shows strong agreement on the main features, while differences in secondary features suggest tree models and linear models do not use the data in exactly the same way.

Tier 2 output file:
- `results/permutation_importance.png`

---

### Tier 3 — Configurable Model Selection Framework

This extension adds a reusable framework for model selection using config files instead of hardcoded model definitions.

Implemented in:
- `model_selector.py`

Configuration files:
- `configs/base_config.json`
- `configs/gradient_boosting_config.json`

#### What the framework does

The `ModelSelector` class:

- reads a JSON configuration file
- loads the dataset and split settings from config
- builds model pipelines from config
- supports both single models and hyperparameter grids
- runs the full comparison pipeline from configuration alone
- saves outputs into timestamped folders under `experiments/`

#### Supported model families

The framework uses a registry pattern and currently supports:

- `DummyClassifier`
- `LogisticRegression`
- `DecisionTreeClassifier`
- `RandomForestClassifier`
- `GradientBoostingClassifier`

#### How to run Tier 3

```bash
python model_selector.py configs/base_config.json
python model_selector.py configs/gradient_boosting_config.json
```

#### Tier 3 results summary

Using `base_config.json`, the framework reproduced the original base-task results and selected:

- **RF_default**
- **PR-AUC = 0.513682**

Using `gradient_boosting_config.json`, the framework evaluated 9 total model pipelines and selected:

- **GB_1**
- **PR-AUC = 0.542178**

Other Gradient Boosting results:
- **GB_2**: 0.539178
- **GB_3**: 0.531339

This shows the framework can add new model families through configuration and find better-performing models without changing the core pipeline logic.


#### How to add a new model family

To add a new model family, a teammate needs to:

1. import the model class in `model_selector.py`
2. add it to `MODEL_REGISTRY`
3. add a new config entry with:
   - `name`
   - `type`
   - `scaler`
   - `params` or `param_grid`

This keeps the framework flexible and avoids rewriting the main pipeline when new models are added.

---

## License

This repository is provided for educational use only. See [LICENSE](LICENSE) for terms.

You may clone and modify this repository for personal learning and practice, and reference code you wrote here in your professional portfolio. Redistribution outside this course is not permitted.
