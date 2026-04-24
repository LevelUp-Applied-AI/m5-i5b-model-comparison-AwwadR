import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


MODEL_REGISTRY = {
    "DummyClassifier": DummyClassifier,
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}


class ModelSelector:
    """Config-driven model selection framework for churn modeling."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = self.load_config(config_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = self.config["output"]["root_dir"]
        self.output_dir = os.path.join(root_dir, f"experiment_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.models: dict[str, Pipeline] = {}
        self.results_df: pd.DataFrame | None = None
        self.fitted_models: dict[str, Pipeline] = {}

    def load_config(self, config_path: str) -> dict:
        """Load JSON configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_data(self):
        """Load dataset and create train/test split from config."""
        dataset_cfg = self.config["dataset"]

        filepath = dataset_cfg["filepath"]
        target = dataset_cfg["target"]
        features = dataset_cfg["features"]
        test_size = dataset_cfg.get("test_size", 0.2)
        random_state = dataset_cfg.get("random_state", 42)
        stratify_flag = dataset_cfg.get("stratify", True)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")

        df = pd.read_csv(filepath)

        missing_features = [col for col in features if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in dataset: {missing_features}")

        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify_flag else None,
        )

        return X_train, X_test, y_train, y_test

    def get_scaler(self, scaler_choice: str):
        """Return scaler object from config choice."""
        if scaler_choice == "standard":
            return StandardScaler()
        if scaler_choice == "passthrough":
            return "passthrough"
        raise ValueError(
            f"Unsupported scaler choice: {scaler_choice}. "
            "Use 'standard' or 'passthrough'."
        )

    def build_models(self) -> None:
        """Build model pipelines from configuration."""
        self.models = {}

        for model_cfg in self.config["models"]:
            model_name = model_cfg["name"]
            model_type = model_cfg["type"]
            scaler_choice = model_cfg.get("scaler", "passthrough")
            params = model_cfg.get("params", {})
            param_grid = model_cfg.get("param_grid")

            if model_type not in MODEL_REGISTRY:
                raise ValueError(
                    f"Unsupported model type: {model_type}. "
                    f"Supported types: {list(MODEL_REGISTRY.keys())}"
                )

            model_class = MODEL_REGISTRY[model_type]

            if param_grid is not None:
                combos = list(ParameterGrid(param_grid))
                for i, combo in enumerate(combos, start=1):
                    full_name = f"{model_name}_{i}"
                    scaler = self.get_scaler(scaler_choice)
                    model = model_class(**combo)
                    pipeline = Pipeline([
                        ("scaler", scaler),
                        ("model", model),
                    ])
                    self.models[full_name] = pipeline
            else:
                scaler = self.get_scaler(scaler_choice)
                model = model_class(**params)
                pipeline = Pipeline([
                    ("scaler", scaler),
                    ("model", model),
                ])
                self.models[model_name] = pipeline

    def run_cv_comparison(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Run stratified CV comparison on all configured models."""
        cv_cfg = self.config["cv"]
        n_splits = cv_cfg.get("n_splits", 5)
        shuffle = cv_cfg.get("shuffle", True)
        random_state = cv_cfg.get("random_state", 42)

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

        rows = []

        for name, pipeline in self.models.items():
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            pr_auc_scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                model_copy = clone(pipeline)
                model_copy.fit(X_train_fold, y_train_fold)

                y_pred = model_copy.predict(X_val_fold)

                if not hasattr(model_copy, "predict_proba"):
                    raise ValueError(
                        f"Model '{name}' does not support predict_proba, "
                        "which is required for PR-AUC."
                    )

                y_proba = model_copy.predict_proba(X_val_fold)[:, 1]

                accuracy_scores.append(accuracy_score(y_val_fold, y_pred))
                precision_scores.append(
                    precision_score(y_val_fold, y_pred, zero_division=0)
                )
                recall_scores.append(
                    recall_score(y_val_fold, y_pred, zero_division=0)
                )
                f1_scores.append(
                    f1_score(y_val_fold, y_pred, zero_division=0)
                )
                pr_auc_scores.append(
                    average_precision_score(y_val_fold, y_proba)
                )

            rows.append({
                "model": name,
                "accuracy_mean": np.mean(accuracy_scores),
                "accuracy_std": np.std(accuracy_scores),
                "precision_mean": np.mean(precision_scores),
                "precision_std": np.std(precision_scores),
                "recall_mean": np.mean(recall_scores),
                "recall_std": np.std(recall_scores),
                "f1_mean": np.mean(f1_scores),
                "f1_std": np.std(f1_scores),
                "pr_auc_mean": np.mean(pr_auc_scores),
                "pr_auc_std": np.std(pr_auc_scores),
            })

        self.results_df = pd.DataFrame(rows)
        return self.results_df

    def save_comparison_table(self) -> None:
        """Save comparison results CSV."""
        if self.results_df is None:
            raise ValueError("No results available. Run CV comparison first.")

        path = os.path.join(self.output_dir, "comparison_table.csv")
        self.results_df.to_csv(path, index=False)

    def fit_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit all models on the full training set."""
        self.fitted_models = {}

        for name, pipeline in self.models.items():
            fitted_pipeline = clone(pipeline)
            fitted_pipeline.fit(X_train, y_train)
            self.fitted_models[name] = fitted_pipeline

    def get_top3_models(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Return top 3 fitted models by test-set PR-AUC."""
        scores = []

        for name, model in self.fitted_models.items():
            y_proba = model.predict_proba(X_test)[:, 1]
            pr_auc = average_precision_score(y_test, y_proba)
            scores.append((name, pr_auc))

        top3 = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
        return top3

    def plot_pr_curves_top3(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Save PR curves for top 3 models."""
        top3 = self.get_top3_models(X_test, y_test)

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, _ in top3:
            PrecisionRecallDisplay.from_estimator(
                self.fitted_models[name],
                X_test,
                y_test,
                ax=ax,
                name=name,
            )

        ax.set_title("Precision-Recall Curves (Top 3 Models)")
        path = os.path.join(self.output_dir, "pr_curves.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def plot_calibration_top3(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Save calibration curves for top 3 models."""
        top3 = self.get_top3_models(X_test, y_test)

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, _ in top3:
            CalibrationDisplay.from_estimator(
                self.fitted_models[name],
                X_test,
                y_test,
                n_bins=10,
                ax=ax,
                name=name,
            )

        ax.set_title("Calibration Curves (Top 3 Models)")
        path = os.path.join(self.output_dir, "calibration.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    def save_experiment_log(self) -> None:
        """Save experiment log CSV."""
        if self.results_df is None:
            raise ValueError("No results available. Run CV comparison first.")

        timestamp = datetime.now().isoformat()
        log_df = pd.DataFrame({
            "model_name": self.results_df["model"],
            "accuracy": self.results_df["accuracy_mean"],
            "precision": self.results_df["precision_mean"],
            "recall": self.results_df["recall_mean"],
            "f1": self.results_df["f1_mean"],
            "pr_auc": self.results_df["pr_auc_mean"],
            "timestamp": timestamp,
        })

        path = os.path.join(self.output_dir, "experiment_log.csv")
        log_df.to_csv(path, index=False)

    def save_best_model(self) -> str:
        """Save best fitted model by PR-AUC."""
        if self.results_df is None:
            raise ValueError("No results available. Run CV comparison first.")
        if not self.fitted_models:
            raise ValueError("No fitted models available. Fit models first.")

        best_name = (
            self.results_df.sort_values("pr_auc_mean", ascending=False)
            .iloc[0]["model"]
        )
        best_model = self.fitted_models[best_name]

        path = os.path.join(self.output_dir, "best_model.joblib")
        dump(best_model, path)

        return best_name

    def save_run_metadata(self) -> None:
        """Save the config used for this experiment."""
        path = os.path.join(self.output_dir, "config_used.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2)

    def run(self) -> None:
        """Run the full configurable model selection pipeline."""
        X_train, X_test, y_train, y_test = self.load_data()
        self.build_models()

        print(
            f"Loaded data: {len(X_train)} train rows, {len(X_test)} test rows, "
            f"train churn rate = {y_train.mean():.2%}"
        )
        print(f"\nBuilt {len(self.models)} model pipelines:")
        print(list(self.models.keys()))

        self.run_cv_comparison(X_train, y_train)
        self.save_comparison_table()

        self.fit_all_models(X_train, y_train)
        self.plot_pr_curves_top3(X_test, y_test)
        self.plot_calibration_top3(X_test, y_test)
        self.save_experiment_log()
        best_name = self.save_best_model()
        self.save_run_metadata()

        print("\n=== Model Comparison Table ===")
        print(self.results_df.to_string(index=False))
        print(f"\nBest model by PR-AUC: {best_name}")
        print(f"Outputs saved to: {self.output_dir}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python model_selector.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        selector = ModelSelector(config_path)
        selector.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()