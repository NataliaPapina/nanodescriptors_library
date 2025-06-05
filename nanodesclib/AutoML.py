import optuna
import shap
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import SelectFromModel


class AutoMLPipeline(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            task='regression',
            pre_filter=True,
            correlation_threshold=0.95,
            var_threshold=1e-5,
            max_features=None,
            feature_selection_method="shap",
            n_trials=50,
            scale_data=True,
            random_state=42
    ):
        self.task = task
        self.n_trials = n_trials
        self.pre_filter = pre_filter
        self.correlation_threshold = correlation_threshold
        self.var_threshold = var_threshold
        self.max_features = max_features
        self.feature_selection_method = feature_selection_method.lower()
        self.random_state = random_state
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        self.selected_features = None

    @staticmethod
    def clean_feature_names(df):
        df.columns = (
            df.columns
            .str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
            .str.replace(r'__+', '_', regex=True)
            .str.strip('_')
        )
        return df

    def _pre_filter(self, X):
        X_filtered = X.copy()

        selector = VarianceThreshold(threshold=self.var_threshold)
        X_filtered = pd.DataFrame(selector.fit_transform(X_filtered), columns=X.columns[selector.get_support()])

        corr_matrix = X_filtered.corr(method="spearman").abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        X_filtered = X_filtered.drop(columns=to_drop)

        return X_filtered

    def _scale(self, X, fit=False):
        if not self.scale_data:
            return X
        if fit:
            self.scaler.fit(X)
        return self.scaler.transform(X)

    def _objective(self, trial, X, y):
        model_name = trial.suggest_categorical("model", ["xgboost", "lightgbm", "random_forest", "lasso", "ridge"])
        model = self._create_model(model_name, trial)
        pipeline = Pipeline([
            ("regressor", model)
        ])
        X_scaled = self._scale(X, fit=True)
        score = cross_val_score(pipeline, X_scaled, y, scoring="neg_root_mean_squared_error", cv=3).mean()
        return score

    def _create_model(self, name, trial_or_params):
        if isinstance(trial_or_params, dict):
            p = trial_or_params
        else:
            trial = trial_or_params
            name = trial.suggest_categorical("model", ["xgboost", "lightgbm", "random_forest", "lasso", "ridge"])
            p = {}

            if name in ["xgboost", "lightgbm"]:
                p["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                p["max_depth"] = trial.suggest_int("max_depth", 3, 10)
                p["learning_rate"] = trial.suggest_float("lr", 1e-3, 0.3, log=True)
            elif name == "random_forest":
                p["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                p["max_depth"] = trial.suggest_int("max_depth", 3, 15)
            elif name in ["lasso", "ridge"]:
                p["alpha"] = trial.suggest_float("alpha", 1e-4, 10.0, log=True)

        if name == "xgboost":
            return XGBRegressor(**p, random_state=self.random_state, verbosity=0)
        elif name == "lightgbm":
            return LGBMRegressor(**p, random_state=self.random_state)
        elif name == "random_forest":
            return RandomForestRegressor(**p, random_state=self.random_state)
        elif name == "lasso":
            return Lasso(**p, random_state=self.random_state)
        elif name == "ridge":
            return Ridge(**p, random_state=self.random_state)

    def fit(self, X, y):

        X = self.clean_feature_names(X)

        if self.pre_filter:
            X = self._pre_filter(X)

        self.feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self._objective(trial, X, y), n_trials=self.n_trials)

        self.best_params = study.best_params
        self.best_model_name = self.best_params.pop("model")
        self.model = self._create_model(self.best_model_name, self.best_params)

        X_scaled = self._scale(X, fit=True)
        self.model.fit(X_scaled, y)

        try:
            if self.feature_selection_method == "shap":
                print("[INFO] Using SHAP for feature selection...")
                if hasattr(self.model, "predict_proba") or isinstance(self.model, (
                XGBRegressor, LGBMRegressor, RandomForestRegressor)):
                    explainer = shap.Explainer(self.model, X_scaled)
                else:
                    explainer = shap.KernelExplainer(self.model.predict, shap.sample(X_scaled, 100))

                shap_values = np.abs(explainer(X_scaled).values).mean(axis=0)
                importance = pd.Series(shap_values, index=list(self.feature_names)).sort_values(ascending=False)

                if self.max_features:
                    self.selected_features = importance.iloc[:self.max_features].index.tolist()
                else:
                    self.selected_features = importance[importance > 0].index.tolist()

                selected_idxs = [list(self.feature_names).index(f) for f in self.selected_features]
                self.model.fit(X_scaled[:, selected_idxs], y)

                self._shap_values = shap_values
                self._explainer = explainer

            elif self.feature_selection_method == "model":
                print("[INFO] Using SelectFromModel for feature selection...")
                from sklearn.feature_selection import SelectFromModel
                selector = SelectFromModel(self.model, threshold="median")
                selector.fit(X_scaled, y)
                mask = selector.get_support()
                self.selected_features = [f for f, m in zip(self.feature_names, mask) if m]
                selected_idxs = [list(self.feature_names).index(f) for f in self.selected_features]
                self.model.fit(X_scaled[:, selected_idxs], y)

            else:
                print(f"[WARN] Unknown feature_selection_method: {self.feature_selection_method}. Using all features.")
                self.selected_features = list(self.feature_names)
                self.model.fit(X_scaled, y)


        except Exception as e:
            print(f"[WARN] SHAP fallback: {e}")
            self.selected_features = self.feature_names
            self.model.fit(X_scaled, y)

        print(f"[INFO] Лучшая модель: {self.best_model_name}")
        print(f"[INFO] Параметры модели: {self.best_params}")

        return self

    def predict(self, X):
        X = self.clean_feature_names(X)

        if hasattr(self, "feature_names"):
            X = X[[col for col in self.feature_names if col in X.columns]]
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[self.feature_names]

        X_scaled = self._scale(X)

        if self.selected_features is not None and len(self.selected_features) > 0:
            idxs = self.feature_names.get_indexer(self.selected_features)
            return self.model.predict(np.asarray(X_scaled)[:, idxs])
        return self.model.predict(np.asarray(X_scaled))

    def score(self, X, y):
        preds = self.predict(X)
        return {
            "R2": r2_score(y, preds),
            "MAE": mean_absolute_error(y, preds),
            "RMSE": mean_squared_error(y, preds, squared=False),
            "MSE": mean_squared_error(y, preds)
        }

    def get_important_features(self):
        return self.selected_features

    def plot_shap_summary(self, X):
        if hasattr(self, '_explainer'):
            X_scaled = self._scale(X)
            shap_values = self._explainer(X_scaled)
            shap.summary_plot(shap_values, X, feature_names=self.feature_names)
        else:
            print("[WARN] SHAP summary not available.")
