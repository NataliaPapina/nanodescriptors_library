import optuna
import shap
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor, \
    ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from nanodesclib.StackingEnsemble import *
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')


class AutoMLPipeline(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            task='regression',
            max_features=None,
            feature_selection_method="shap",
            n_trials=50,
            scale_data=True,
            random_state=0,
            enable_pls=True,
            timeout=1200,
            pre_filter=True,
            correlation_threshold=0.95,
            var_threshold=1e-5,
            warm_start=True,
            coarse_trials_ratio=0.4
    ):
        self.task = task
        self.n_trials = n_trials
        self.max_features = max_features
        self.pre_filter = pre_filter
        self.correlation_threshold = correlation_threshold
        self.var_threshold = var_threshold
        self.feature_selection_method = feature_selection_method.lower()
        self.random_state = random_state
        self.scale_data = scale_data
        self.enable_pls = enable_pls
        self.scaler = StandardScaler()
        self.selected_features = None
        self.fitted_feature_names_ = None
        self.timeout = timeout
        self._timeout_flag = False
        self.cv_feature_selection = True
        self.warm_start = warm_start
        self.coarse_trials_ratio = coarse_trials_ratio

    def _pre_filter(self, X):
        """Фильтрация признаков по дисперсии и корреляции"""
        X_filtered = X.copy()

        numeric_cols = X_filtered.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            self.selector = VarianceThreshold(threshold=self.var_threshold)
            self.selector.fit(X_filtered[numeric_cols])

            X_numeric_filtered = self.selector.transform(X_filtered[numeric_cols])
            kept_numeric_cols = numeric_cols[self.selector.get_support()]

            X_filtered = pd.concat([
                pd.DataFrame(X_numeric_filtered, columns=kept_numeric_cols, index=X_filtered.index),
                X_filtered.drop(columns=numeric_cols)
            ], axis=1)

        numeric_cols_remaining = X_filtered.select_dtypes(include=[np.number]).columns

        if len(numeric_cols_remaining) > 1:
            corr_matrix = X_filtered[numeric_cols_remaining].corr(method="spearman").abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
            self.high_corr_columns_to_drop = to_drop
            X_filtered = X_filtered.drop(columns=to_drop)

        return X_filtered

    def _simple_feature_selection(self, X_train, y_train, feature_names, n_features=20):
        """feature selection на основе RandomForest"""
        print("=== SIMPLE FEATURE SELECTION ===")

        try:
            rf = RandomForestRegressor(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            importances = rf.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            print("\nTop features by importance:")
            for i, (feature, importance) in enumerate(feature_importance[:15]):
                print(f"  {i + 1:2d}. {feature}: {importance:.4f}")

            selected_features = [feature for feature, _ in feature_importance[:n_features]]
            print(f"Selected {len(selected_features)} features")

            return selected_features

        except Exception as e:
            print(f"Simple feature selection failed: {e}")
            return feature_names[:min(n_features, len(feature_names))]

    def _calculate_vip(self, model):
        """Calculate Variable Importance in Projection (VIP) scores for PLS"""
        try:
            t = model.x_scores_
            w = model.x_weights_
            q = model.y_weights_
            p = w.shape[0]
            h = t.shape[1]

            vips = np.zeros((p,))
            s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
            total_s = np.sum(s)

            for i in range(p):
                weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
                vips[i] = np.sqrt(p * (s.T @ weight) / total_s)

            return vips
        except Exception as e:
            w = model.x_weights_
            p = w.shape[0]
            print(f"Error calculating VIP: {e}")
            return np.ones(p)

    def _genetic_algorithm_feature_selection(self, X, y, n_population=20, n_generations=30,
                                             n_components=3, cv_folds=5):
        """Genetic Algorithm for feature selection optimized for PLS"""
        n_features = X.shape[1]

        def fitness(individual):
            """Fitness function based on cross-validated R2"""
            selected_features = np.where(individual)[0]
            if len(selected_features) < 2:
                return -1000

            X_selected = X[:, selected_features]

            try:
                pls = PLSRegression(n_components=min(n_components, len(selected_features)))
                scores = cross_val_score(pls, X_selected, y, cv=min(cv_folds, len(X)),
                                         scoring='r2')
                return np.mean(scores) - 0.01 * len(selected_features)
            except:
                return -1000

        population = np.random.choice([0, 1], size=(n_population, n_features), p=[0.7, 0.3])

        best_individual = None
        best_fitness = -np.inf

        for generation in range(n_generations):
            fitness_scores = np.array([fitness(ind) for ind in population])

            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx].copy()

            new_population = []
            for _ in range(n_population):
                contestants = np.random.choice(n_population, size=3, replace=False)
                winner = population[contestants[np.argmax(fitness_scores[contestants])]]
                new_population.append(winner.copy())

            for i in range(0, n_population - 1, 2):
                if np.random.random() < 0.8:
                    crossover_point = np.random.randint(1, n_features)
                    new_population[i][crossover_point:], new_population[i + 1][crossover_point:] = \
                        new_population[i + 1][crossover_point:].copy(), new_population[i][crossover_point:].copy()

            for i in range(n_population):
                for j in range(n_features):
                    if np.random.random() < 0.1:
                        new_population[i][j] = 1 - new_population[i][j]

            population = np.array(new_population)

        return best_individual, best_fitness

    def _pls_with_feature_selection(self, X_train, X_test, y_train, y_test, feature_names):
        """PLS с Genetic Algorithm и VIP отбором признаков"""
        print("=== PLS WITH FEATURE SELECTION ===")

        try:
            X_temp, X_val, y_temp, y_val = self.improved_train_test_split(X_train, y_train, test_size=0.15)

            pls_scaler = StandardScaler()
            X_temp_scaled = pls_scaler.fit_transform(X_temp)
            X_val_scaled = pls_scaler.transform(X_val)

            n_components_range = min(8, X_temp_scaled.shape[1], X_temp_scaled.shape[0] - 1)
            best_components = 3

            if n_components_range > 1:
                r2_scores = []
                for n_comp in range(1, n_components_range + 1):
                    try:
                        pls = PLSRegression(n_components=n_comp)
                        scores = cross_val_score(pls, X_temp_scaled, y_temp,
                                                 cv=min(5, len(X_temp)), scoring='r2')
                        r2_scores.append(np.mean(scores))
                    except:
                        r2_scores.append(-1)

                if r2_scores:
                    best_components = np.argmax(r2_scores) + 1
                    print(f"Optimal PLS components: {best_components}")

            print("Running Genetic Algorithm for feature selection...")
            best_features, best_fitness = self._genetic_algorithm_feature_selection(
                X_temp_scaled, y_temp, n_components=best_components,
                n_population=15, n_generations=20
            )

            selected_indices = np.where(best_features)[0]
            print(f"Selected {len(selected_indices)} features via GA")

            if len(selected_indices) == 0:
                print("No features selected by GA, using all features")
                selected_indices = np.arange(X_temp_scaled.shape[1])

            X_temp_selected = X_temp_scaled[:, selected_indices]
            X_val_selected = X_val_scaled[:, selected_indices]

            pls_model = PLSRegression(n_components=min(best_components, len(selected_indices)))
            pls_model.fit(X_temp_selected, y_temp)

            vip_scores = self._calculate_vip(pls_model)

            feature_vip = list(zip([feature_names[selected_indices[i]] for i in range(len(selected_indices))],
                                   vip_scores))
            feature_vip.sort(key=lambda x: x[1], reverse=True)

            print("\nTop features by VIP:")
            for feature, vip in feature_vip[:10]:
                print(f"  {feature}: {vip:.4f}")

            best_val_r2 = -np.inf
            best_num_features = min(10, len(selected_indices))
            best_final_indices = selected_indices

            for num_features in range(min(15, len(selected_indices)), 2, -1):
                top_indices = [selected_indices[i] for i in np.argsort(vip_scores)[-num_features:]]

                X_temp_final = X_temp_scaled[:, top_indices]
                X_val_final = X_val_scaled[:, top_indices]

                try:
                    pls_temp = PLSRegression(n_components=min(best_components, num_features))
                    pls_temp.fit(X_temp_final, y_temp)

                    y_pred_val = pls_temp.predict(X_val_final)
                    val_r2 = r2_score(y_val, y_pred_val)

                    if val_r2 > best_val_r2:
                        best_val_r2 = val_r2
                        best_num_features = num_features
                        best_final_indices = top_indices
                except:
                    continue

            print(f"Optimal number of features after VIP: {best_num_features}")

            scaler_final = self.scaler
            X_train_final_scaled = scaler_final.fit_transform(X_train)
            X_test_final_scaled = scaler_final.transform(X_val)

            final_pls = PLSRegression(n_components=min(best_components, best_num_features))
            final_pls.fit(X_train_final_scaled[:, best_final_indices], y_train)

            y_pred_test = final_pls.predict(X_test_final_scaled[:, best_final_indices])
            test_r2 = r2_score(y_val, y_pred_test)

            print(f"PLS Results - Test R²: {test_r2:.4f}")

            return final_pls, best_final_indices, scaler_final, test_r2

        except Exception as e:
            print(f"PLS with feature selection failed: {e}")
            return None, None, None, -np.inf

    def improved_train_test_split(self, X, y, test_size=0.15, random_state=0):
        """Улучшенное разделение с проверкой статистик"""
        max_attempts = 10

        for attempt in range(max_attempts):
            current_seed = random_state + attempt

            n_bins = min(10, max(3, len(y) // 20))
            stratifier = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            y_binned = stratifier.fit_transform(y.values.reshape(-1, 1)).ravel()

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=current_seed,
                stratify=y_binned
            )

            train_mean, test_mean = y_train.mean(), y_test.mean()
            train_std, test_std = y_train.std(), y_test.std()

            mean_diff = abs(train_mean - test_mean)
            std_diff = abs(train_std - test_std)

            if mean_diff < 0.008 and std_diff < 0.02:
                print(f"✅ Удачное разделение (попытка {attempt + 1})")
                break
        else:
            print("⚠️  Не удалось найти идеальное разделение, используем последнее")

        print("=== ПРОВЕРКА РАЗДЕЛЕНИЯ ===")
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Train y: mean={y_train.mean():.3f}, std={y_train.std():.3f}")
        print(f"Test y:  mean={y_test.mean():.3f}, std={y_test.std():.3f}")
        print(f"Difference in means: {abs(y_train.mean() - y_test.mean()):.4f}")
        print(f"Difference in std: {abs(y_train.std() - y_test.std()):.4f}")

        return X_train, X_test, y_train, y_test

    def _get_data_size(self, X):
        """Определяет размер датасета"""
        n_samples = X.shape[0]

        if n_samples < 100:
            return "small"
        elif n_samples < 1000:
            return "medium"
        elif n_samples < 10000:
            return "large"
        else:
            return "very_large"

    @staticmethod
    def clean_feature_names(df):
        df.columns = (
            df.columns
            .str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
            .str.replace(r'__+', '_', regex=True)
            .str.strip('_')
        )
        return df

    def _align_features(self, X, expected_features):
        """Выравнивает признаки в X согласно expected_features"""
        if isinstance(X, pd.DataFrame):
            X_aligned = pd.DataFrame(0.0, index=X.index, columns=expected_features)
            common_features = set(X.columns) & set(expected_features)
            for feature in common_features:
                X_aligned[feature] = X[feature]
        else:
            X_aligned = pd.DataFrame(X, columns=expected_features)

        missing_features = set(expected_features) - set(X_aligned.columns)
        if missing_features:
            print(f"⚠️  Отсутствуют признаки: {list(missing_features)[:5]}...")

        return X_aligned

    def _objective(self, trial, X, y, data_size, n_features, stage="coarse"):
        """Основная функция для оптимизации"""
        try:
            if stage == "coarse":
                return self._coarse_objective(trial, X, y, data_size, n_features)
            elif stage == "fine":
                return self._fine_objective(trial, X, y, data_size, n_features)
            else:
                raise ValueError(f"Unknown stage: {stage}")

        except Exception as e:
            if not self._timeout_flag:
                print(f"❌ Ошибка в objective (stage={stage}): {e}")
            return -1e10

    def _coarse_objective(self, trial, X, y, data_size, n_features):
        """Грубый поиск с широкими диапазонами параметров"""
        print(f"🔍 Coarse search trial {trial.number}", end="\r")

        models = [
            "stacking", "lightgbm", "xgboost", "random_forest", "histgradientboosting",
            "extratrees", "lasso", "ridge", "elasticnet", "knn",
        ]

        if self.enable_pls and n_features > 1:
            models.append("pls")

        if data_size in ["medium", "large"] and n_features < 100:
            models.append("stacking")

        if data_size in ["medium", "large"] and n_features < 40:
            models.extend(["catboost", "svr"])

        if data_size == "medium" or data_size == "large":
            models.extend(["ada_boost"])

        if data_size == "large" and n_features < 50:
            models.append("gaussian_process")

        if data_size == "medium" and n_features < 30:
            models.append("mlp")

        models_per_type = max(1, self.n_trials // len(models))
        current_batch = trial.number // models_per_type
        model_index = current_batch % len(models)

        model_name = models[model_index]
        trial.set_user_attr("model", model_name)

        if model_name == "stacking":
            return self._evaluate_stacking_model(trial, X, y, data_size, n_features)
        else:
            return self._evaluate_model(trial, model_name, X, y, data_size, n_features)

    def _fine_objective(self, trial, X, y, data_size, n_features):
        """Тонкая настройка"""
        print(f"🎯 Fine tuning trial {trial.number}")

        if not hasattr(self, 'best_coarse_params') or self.best_coarse_params is None:
            return self._coarse_objective(trial, X, y, data_size, n_features)

        if hasattr(self, 'coarse_study') and self.coarse_study is not None:
            best_coarse_trial = self.coarse_study.best_trial
            model_name = best_coarse_trial.user_attrs.get("model")
        else:
            model_name = self.best_coarse_params.get("model")

        if model_name is None:
            model_name = "random_forest"
            print("⚠️ Could not determine model, using random_forest as default")

        if model_name == "stacking":
            ensemble_type = self.best_coarse_params.get("stack_type")
            meta_type = self.best_coarse_params.get("meta_type", "linear")
            n_folds = self.best_coarse_params.get("stack_folds", 5)

            print(f"🔒 Stacking fine-tuning: fixed composition={ensemble_type}, meta={meta_type}, folds={n_folds}")

            trial.set_user_attr("fixed_stack_type", ensemble_type)
            trial.set_user_attr("fixed_meta_type", meta_type)
            trial.set_user_attr("fixed_stack_folds", n_folds)

        trial.set_user_attr("model", model_name)

        fine_params = {}
        fine_params["model"] = model_name

        prefix_map = {
            "xgboost": "fine_xgb_", "lightgbm": "fine_lgb_", "catboost": "fine_cb_",
            "random_forest": "fine_rf_", "extratrees": "fine_et_", "histgradientboosting": "fine_hgb_",
            "lasso": "fine_lasso_", "ridge": "fine_ridge_", "elasticnet": "fine_enet_", "svr": "fine_svr_",
            "pls": "fine_pls_", "knn": "fine_knn_", "mlp": "fine_mlp_", "gaussian_process": "fine_gp_",
            "ada_boost": "fine_ada_", "stacking": "fine_stack_"
        }

        prefix = prefix_map.get(model_name, "")

        for param_name, best_value in self.best_coarse_params.items():
            if param_name == "model":
                continue

            if model_name == "stacking":
                if param_name in ["stack_type", "meta_type", "stack_folds"]:
                    fine_params[param_name] = best_value
                    continue

            fine_param_name = f"{prefix}{param_name}"

            if not param_name.startswith(prefix):
                continue

            clean_name = param_name.replace("xgb_", "").replace("lgb_", "").replace("rf_", "").replace("et_",
                                                                                                       "").replace(
                "hgb_", "").replace("lasso_", "").replace("ridge_", "").replace("enet_", "").replace("svr_",
                                                                                                     "").replace("pls_",
                                                                                                                 "").replace(
                "knn_", "").replace("mlp_", "").replace("gp_", "").replace("ada_", "").replace("cb_", "")

            if clean_name in ["criterion", "bootstrap", "kernel", "weights", "max_features"]:
                fine_params[fine_param_name] = best_value
                continue

            if clean_name == "min_samples_split":
                low = max(2, int(best_value * 0.5))
                high = max(2, int(best_value * 1.5))
                if low < high:
                    fine_params[fine_param_name] = trial.suggest_int(param_name, low, high)
                else:
                    fine_params[fine_param_name] = max(2, best_value)
                continue

            elif clean_name == "min_samples_leaf":
                low = max(1, int(best_value * 0.5))
                high = max(1, int(best_value * 1.5))
                if low < high:
                    fine_params[fine_param_name] = trial.suggest_int(param_name, low, high)
                else:
                    fine_params[fine_param_name] = max(1, best_value)
                continue

            if clean_name == "n_estimators":
                new_value = trial.suggest_int(param_name,
                                              max(40, int(best_value * 0.5)),
                                              int(best_value * 1.5))
                fine_params[fine_param_name] = new_value

            elif clean_name == "max_depth":
                new_value = trial.suggest_int(param_name,
                                              max(2, int(best_value * 0.5)),
                                              int(best_value * 1.5))
                fine_params[fine_param_name] = new_value

            elif clean_name in ["learning_rate", "subsample", "colsample_bytree"]:
                new_value = trial.suggest_float(param_name,
                                                max(0.001, best_value * 0.5),
                                                min(1.0, best_value * 4))
                fine_params[fine_param_name] = new_value

            elif clean_name in ["reg_alpha", "reg_lambda", "l2_leaf_reg"]:
                new_value = trial.suggest_float(param_name,
                                                max(1e-6, best_value * 0.3),
                                                best_value * 3.0,
                                                log=True)
                fine_params[fine_param_name] = new_value

            elif isinstance(best_value, int) and best_value > 1:
                new_value = trial.suggest_int(param_name,
                                              max(1, int(best_value * 0.5)),
                                              int(best_value * 1.5))
                fine_params[fine_param_name] = new_value

            elif isinstance(best_value, float):
                new_value = trial.suggest_float(param_name,
                                                max(1e-9, best_value * 0.5),
                                                best_value * 1.5)
                fine_params[fine_param_name] = new_value
            else:
                fine_params[fine_param_name] = best_value

        return self._evaluate_model(trial, model_name, X, y, data_size, n_features)

    def _evaluate_model_with_params(self, trial, params, X, y, data_size, n_features):
        """Оценка модели с готовыми параметрами"""
        try:
            model_name = params["model"]
            cleaned_params = {}
            prefix_map = {
                "xgboost": "fine_xgb_", "lightgbm": "fine_lgb_", "catboost": "fine_cb_",
                "random_forest": "fine_rf_", "extratrees": "fine_et_", "histgradientboosting": "fine_hgb_",
                "lasso": "fine_lasso_", "ridge": "fine_ridge_", "elasticnet": "fine_enet_", "svr": "fine_svr_",
                "pls": "fine_pls_", "knn": "fine_knn_", "mlp": "fine_mlp_", "gaussian_process": "fine_gp_",
                "ada_boost": "fine_ada_"
            }

            prefix = prefix_map.get(model_name, "fine_")

            all_prefixes = list(prefix_map.values()) + ["fine_"]

            for param_name, value in params.items():
                if param_name == "model":
                    continue

                if param_name.startswith(prefix):
                    clean_name = param_name[len(prefix):]
                    cleaned_params[clean_name] = value
                elif not any(param_name.startswith(p) for p in all_prefixes if p != prefix):
                    cleaned_params[param_name] = value

            final_cleaned_params = self._clean_params_for_final_model(cleaned_params, model_name)

            model = self._instantiate_clean_model(model_name, final_cleaned_params)

            if model is None:
                return -1e10

            X_array = np.asarray(X)
            y_array = np.asarray(y)

            slow_models = ["xgboost", "catboost", "histgradientboosting", "extratrees", "svr"]

            if len(X) < 100:
                n_splits = min(3, len(X) // 5)
            elif len(X) < 1000 or model_name in slow_models:
                n_splits = min(4, len(X) // 20)
            else:
                n_splits = min(6, len(X) // 50)

            n_splits = max(2, n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

            if self.scale_data:
                model_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
            else:
                model_pipeline = model

            fold_scores = []
            train_r2_scores = []
            val_r2_scores = []

            for train_idx, val_idx in kf.split(X_array):
                X_train_fold, X_val_fold = X_array[train_idx], X_array[val_idx]
                y_train_fold, y_val_fold = y_array[train_idx], y_array[val_idx]

                try:
                    model_pipeline.fit(X_train_fold, y_train_fold)
                    y_pred_val = model_pipeline.predict(X_val_fold)
                    y_pred_train = model_pipeline.predict(X_train_fold)

                    fold_score = -mean_squared_error(y_val_fold, y_pred_val)
                    train_r2 = r2_score(y_train_fold, y_pred_train)
                    val_r2 = r2_score(y_val_fold, y_pred_val)

                    fold_scores.append(fold_score)
                    train_r2_scores.append(train_r2)
                    val_r2_scores.append(val_r2)

                except Exception as e:
                    print(f"⚠️ Fold error: {e}")
                    fold_scores.append(-1e10)
                    train_r2_scores.append(0)
                    val_r2_scores.append(0)

            mean_score = np.mean(fold_scores)
            mean_train_r2 = np.mean(train_r2_scores)
            mean_val_r2 = np.mean(val_r2_scores)
            overfit_gap = mean_train_r2 - mean_val_r2

            penalty = 0
            if overfit_gap > 0.2:
                penalty = overfit_gap * 2
                mean_val_r2 -= penalty
            elif overfit_gap > 0.1:
                penalty = overfit_gap
                mean_val_r2 -= penalty

            trial.set_user_attr("overfit_penalty", penalty)
            trial.set_user_attr("overfit_gap", overfit_gap)
            trial.set_user_attr("train_r2", mean_train_r2)
            trial.set_user_attr("val_r2", mean_val_r2)

            return mean_val_r2

        except Exception as e:
            print(f"❌ Model evaluation failed for {model_name}: {e}")
            return -1e10

        except Exception as e:
            print(f"❌ Model evaluation failed for {model_name}: {e}")
            return -1e10

    def _evaluate_model(self, trial, model_name, X, y, data_size, n_features):
        """Универсальная функция оценки модели с кросс-валидацией"""
        try:
            model = self._create_model(model_name, trial, data_size, n_features)
            if model is None:
                return -1e10

        except Exception as e:
            print(f"❌ Model creation failed for {model_name}: {e}")
            return -1e10

        X_array = np.asarray(X)
        y_array = np.asarray(y)

        slow_models = ["xgboost", "catboost", "histgradientboosting", "extratrees", "svr"]

        if len(X) < 100:
            n_splits = min(3, len(X) // 5)
        elif len(X) < 1000 or model_name in slow_models:
            n_splits = min(4, len(X) // 20)
        else:
            n_splits = min(6, len(X) // 50)

        n_splits = max(2, n_splits)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        try:
            if self.scale_data:
                model_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
            else:
                model_pipeline = model

            fold_scores = []
            train_r2_scores = []
            val_r2_scores = []

            for train_idx, val_idx in kf.split(X_array):
                X_train_fold, X_val_fold = X_array[train_idx], X_array[val_idx]
                y_train_fold, y_val_fold = y_array[train_idx], y_array[val_idx]

                try:
                    model_pipeline.fit(X_train_fold, y_train_fold)
                    y_pred_val = model_pipeline.predict(X_val_fold)
                    y_pred_train = model_pipeline.predict(X_train_fold)

                    fold_score = -mean_squared_error(y_val_fold, y_pred_val)
                    train_r2 = r2_score(y_train_fold, y_pred_train)
                    val_r2 = r2_score(y_val_fold, y_pred_val)

                    fold_scores.append(fold_score)
                    train_r2_scores.append(train_r2)
                    val_r2_scores.append(val_r2)

                except Exception as e:
                    print(f"⚠️ Fold error: {e}")
                    fold_scores.append(-1e10)
                    train_r2_scores.append(0)
                    val_r2_scores.append(0)

            mean_score = np.mean(fold_scores)
            mean_train_r2 = np.mean(train_r2_scores)
            mean_val_r2 = np.mean(val_r2_scores)
            overfit_gap = mean_train_r2 - mean_val_r2

            penalty = 0
            if overfit_gap > 0.1:
                penalty = overfit_gap * 2
                mean_val_r2 -= penalty

            trial.set_user_attr("overfit_penalty", penalty)
            trial.set_user_attr("overfit_gap", overfit_gap)
            trial.set_user_attr("train_r2", mean_train_r2)
            trial.set_user_attr("val_r2", mean_val_r2)

            return mean_val_r2

        except Exception as e:
            print(f"❌ Model evaluation failed for {model_name}: {e}")
            return -1e10

    def _create_model(self, name, trial_or_params, data_size, n_features):
        """Создает модель с учетом параметров"""

        if name == "stacking":
            if isinstance(trial_or_params, dict):
                return self._create_final_stacking_ensemble(trial_or_params, data_size, n_features)
            else:
                return self._create_stacking_ensemble(trial_or_params, data_size, n_features)
        if isinstance(trial_or_params, dict):
            params = trial_or_params.copy()
            params.pop('model', None)
            cleaned_params = self._clean_params_for_final_model(params, name)
            return self._instantiate_clean_model(name, cleaned_params)

        else:
            trial = trial_or_params
            params = {}

            if data_size == "small":
                n_est_range = (50, 1000)
                depth_range = (2, 16)
                lr_range = (1e-3, 0.3)
                knn_neighbors_range = (3, 15)
            elif data_size == "medium":
                n_est_range = (100, 1500)
                depth_range = (3, 18)
                lr_range = (1e-3, 0.4)
                knn_neighbors_range = (3, 25)
            elif data_size == "large":
                n_est_range = (150, 2000)
                depth_range = (4, 20)
                lr_range = (1e-3, 0.5)
                knn_neighbors_range = (3, 35)
            else:
                n_est_range = (200, 3000)
                depth_range = (5, 25)
                lr_range = (1e-3, 0.6)
                knn_neighbors_range = (3, 50)

            if name == "pls":
                params["n_components"] = trial.suggest_int("pls_n_components", 1, min(10, n_features))

            elif name == "xgboost":
                params["n_estimators"] = trial.suggest_int("xgb_n_estimators", n_est_range[0], n_est_range[1]//3)
                params["max_depth"] = trial.suggest_int("xgb_max_depth", depth_range[0], depth_range[1])
                params["learning_rate"] = trial.suggest_float("xgb_learning_rate", lr_range[0], lr_range[1], log=True)
                params["subsample"] = trial.suggest_float("xgb_subsample", 0.6, 1.0)
                params["colsample_bytree"] = trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0)
                params["reg_alpha"] = trial.suggest_float("xgb_reg_alpha", 1e-3, 10.0, log=True)
                params["reg_lambda"] = trial.suggest_float("xgb_reg_lambda", 1e-3, 10.0, log=True)

            elif name == "lightgbm":
                params["n_estimators"] = trial.suggest_int("lgb_n_estimators", n_est_range[0], n_est_range[1]//3)
                params["max_depth"] = trial.suggest_int("lgb_max_depth", depth_range[0], depth_range[1])
                params["learning_rate"] = trial.suggest_float("lgb_learning_rate", lr_range[0], lr_range[1], log=True)
                params["subsample"] = trial.suggest_float("lgb_subsample", 0.6, 1.0)
                params["colsample_bytree"] = trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0)
                params["reg_alpha"] = trial.suggest_float("lgb_reg_alpha", 1e-3, 10.0, log=True)
                params["reg_lambda"] = trial.suggest_float("lgb_reg_lambda", 1e-3, 10.0, log=True)

            elif name == "catboost":
                params["iterations"] = trial.suggest_int("cb_iterations", n_est_range[0], n_est_range[1])
                params["depth"] = trial.suggest_int("cb_depth", depth_range[0], min(16, depth_range[1]))
                params["learning_rate"] = trial.suggest_float("cb_learning_rate", lr_range[0], lr_range[1], log=True)
                params["l2_leaf_reg"] = trial.suggest_float("cb_l2_leaf_reg", 1.0, 10.0)
                params["border_count"] = trial.suggest_int("cb_border_count", 1, 255)
                params["random_strength"] = trial.suggest_float("cb_random_strength", 1e-9, 10.0, log=True)

            elif name == "random_forest":
                params["n_estimators"] = trial.suggest_int("rf_n_estimators", n_est_range[0], n_est_range[1]//3)
                params["max_depth"] = trial.suggest_int("rf_max_depth", depth_range[0], depth_range[1])
                params["min_samples_split"] = trial.suggest_int("rf_min_samples_split", 2, 20)
                params["min_samples_leaf"] = trial.suggest_int("rf_min_samples_leaf", 1, 10)
                params["max_features"] = trial.suggest_categorical("rf_max_features", ["sqrt", "log2"])

            elif name == "extratrees":
                params["n_estimators"] = trial.suggest_int("et_n_estimators", n_est_range[0], n_est_range[1]//3)
                params["criterion"] = trial.suggest_categorical("et_criterion", ['squared_error', 'absolute_error', 'friedman_mse'])
                params["max_depth"] = trial.suggest_int("et_max_depth", depth_range[0], depth_range[1])
                params["min_samples_split"] = trial.suggest_int("et_min_samples_split", 2, 20)
                params["min_samples_leaf"] = trial.suggest_int("et_min_samples_leaf", 1, 10)
                params["max_features"] = trial.suggest_float("et_max_features", 0.1, 0.9)
                params["bootstrap"] = trial.suggest_categorical("et_bootstrap", [True, False])

            elif name == "histgradientboosting":
                params["max_iter"] = trial.suggest_int("hgb_max_iter", n_est_range[0], n_est_range[1]//3)
                params["max_depth"] = trial.suggest_int("hgb_max_depth", depth_range[0], depth_range[1])
                params["learning_rate"] = trial.suggest_float("hgb_learning_rate", lr_range[0], lr_range[1], log=True)
                params["min_samples_leaf"] = trial.suggest_int("hgb_min_samples_leaf", 5, 50)
                params["max_bins"] = trial.suggest_int("hgb_max_bins", 50, 250)
                params["l2_regularization"] = trial.suggest_float("hgb_l2_regularization", 0, 0.1)

            elif name == "lasso":
                params["alpha"] = trial.suggest_float("lasso_alpha", 1e-4, 10.0, log=True)

            elif name == "ridge":
                params["alpha"] = trial.suggest_float("ridge_alpha", 1e-4, 10.0, log=True)

            elif name == "elasticnet":
                params["alpha"] = trial.suggest_float("enet_alpha", 1e-4, 10.0, log=True)
                params["l1_ratio"] = trial.suggest_float("enet_l1_ratio", 0.0, 1.0)

            elif name == "svr":
                params["C"] = trial.suggest_float("svr_C", 1e-2, 100.0, log=True)
                params["kernel"] = trial.suggest_categorical("svr_kernel", ["linear", "rbf"])

            elif name == "knn":
                params["n_neighbors"] = trial.suggest_int("knn_n_neighbors", knn_neighbors_range[0],
                                                          knn_neighbors_range[1])
                params["weights"] = trial.suggest_categorical("knn_weights", ["uniform", "distance"])

            elif name == "mlp":
                params["hidden_layer_sizes"] = (100,)
                params["alpha"] = trial.suggest_float("mlp_alpha", 1e-5, 1.0, log=True)
                params["learning_rate_init"] = trial.suggest_float("mlp_learning_rate_init", lr_range[0], lr_range[1], log=True)

            elif name == "ada_boost":
                params["n_estimators"] = trial.suggest_int("ada_n_estimators", n_est_range[0], n_est_range[1]//3)
                params["learning_rate"] = trial.suggest_float("ada_learning_rate", lr_range[0], lr_range[1], log=True)

            else:
                params["n_estimators"] = trial.suggest_int("rf_n_estimators", n_est_range[0], n_est_range[1]//3)
                params["max_depth"] = trial.suggest_int("rf_max_depth", depth_range[0], depth_range[1])

            return self._instantiate_clean_model(name, params)

    def _instantiate_clean_model(self, name, params):
        """Создание модели с защитой от некорректных параметров"""
        safe_params = params.copy()

        for param in ['n_jobs', 'thread_count', 'num_threads']:
            safe_params.pop(param, None)

        conflict_params = ['verbosity', 'silent', 'verbose', 'distribution', 'param_name']
        for param in conflict_params:
            safe_params.pop(param, None)

        try:
            if name == "random_forest":
                return RandomForestRegressor(**safe_params, random_state=self.random_state, n_jobs=-1)
            elif name == "extratrees":
                return ExtraTreesRegressor(**safe_params, random_state=self.random_state, n_jobs=-1)
            elif name == "xgboost":
                return XGBRegressor(**safe_params, random_state=self.random_state, verbosity=0)
            elif name == "lightgbm":
                return LGBMRegressor(**safe_params, random_state=self.random_state, verbose=-1)
            elif name == "catboost":
                return CatBoostRegressor(**safe_params, random_seed=self.random_state, verbose=False)
            elif name == "lasso":
                return Lasso(**safe_params, random_state=self.random_state, max_iter=1000)
            elif name == "ridge":
                return Ridge(**safe_params, random_state=self.random_state, max_iter=1000)
            elif name == "elasticnet":
                return ElasticNet(**safe_params, random_state=self.random_state, max_iter=1000)
            elif name == "svr":
                return SVR(**safe_params)
            elif name == "pls":
                return PLSRegression(**safe_params)
            elif name == "knn":
                return KNeighborsRegressor(**safe_params, n_jobs=-1)
            elif name == "ada_boost":
                return AdaBoostRegressor(**safe_params, random_state=self.random_state)
            elif name == "histgradientboosting":
                return HistGradientBoostingRegressor(**safe_params, random_state=self.random_state)
            elif name == "mlp":
                return MLPRegressor(**safe_params, random_state=self.random_state, max_iter=1000)
            else:
                print(f"⚠️ Unknown model {name}, using RandomForest")
                return RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)

        except Exception as e:
            print(f"❌ Error creating model {name}: {e}")
            return RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)

    def _clean_params_for_final_model(self, params, model_name):
        """очистка параметров от префиксов"""
        prefix_map = {
            "xgboost": ["xgb_", "fine_xgb_"],
            "lightgbm": ["lgb_", "fine_lgb_"],
            "catboost": ["cb_", "fine_cb_"],
            "random_forest": ["rf_", "fine_rf_"],
            "extratrees": ["et_", "fine_et_"],
            "histgradientboosting": ["hgb_", "fine_hgb_"],
            "lasso": ["lasso_", "fine_lasso_"],
            "ridge": ["ridge_", "fine_ridge_"],
            "elasticnet": ["enet_", "fine_enet_"],
            "svr": ["svr_", "fine_svr_"],
            "pls": ["pls_", "fine_pls_"],
            "knn": ["knn_", "fine_knn_"],
            "mlp": ["mlp_", "fine_mlp_"],
            "gaussian_process": ["gp_", "fine_gp_"],
            "ada_boost": ["ada_", "fine_ada_"],
            "stacking": ["stack_", "meta_"]
        }

        prefix = prefix_map.get(model_name, "")
        cleaned_params = {}

        print(f"🔧 Cleaning params for '{model_name}': prefix '{prefix}'")
        print(f"📊 Original params: {params}")

        for param_name, value in params.items():
            if param_name == "model":
                continue

            if param_name.startswith("fine_variation_"):
                continue

            clean_name = param_name
            for pr in prefix:
                if clean_name.startswith(pr):
                    clean_name = clean_name[len(pr):]
                    break

            has_other_prefix = False
            for other_model, other_prefixes in prefix_map.items():
                if other_model != model_name:
                    for other_prefix in other_prefixes:
                        if param_name.startswith(other_prefix):
                            has_other_prefix = True
                            break
                    if has_other_prefix:
                        break

            if not has_other_prefix:
                cleaned_params[clean_name] = value

        return cleaned_params

    def fit(self, X, y):
        try:
            X = self.clean_feature_names(X)
            self.original_feature_names = X.columns.tolist()
            self.fitted_feature_names_ = self.original_feature_names.copy()

            print("=== АНАЛИЗ ДАННЫХ ===")
            print(f"Размер данных: {X.shape}")
            print(f"Диапазон y: [{y.min():.3f}, {y.max():.3f}]")

            X_train, X_test_final, y_train, y_test_final = self.improved_train_test_split(X, y, test_size=0.15)

            if self.pre_filter:
                X_train = self._pre_filter(X_train)
                print(f"После фильтрации: {X_train.shape[1]} признаков")
                self.fitted_feature_names_ = X_train.columns.tolist()

            if X_train.shape[1] == 0:
                raise ValueError("После предварительной фильтрации не осталось признаков!")

            n_features_to_select = min(self.max_features or X_train.shape[1], X_train.shape[1])

            if self.feature_selection_method == "shap" and X_train.shape[1] > 1:
                try:
                    X_shap_train, X_shap_val, y_shap_train, y_shap_val = self.improved_train_test_split(
                        X_train, y_train, test_size=0.15)
                    selected_features = self._shap_feature_selection(
                        X_shap_train.values, X_shap_val.values, y_shap_train,
                        self.fitted_feature_names_, n_features_to_select
                    )
                    selection_method_used = "SHAP"
                    print("🎯 Используем SHAP feature selection")

                except Exception as e:
                    print(f"❌ Feature selection failed, using all features: {e}")
                    selected_features = self.fitted_feature_names_
                    selection_method_used = "ALL"

            elif self.feature_selection_method == "pls" and self.enable_pls and X_train.shape[1] > 1:
                try:
                    X_pls_train, X_pls_val, y_pls_train, y_pls_val = self.improved_train_test_split(
                        X_train, y_train, test_size=0.15
                    )
                    pls_model, pls_features_indices, pls_scaler, pls_train_r2 = self._pls_with_feature_selection(
                        X_pls_train, X_pls_val, y_pls_train, y_pls_val, self.fitted_feature_names_
                    )
                    if pls_model is not None and len(pls_features_indices) > 0:
                        selected_features = [self.fitted_feature_names_[i] for i in pls_features_indices]
                        if self.max_features and len(selected_features) > self.max_features:
                            selected_features = selected_features[:self.max_features]
                        selection_method_used = "PLS+GA"
                    else:
                        raise ValueError("PLS feature selection returned no features")
                except Exception as e:
                    print(f"❌ PLS feature selection failed: {e}")
                    selected_features = self.fitted_feature_names_
                    selection_method_used = "ALL"
            else:
                selected_features = self._simple_feature_selection(
                    X_train.values, y_train,
                    self.fitted_feature_names_,
                    n_features_to_select
                )
                selection_method_used = "RandomForest"

            self.final_selected_features = selected_features
            print(f"🎯 Используем {len(selected_features)} признаков (метод: {selection_method_used})")

            X_optuna = X_train[selected_features]
            data_size = self._get_data_size(X_optuna)

            if self.warm_start:
                coarse_trials = max(10, int(self.n_trials * self.coarse_trials_ratio))
                fine_trials = max(10, self.n_trials - coarse_trials)

                print(f"\n🚀 WARM START OPTIMIZATION")
                print(f"   Coarse: {coarse_trials}, Fine: {fine_trials}")

                coarse_study = optuna.create_study(direction="maximize", study_name="coarse_search")
                #coarse_pbar = tqdm(total=coarse_trials, desc="Coarse search")

                #def coarse_callback(study, trial):
                    #model = trial.user_attrs.get("model", "unknown")
                    #coarse_pbar.set_postfix({"best": f"{study.best_value:.4f}", "model": model})
                    #coarse_pbar.update(1)
                    #print(f"📊 Callback: trial {trial.number}, state: {trial.state}, value: {trial.value}")

                print(f"🚀 Starting coarse optimization with {coarse_trials} trials...")
                coarse_study.optimize(
                    lambda trial: self._objective(trial, X_optuna.values, y_train, data_size, X_optuna.shape[1],
                                                  "coarse"),
                    n_trials=coarse_trials,
                    #callbacks=[coarse_callback],
                    timeout=self.timeout * 0.4,
                    show_progress_bar=True
                )
                print(f"🏁 Coarse optimization finished. Completed trials: {len(coarse_study.trials)}")
                #coarse_pbar.close()

                self.best_coarse_params = coarse_study.best_params.copy() if len(coarse_study.trials) > 0 else {}

                if not self.best_coarse_params or "model" not in self.best_coarse_params:
                    best_trial = coarse_study.best_trial if len(coarse_study.trials) > 0 else None
                    model_name = best_trial.user_attrs.get("model", "random_forest") if best_trial else "random_forest"
                    self.best_coarse_params = {"model": model_name}

                print(f"✅ Coarse best: {self.best_coarse_params.get('model')}, score: {coarse_study.best_value:.4f}")

                study = optuna.create_study(direction="maximize", study_name="fine_tuning")
                study.add_trial(coarse_study.best_trial)

                fine_pbar = tqdm(total=fine_trials, desc="Fine tuning")

                def fine_callback(study, trial):
                    fine_pbar.set_postfix({"best": f"{study.best_value:.4f}"})
                    fine_pbar.update(1)

                study.optimize(
                    lambda trial: self._objective(trial, X_optuna.values, y_train, data_size, X_optuna.shape[1],
                                                  "fine"),
                    n_trials=fine_trials,
                    callbacks=[fine_callback],
                    timeout=self.timeout * 0.6,
                )
                fine_pbar.close()

            else:
                study = optuna.create_study(direction="maximize")
                pbar = tqdm(total=self.n_trials, desc="Optimization")

                def standard_callback(study, trial):
                    pbar.set_postfix({"best": f"{study.best_value:.4f}"})
                    pbar.update(1)

                study.optimize(
                    lambda trial: self._objective(trial, X_optuna.values, y_train, data_size, X_optuna.shape[1],
                                                  "coarse"),
                    n_trials=self.n_trials,
                    callbacks=[standard_callback],
                    timeout=self.timeout,
                )
                pbar.close()

            print(f"\n=== ФИНАЛЬНОЕ ОБУЧЕНИЕ ===")

            if len(study.trials) == 0:
                print("⚠️ No successful trials, using default model")
                self.best_model_name = "random_forest"
                self.best_params = {"n_estimators": 100, "max_depth": 6}
            else:
                self.best_params = study.best_params.copy()

                if "model" in self.best_params:
                    self.best_model_name = self.best_params.pop("model")
                else:
                    self.best_model_name = study.best_trial.user_attrs.get("model", "random_forest")

                print(f"🎯 Best model: {self.best_model_name}")
                print(f"📊 Best score: {study.best_value:.4f}")
                print(f"📊 Best parameters: {self.best_params}")

            final_params = self._clean_params_for_final_model(self.best_params, self.best_model_name)
            self.model = self._create_model(self.best_model_name, final_params, data_size, X_optuna.shape[1])

            X_final_train = X_train[self.final_selected_features]
            if self.scale_data:
                X_final_scaled = self.scaler.fit_transform(X_final_train)
            else:
                X_final_scaled = X_final_train.values

            self.model.fit(X_final_scaled, y_train)
            self.selected_features = self.final_selected_features
            self.study = study

            self._final_validation(X_train, y_train, X_test_final, y_test_final)

            print("✅ AutoML обучение завершено!")
            return self

        except Exception as e:
            print(f"❌ Critical error in fit: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _final_validation(self, X_train, y_train, X_test, y_test):
        """Финальная проверка модели"""
        print("\n" + "=" * 50)
        print("ФИНАЛЬНАЯ ВАЛИДАЦИЯ")
        print("=" * 50)

        train_pred = self.predict(X_train)
        test_pred = self.predict(X_test)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print(f"Модель: {self.best_model_name}")
        print(f"Признаков: {len(self.selected_features)}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²:  {test_r2:.4f}")
        print(f"Разница:  {train_r2 - test_r2:.4f}")

        if train_r2 - test_r2 > 0.3:
            print("🚨 Внимание: возможное переобучение!")
        elif test_r2 > 0.7:
            print("✅ Отличное качество модели!")
        elif test_r2 > 0.5:
            print("⚠️  Умеренное качество модели")
        else:
            print("❌ Низкое качество модели")

    def predict(self, X):
        """Предсказание с выравниванием признаков"""
        try:
            X = self.clean_feature_names(X)
            X_aligned = self._align_features(X, self.selected_features)

            if self.scale_data:
                X_scaled = self.scaler.transform(X_aligned)
            else:
                X_scaled = X_aligned.values

            return self.model.predict(X_scaled)

        except Exception as e:
            print(f"❌ Predict error: {e}")
            return np.zeros(len(X))

    def score(self, X, y):
        """Оценка модели"""
        try:
            preds = self.predict(X)
            return {
                "R2": r2_score(y, preds),
                "MAE": mean_absolute_error(y, preds),
                "RMSE": root_mean_squared_error(y, preds),
            }
        except Exception as e:
            print(f"❌ Score error: {e}")
            return {"R2": -1, "MAE": 10, "RMSE": 10}

    def _shap_feature_selection(self, X_train, X_val, y_train, feature_names, n_features=20):
        """SHAP-based feature selection"""
        print("=== SHAP FEATURE SELECTION ===")

        try:
            base_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1
            )

            base_model.fit(X_train, y_train)

            try:
                explainer = shap.TreeExplainer(base_model, X_train)
                shap_values = explainer.shap_values(X_val)
            except Exception as e:
                print(f"⚠️  TreeExplainer failed, using KernelExplainer: {e}")
                explainer = shap.KernelExplainer(base_model.predict,
                                                 X_train[:100])
                shap_values = explainer.shap_values(X_val)

            if len(shap_values.shape) == 2:
                shap_importance = np.abs(shap_values).mean(axis=0)
            else:
                shap_importance = np.abs(shap_values).mean(axis=0).mean(axis=0) if len(
                    shap_values.shape) == 3 else np.abs(shap_values).mean()

            feature_importance = list(zip(feature_names, shap_importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)

            print("\nTop features by SHAP importance:")
            for i, (feature, importance) in enumerate(feature_importance[:15]):
                print(f"  {i + 1:2d}. {feature}: {importance:.4f}")

            selected_features = [feature for feature, _ in feature_importance[:n_features]]
            print(f"Selected {len(selected_features)} features via SHAP")

            return selected_features

        except Exception as e:
            print(f"❌ SHAP feature selection failed, using simple method: {e}")
            return self._simple_feature_selection(X_train, y_train, feature_names, n_features)

    def get_important_features(self):
        return self.selected_features

    def plot_shap_summary(self, X):
        """SHAP анализ"""
        try:
            if not hasattr(self, 'selected_features') or not self.selected_features:
                print("[WARN] No selected features for SHAP analysis")
                return

            X_aligned = self._align_features(X, self.selected_features)
            X_scaled = self.scaler.transform(X_aligned) if self.scale_data else np.asarray(X_aligned)

            if self.best_model_name == "random_forest":
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_scaled)
            else:
                explainer = shap.Explainer(self.model, X_scaled)
                shap_values = explainer(X_scaled)

            print("\n=== SHAP АНАЛИЗ ===")
            shap.summary_plot(shap_values, X_aligned, show=False)

        except Exception as e:
            print(f"[WARN] SHAP summary failed: {e}")

    def _create_stacking_ensemble(self, trial, data_size, n_features):

        ensemble_type = trial.suggest_categorical("stack_type", [
            "tree_boosters",
            "tree_ensemble",
            "mixed_bagging",
            "gradient_boosters_only",
            "bagging_ensemble",
            "mixed_diverse",
            "fast_ensemble",
            "conservative_mix"
        ])

        meta_type = trial.suggest_categorical("meta_type", [
            "linear", "ridge", "lightgbm", "random_forest", "xgboost"
        ])

        n_folds = trial.suggest_int("stack_folds", 3, 5)

        base_model_configs = []

        if ensemble_type == "tree_boosters":
            base_models_types = ["lightgbm", "xgboost", "catboost"]
            print("🎯 Composition: Gradient Boosters Only")

        elif ensemble_type == "tree_ensemble":
            base_models_types = ["random_forest", "extratrees", "lightgbm", "xgboost"]
            print("🎯 Composition: Tree Ensemble (Bagging + Boosting)")

        elif ensemble_type == "mixed_bagging":
            base_models_types = ["random_forest", "extratrees", "lightgbm", "histgradientboosting"]
            print("🎯 Composition: Mixed Bagging")

        elif ensemble_type == "gradient_boosters_only":
            base_models_types = ["lightgbm", "xgboost", "catboost", "histgradientboosting"]
            print("🎯 Composition: All Gradient Boosters")

        elif ensemble_type == "bagging_ensemble":
            base_models_types = ["random_forest", "extratrees", "histgradientboosting"]
            if data_size in ["medium", "large"]:
                base_models_types.append("ada_boost")
            print("🎯 Composition: Bagging Ensemble")

        elif ensemble_type == "mixed_diverse":
            base_models_types = ["lightgbm", "random_forest", "xgboost", "extratrees"]
            if n_features < 50:
                base_models_types.append("ridge")
            print("🎯 Composition: Highly Diverse Mix")

        elif ensemble_type == "fast_ensemble":
            base_models_types = ["lightgbm", "histgradientboosting", "extratrees"]
            if data_size == "small":
                base_models_types.append("random_forest")
            print("🎯 Composition: Fast Ensemble")

        else:
            base_models_types = ["random_forest", "lightgbm", "ridge"]
            if data_size in ["medium", "large"]:
                base_models_types.append("xgboost")
            print("🎯 Composition: Conservative Mix")

        if ensemble_type in ["mixed_diverse", "tree_ensemble", "mixed_bagging"]:
            n_models = trial.suggest_int("n_base_models",
                                         max(3, len(base_models_types) - 1),
                                         len(base_models_types))
            if n_models < len(base_models_types):
                import random
                random.seed(self.random_state + trial.number)
                base_models_types = random.sample(base_models_types, n_models)
                print(f"  Selected {n_models} models from pool: {base_models_types}")

        for model_type in base_models_types:
            params = self._get_base_model_params_for_stacking(model_type, trial, data_size, n_features)
            base_model_configs.append({
                'type': model_type,
                'params': params
            })

        meta_config = {'type': meta_type, 'params': {}}

        if meta_type == "ridge":
            meta_config['params']['alpha'] = trial.suggest_float("meta_alpha", 0.1, 10.0, log=True)
        elif meta_type == "lightgbm":
            meta_config['params'].update({
                'n_estimators': trial.suggest_int("meta_lgb_n_est", 50, 200),
                'max_depth': trial.suggest_int("meta_lgb_depth", 3, 8),
                'learning_rate': trial.suggest_float("meta_lgb_lr", 0.01, 0.3, log=True),
                'verbose': -1,
                'random_state': self.random_state
            })
        elif meta_type == "random_forest":
            meta_config['params'].update({
                'n_estimators': trial.suggest_int("meta_rf_n_est", 50, 200),
                'max_depth': trial.suggest_int("meta_rf_depth", 3, 10),
                'min_samples_split': trial.suggest_int("meta_rf_min_split", 2, 10),
                'random_state': self.random_state,
                'n_jobs': -1
            })
        elif meta_type == "xgboost":
            meta_config['params'].update({
                'n_estimators': trial.suggest_int("meta_xgb_n_est", 50, 200),
                'max_depth': trial.suggest_int("meta_xgb_depth", 3, 8),
                'learning_rate': trial.suggest_float("meta_xgb_lr", 0.01, 0.3, log=True),
                'verbosity': 0,
                'random_state': self.random_state
            })

        print(f"  Meta-model: {meta_type}, Folds: {n_folds}")
        print(f"  Total base models: {len(base_model_configs)}")

        return StackingEnsembleRegressor(
            base_model_configs=base_model_configs,
            meta_model_config=meta_config,
            n_folds=n_folds,
            random_state=self.random_state
        )

    def _get_base_model_params_for_stacking(self, model_type, trial, data_size, n_features):
        """Параметры для базовых моделей стекинга"""
        if data_size == "small":
            n_est_range = (50, 200)
            depth_range = (3, 8)
        elif data_size == "medium":
            n_est_range = (80, 300)
            depth_range = (4, 10)
        else:
            n_est_range = (100, 400)
            depth_range = (5, 12)

        params = {}

        if model_type == "lightgbm":
            params.update({
                'n_estimators': trial.suggest_int(f"stack_lgb_n_est", n_est_range[0], n_est_range[1]),
                'max_depth': trial.suggest_int(f"stack_lgb_depth", depth_range[0], depth_range[1]),
                'learning_rate': trial.suggest_float(f"stack_lgb_lr", 1e-3, 0.2, log=True),
                'subsample': trial.suggest_float(f"stack_lgb_subsample", 0.7, 1.0),
                'verbose': -1,
                'random_state': self.random_state
            })

        elif model_type == "xgboost":
            params.update({
                'n_estimators': trial.suggest_int(f"stack_xgb_n_est", n_est_range[0], n_est_range[1]),
                'max_depth': trial.suggest_int(f"stack_xgb_depth", depth_range[0], depth_range[1]),
                'learning_rate': trial.suggest_float(f"stack_xgb_lr", 1e-3, 0.2, log=True),
                'subsample': trial.suggest_float(f"stack_xgb_subsample", 0.7, 1.0),
                'verbosity': 0,
                'random_state': self.random_state
            })

        elif model_type == "random_forest":
            params.update({
                'n_estimators': trial.suggest_int(f"stack_rf_n_est", n_est_range[0], n_est_range[1]),
                'max_depth': trial.suggest_int(f"stack_rf_depth", depth_range[0], depth_range[1]),
                'min_samples_split': trial.suggest_int(f"stack_rf_min_split", 2, 10),
                'min_samples_leaf': trial.suggest_int(f"stack_rf_min_leaf", 1, 5),
                'random_state': self.random_state,
                'n_jobs': -1
            })

        elif model_type == "extratrees":
            params.update({
                'n_estimators': trial.suggest_int(f"stack_et_n_est", n_est_range[0], n_est_range[1]),
                'max_depth': trial.suggest_int(f"stack_et_depth", depth_range[0], depth_range[1]),
                'min_samples_split': trial.suggest_int(f"stack_et_min_split", 2, 10),
                'min_samples_leaf': trial.suggest_int(f"stack_et_min_leaf", 1, 5),
                'random_state': self.random_state,
                'n_jobs': -1
            })

        elif model_type == "catboost":
            params.update({
                'iterations': trial.suggest_int(f"stack_cb_iter", n_est_range[0], n_est_range[1]),
                'depth': trial.suggest_int(f"stack_cb_depth", depth_range[0], min(10, depth_range[1])),
                'learning_rate': trial.suggest_float(f"stack_cb_lr", 1e-3, 0.2, log=True),
                'random_seed': self.random_state,
                'verbose': False
            })

        elif model_type == "histgradientboosting":
            params.update({
                'max_iter': trial.suggest_int(f"stack_hgb_iter", n_est_range[0], n_est_range[1]),
                'max_depth': trial.suggest_int(f"stack_hgb_depth", depth_range[0], depth_range[1]),
                'learning_rate': trial.suggest_float(f"stack_hgb_lr", 1e-3, 0.2, log=True),
                'min_samples_leaf': trial.suggest_int(f"stack_hgb_min_leaf", 5, 20),
                'random_state': self.random_state
            })

        elif model_type == "ada_boost":
            params.update({
                'n_estimators': trial.suggest_int(f"stack_ada_n_est", n_est_range[0], n_est_range[1] // 2),
                'learning_rate': trial.suggest_float(f"stack_ada_lr", 1e-3, 1.0, log=True),
                'random_state': self.random_state
            })

        elif model_type == "ridge":
            params.update({
                'alpha': trial.suggest_float(f"stack_ridge_alpha", 1e-4, 10.0, log=True),
                'random_state': self.random_state
            })

        elif model_type == "lasso":
            params.update({
                'alpha': trial.suggest_float(f"stack_lasso_alpha", 1e-4, 10.0, log=True),
                'random_state': self.random_state,
                'max_iter': 1000
            })

        return params

    def _evaluate_stacking_model(self, trial, X, y, data_size, n_features):
        """Оценка стэкинг модели"""
        try:
            model = self._create_stacking_ensemble(trial, data_size, n_features)

            if model is None:
                return -1e10

            X_array = np.asarray(X)
            y_array = np.asarray(y)

            n_splits = min(4, len(X) // 20)
            n_splits = max(2, n_splits)

            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

            fold_scores = []
            train_r2_scores = []
            val_r2_scores = []

            for train_idx, val_idx in kf.split(X_array):
                X_train_fold, X_val_fold = X_array[train_idx], X_array[val_idx]
                y_train_fold, y_val_fold = y_array[train_idx], y_array[val_idx]

                try:
                    model.fit(X_train_fold, y_train_fold)
                    y_pred_val = model.predict(X_val_fold)
                    y_pred_train = model.predict(X_train_fold)

                    fold_score = -mean_squared_error(y_val_fold, y_pred_val)
                    train_r2 = r2_score(y_train_fold, y_pred_train)
                    val_r2 = r2_score(y_val_fold, y_pred_val)

                    fold_scores.append(fold_score)
                    train_r2_scores.append(train_r2)
                    val_r2_scores.append(val_r2)

                except Exception as e:
                    print(f"⚠️ Stacking fold error: {e}")
                    fold_scores.append(-1e10)
                    train_r2_scores.append(0)
                    val_r2_scores.append(0)

            mean_val_r2 = np.mean(val_r2_scores)
            mean_train_r2 = np.mean(train_r2_scores)
            overfit_gap = mean_train_r2 - mean_val_r2

            if overfit_gap > 0.15:
                penalty = overfit_gap * 2
                mean_val_r2 -= penalty

            trial.set_user_attr("model_type", "stacking")
            trial.set_user_attr("overfit_gap", overfit_gap)
            trial.set_user_attr("train_r2", mean_train_r2)
            trial.set_user_attr("val_r2", mean_val_r2)

            return mean_val_r2

        except Exception as e:
            print(f"❌ Stacking evaluation failed: {e}")
            return -1e10

    def _create_final_stacking_ensemble(self, params, data_size, n_features):
        """Создание финального стекинг ансамбля с лучшими параметрами"""
        try:
            ensemble_type = params.get("stack_type", "tree_ensemble")
            meta_type = params.get("meta_type", "linear")
            n_folds = params.get("stack_folds", 5)

            base_model_configs = []

            if ensemble_type == "tree_boosters":
                base_models_types = ["lightgbm", "xgboost", "catboost"]
                print("🎯 Final: Gradient Boosters Only")

            elif ensemble_type == "tree_ensemble":
                base_models_types = ["random_forest", "extratrees", "lightgbm", "xgboost"]
                print("🎯 Final: Tree Ensemble (Bagging + Boosting)")

            elif ensemble_type == "mixed_bagging":
                base_models_types = ["random_forest", "extratrees", "lightgbm", "histgradientboosting"]
                print("🎯 Final: Mixed Bagging")

            elif ensemble_type == "gradient_boosters_only":
                base_models_types = ["lightgbm", "xgboost", "catboost", "histgradientboosting"]
                print("🎯 Final: All Gradient Boosters")

            elif ensemble_type == "bagging_ensemble":
                base_models_types = ["random_forest", "extratrees", "histgradientboosting"]
                if data_size in ["medium", "large"]:
                    base_models_types.append("ada_boost")
                print("🎯 Final: Bagging Ensemble")

            elif ensemble_type == "mixed_diverse":
                base_models_types = ["lightgbm", "random_forest", "xgboost", "extratrees"]
                if n_features < 50:
                    base_models_types.append("ridge")
                print("🎯 Final: Highly Diverse Mix")

            elif ensemble_type == "fast_ensemble":
                base_models_types = ["lightgbm", "histgradientboosting", "extratrees"]
                if data_size == "small":
                    base_models_types.append("random_forest")
                print("🎯 Final: Fast Ensemble")

            else:
                base_models_types = ["random_forest", "lightgbm", "ridge"]
                if data_size in ["medium", "large"]:
                    base_models_types.append("xgboost")
                print("🎯 Final: Conservative Mix")

            n_models_param = params.get("n_base_models")
            if n_models_param and n_models_param < len(base_models_types):
                # какие модели были выбраны в лучшем trial
                # берем первые n_models_param моделей
                base_models_types = base_models_types[:n_models_param]
                print(f"  Using subset of {n_models_param} models")

            for model_type in base_models_types:
                model_params = self._extract_final_params_for_model(params, model_type, "stack")
                base_model_configs.append({
                    'type': model_type,
                    'params': model_params
                })

            meta_config = {'type': meta_type, 'params': {}}

            if meta_type == "ridge":
                meta_config['params']['alpha'] = params.get("meta_alpha", 1.0)
            elif meta_type == "lightgbm":
                meta_config['params'].update({
                    'n_estimators': params.get("meta_lgb_n_est", 100),
                    'max_depth': params.get("meta_lgb_depth", 5),
                    'learning_rate': params.get("meta_lgb_lr", 0.1),
                    'verbose': -1,
                    'random_state': self.random_state
                })
            elif meta_type == "random_forest":
                meta_config['params'].update({
                    'n_estimators': params.get("meta_rf_n_est", 100),
                    'max_depth': params.get("meta_rf_depth", 5),
                    'min_samples_split': params.get("meta_rf_min_split", 2),
                    'random_state': self.random_state,
                    'n_jobs': -1
                })
            elif meta_type == "xgboost":
                meta_config['params'].update({
                    'n_estimators': params.get("meta_xgb_n_est", 100),
                    'max_depth': params.get("meta_xgb_depth", 5),
                    'learning_rate': params.get("meta_xgb_lr", 0.1),
                    'verbosity': 0,
                    'random_state': self.random_state
                })

            print(f"  Meta-model: {meta_type}, Folds: {n_folds}")
            print(f"  Total base models: {len(base_model_configs)}")

            return StackingEnsembleRegressor(
                base_model_configs=base_model_configs,
                meta_model_config=meta_config,
                n_folds=n_folds,
                random_state=self.random_state
            )

        except Exception as e:
            print(f"❌ Final stacking creation failed: {e}")
            base_model_configs = [
                {'type': 'random_forest', 'params': {'n_estimators': 100, 'random_state': self.random_state}},
                {'type': 'lightgbm', 'params': {'n_estimators': 100, 'verbose': -1, 'random_state': self.random_state}}
            ]
            meta_config = {'type': 'linear', 'params': {}}

            return StackingEnsembleRegressor(
                base_model_configs=base_model_configs,
                meta_model_config=meta_config,
                n_folds=5,
                random_state=self.random_state
            )

    def _extract_final_params_for_model(self, params, model_type, prefix):
        """Извлекает параметры для конкретной модели из общего словаря"""
        model_params = {}

        prefix_map = {
            "lightgbm": f"{prefix}_lgb_",
            "xgboost": f"{prefix}_xgb_",
            "random_forest": f"{prefix}_rf_",
            "extratrees": f"{prefix}_et_",
            "catboost": f"{prefix}_cb_",
            "histgradientboosting": f"{prefix}_hgb_",
            "ada_boost": f"{prefix}_ada_",
            "ridge": f"{prefix}_ridge_",
            "lasso": f"{prefix}_lasso_"
        }

        model_prefix = prefix_map.get(model_type, f"{prefix}_")

        for param_name, value in params.items():
            if param_name.startswith(model_prefix):
                clean_name = param_name[len(model_prefix):]
                model_params[clean_name] = value

        if 'random_state' not in model_params:
            model_params['random_state'] = self.random_state

        if model_type == 'lightgbm':
            if 'verbose' not in model_params:
                model_params['verbose'] = -1
        elif model_type == 'xgboost':
            if 'verbosity' not in model_params:
                model_params['verbosity'] = 0
        elif model_type == 'catboost':
            if 'verbose' not in model_params:
                model_params['verbose'] = False
        elif model_type in ['random_forest', 'extratrees']:
            if 'n_jobs' not in model_params:
                model_params['n_jobs'] = -1
        elif model_type in ['lasso', 'ridge', 'elasticnet']:
            if 'max_iter' not in model_params:
                model_params['max_iter'] = 1000

        return model_params