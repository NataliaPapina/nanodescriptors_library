from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.base import clone as sk_clone


class StackingEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, base_model_configs, meta_model_config, n_folds=5, random_state=0):
        """
        base_model_configs: список словарей с параметрами для создания моделей
        meta_model_config: словарь с параметрами для мета-модели
        """
        self.base_model_configs = base_model_configs
        self.meta_model_config = meta_model_config
        self.n_folds = n_folds
        self.random_state = random_state

    def _create_model_from_config(self, config):
        """Создает модель из конфигурации"""
        model_type = config.get('type')
        params = config.get('params', {})

        if model_type == 'lightgbm':
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**params)
        elif model_type == 'xgboost':
            from xgboost import XGBRegressor
            return XGBRegressor(**params)
        elif model_type == 'catboost':
            from catboost import CatBoostRegressor
            return CatBoostRegressor(**params)
        elif model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**params)
        elif model_type == 'extratrees':  # 🔥 ДОБАВЬ ЭТО
            from sklearn.ensemble import ExtraTreesRegressor
            return ExtraTreesRegressor(**params)
        elif model_type == 'svr':
            from sklearn.svm import SVR
            return SVR(**params)
        elif model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)
        elif model_type == 'ridge':
            from sklearn.linear_model import Ridge
            return Ridge(**params)
        elif model_type == 'histgradientboosting':
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(**params)
        elif model_type == 'ada_boost':
            from sklearn.ensemble import AdaBoostRegressor
            return AdaBoostRegressor(**params)
        elif model_type == 'lasso':
            from sklearn.linear_model import Lasso
            return Lasso(**params)
        elif model_type == 'elasticnet':
            from sklearn.linear_model import ElasticNet
            return ElasticNet(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X, y):
        """Обучение без клонирования - создаем новые модели"""
        self.base_models_ = []
        for config in self.base_model_configs:
            self.base_models_.append(self._create_model_from_config(config))

        self.meta_model_ = self._create_model_from_config(self.meta_model_config)

        if hasattr(y, 'values'):
            y_array = y.values
        else:
            y_array = np.asarray(y)

        X_array = np.asarray(X)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((X_array.shape[0], len(self.base_models_)))

        print(f"🏗️  Training safe stacking with {len(self.base_models_)} base models")

        for i, model in enumerate(self.base_models_):
            print(f"  Training base model {i + 1}: {type(model).__name__}")

            fold_models = []

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_array, y_array)):
                fold_model = self._create_model_from_config(self.base_model_configs[i])

                X_train_fold, X_val_fold = X_array[train_idx], X_array[val_idx]
                y_train_fold = y_array[train_idx]

                fold_model.fit(X_train_fold, y_train_fold)
                y_pred_val = fold_model.predict(X_val_fold)
                meta_features[val_idx, i] = y_pred_val

                fold_models.append(fold_model)

            model.fit(X_array, y_array)

        self.meta_model_.fit(meta_features, y_array)

        return self

    def predict(self, X):
        """Предсказание"""
        X_array = np.asarray(X)

        base_predictions = []
        for model in self.base_models_:
            pred = model.predict(X_array)
            base_predictions.append(pred)

        base_predictions_stack = np.column_stack(base_predictions)
        return self.meta_model_.predict(base_predictions_stack)