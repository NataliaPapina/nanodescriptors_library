import pandas as pd
import numpy as np
from typing import List, Optional, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OrdinalEncoder
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder
from sklearn.feature_selection import VarianceThreshold


class DataPreprocessor:
    def __init__(
        self,
        target_column: str,
        columns_to_drop: Optional[List[str]] = None,
        drop_nan_threshold: float = 0.5,
        encoding: Literal['onehot', 'ordinal', 'target', 'none'] = 'onehot',
        scaling: Literal['standard', 'minmax', 'robust', 'none'] = 'standard',
        nan_strategy: Literal['mean', 'median', 'mode', 'drop'] = 'mean',
        use_knn_imputer: bool = False,
    ):
        self.target_column = target_column
        self.columns_to_drop = columns_to_drop or []
        self.drop_nan_threshold = drop_nan_threshold
        self.encoding = encoding
        self.scaling = scaling
        self.nan_strategy = nan_strategy
        self.use_knn_imputer = use_knn_imputer
        self.encoder = None
        self.scaler = None
        self.removed_columns = []
        self.high_corr_columns_to_drop = []
        self.selector = None

    def _convert_special_strings_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include=['object']).columns:
            mask = df[col].isin(['-', '—', 'no'])
            if mask.any():
                df.loc[mask, col] = np.nan
                try:
                    df[col] = df[col].astype(float)
                except Exception:
                    pass
        return df

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.nan_strategy == 'drop':
            return df.dropna()

        if self.use_knn_imputer:
            df_num = df._get_numeric_data()
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(df_num)
            df_imputed = pd.DataFrame(imputed_data, columns=df_num.columns, index=df.index)
            df[df_num.columns] = df_imputed
        else:
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype.kind in 'biufc':
                        if self.nan_strategy == 'mean':
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif self.nan_strategy == 'median':
                            df[col].fillna(df[col].median(), inplace=True)
                    else:
                        mode = df[col].mode()
                        if not mode.empty:
                            df[col] = df[col].fillna(mode[0])
                        else:
                            df[col] = df[col].fillna("unknown")
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except Exception:
                pass

        return df

    def _final_fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype.kind in 'biufc':
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    mode = df[col].mode()
                    if not mode.empty:
                        df[col] = df[col].fillna(mode[0])
                    else:
                        df[col] = df[col].fillna("unknown")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop_duplicates()

        df.drop(columns=self.columns_to_drop, errors="ignore", inplace=True)
        self.removed_columns.extend(self.columns_to_drop)

        df.dropna(subset=[self.target_column], inplace=True)

        df = self._convert_special_strings_to_nan(df)

        for col in df.columns:
            if df[col].isna().mean() > self.drop_nan_threshold:
                df.drop(columns=col, inplace=True)
                self.removed_columns.append(col)

        df = self._fill_missing(df)

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.drop(
            self.target_column, errors="ignore"
        )

        if self.encoding == "onehot":
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        elif self.encoding == "ordinal":
            self.encoder = OrdinalEncoder()
            df[cat_cols] = self.encoder.fit_transform(df[cat_cols])
        elif self.encoding == "target":
            self.encoder = TargetEncoder(cols=cat_cols)
            df[cat_cols] = self.encoder.fit_transform(df[cat_cols], df[self.target_column])
        elif self.encoding != "none":
            raise ValueError(f"Unknown encoding: {self.encoding}")

        num_cols = df.select_dtypes(include=["number"]).columns.drop(self.target_column, errors="ignore")

        if self.scaling == "standard":
            self.scaler = StandardScaler()
        elif self.scaling == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaling == "robust":
            self.scaler = RobustScaler()
        elif self.scaling != "none":
            raise ValueError(f"Unknown scaling: {self.scaling}")

        if self.scaler is not None:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])

        df = self._final_fill_missing(df)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.drop(columns=self.removed_columns, errors="ignore", inplace=True)
        df.dropna(subset=[self.target_column], inplace=True)
        df = self._convert_special_strings_to_nan(df)
        df = self._fill_missing(df)

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.drop(self.target_column, errors="ignore")
        if len(cat_cols) > 0:
            if self.encoding == "ordinal" and self.encoder is not None:
                df[cat_cols] = self.encoder.transform(df[cat_cols])
            elif self.encoding == "target" and self.encoder is not None:
                df[cat_cols] = self.encoder.transform(df[cat_cols])
            elif self.encoding == "onehot":
                df = pd.get_dummies(df, drop_first=True)

        num_cols = df.select_dtypes(include=["number"]).columns.drop(self.target_column, errors="ignore")

        if len(num_cols) > 0 and self.scaler is not None:
            df[num_cols] = self.scaler.transform(df[num_cols])

        df = self._final_fill_missing(df)

        return df
