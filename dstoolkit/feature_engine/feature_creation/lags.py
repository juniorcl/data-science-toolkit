import warnings

import numpy  as np
import pandas as pd

from typing import List, Dict

from scipy.stats import skew, kurtosis


class SimpleLagTimeFeatureCreator:

    def __init__(self, windows: List[int] = [2, 3, 4], functions: List[str] = ["mean", "median", "max", "min"], add_div: bool = True, add_diff: bool = True):
        """
        Initiate the lag feature creator.

        Args:
            windows: List of window sizes for calculating statistics
            functions: List of functions to be applied (mean, median, max, min, etc.)
        """
        self.windows = windows
        self.functions = functions
        self.add_div = add_div
        self.add_diff = add_diff

        self._function_map = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min,
            'sum': np.sum,
            'std': lambda x: np.std(x, ddof=1),
            'kurt': lambda x: kurtosis(x, bias=False),
            'skew': lambda x: skew(x, bias=False),
            'slope': self._calc_slope
        }

        if 1 in self.windows:
            warnings.warn(
                "Window size 1 was found. Windows with size equal to 1 are not recommended for lag/rolling calculations, because they do not aggregate temporal information.",
                UserWarning
            )
            self.windows.remove(1)

    def _calc_slope(self, x: np.ndarray) -> float:
        """Calculates the slope using least squares."""
        return np.polyfit(np.arange(len(x)), x, 1)[0]

    def _create_lag_features(self, series: pd.Series) -> Dict[str, pd.Series]:
        """Creates all lagged features for a time series."""
        lag_one = series.shift(1)
        features = {f'{series.name}_sum_1_lag': lag_one}

        valid_funcs = [func for func in self.functions if func in self._function_map]

        for func in valid_funcs:
            func_operation = self._function_map[func]
            for win in self.windows:
                feature_values = lag_one.rolling(window=win, min_periods=2).apply(func_operation, raw=True)
                features[f'{series.name}_{func}_{win}_lag'] = feature_values

        return features

    def _create_lag_div_features(self, series: pd.Series, max_lag: int = 4) -> Dict[str, pd.Series]:
        """
        Creates div between all lag combinations: lag_i/lag_j for i < j
        """
        features = {}
        lags = {i: series.shift(i) for i in range(1, max_lag + 1)}

        for i in range(1, max_lag):
            
            for j in range(i + 1, max_lag + 1):
                
                pct_diff = lags[i] / lags[j].replace({0: np.nan})  # avoid division by zero
                features[f'{series.name}_div_lag_{i}_vs_{j}'] = pct_diff

        return features

    def _create_lag_diff_features(self, series: pd.Series, max_lag: int = 4) -> Dict[str, pd.Series]:
        """
        Creates differences between all lag combinations: lag_i - lag_j for i < j
        """
        features = {}
        lags = {i: series.shift(i) for i in range(1, max_lag + 1)}

        for i in range(1, max_lag):
            for j in range(i + 1, max_lag + 1):
                diff = lags[i] - lags[j]
                features[f'{series.name}_diff_lag_{i}_vs_{j}'] = diff

        return features

    def create(self, df: pd.DataFrame, target: str, time: str) -> pd.DataFrame:
        """
        Calculates all features with temporal lag for the target column.

        Args:
            df: Input DataFrame
            target: Name of the target column for feature calculation
            time: Name of the column with temporal data

        Returns:
            DataFrame with the new features added
        """
        if target not in df.columns:
            raise ValueError(f"Coluna '{target}' não encontrada no DataFrame")

        df = df.sort_values(by=time)

        lag_features = self._create_lag_features(df[target])

        if self.add_div:
            max_lag = max(self.windows) if self.windows else 4
            pct_diff_features = self._create_lag_div_features(df[target], max_lag=max_lag)
            lag_features.update(pct_diff_features)

        if self.add_diff:
            max_lag = max(self.windows) if self.windows else 4
            diff_features = self._create_lag_diff_features(df[target], max_lag=max_lag)
            lag_features.update(diff_features)

        return df.assign(**lag_features)


class GroupedLagTimeFeatureCreator:
    
    def __init__(
        self,
        windows: List[int] = [2, 3, 4],
        functions: List[str] = ["mean", "median", "max", "min"],
        add_div: bool = True,
        add_diff: bool = True
    ):
        """
        Initialize the lag feature creator.

        Args:
            windows: List of window sizes for rolling statistics.
            functions: List of statistical functions to apply.
            add_div: Whether to add division-based lag features.
            add_diff: Whether to add difference-based lag features.
        """
        self.windows = windows
        self.functions = functions
        self.add_div = add_div
        self.add_diff = add_diff

        self._function_map = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min,
            'sum': np.sum,
            'std': lambda x: np.std(x, ddof=1),
            'kurt': lambda x: kurtosis(x, bias=False),
            'skew': lambda x: skew(x, bias=False),
            'slope': self._calc_slope
        }

        if 1 in self.windows:
            warnings.warn(
                "Window size 1 found. This is not recommended as it does not aggregate temporal information.",
                UserWarning
            )
            self.windows.remove(1)

    def _calc_slope(self, x: np.ndarray) -> float:
        """Compute the slope of the series using linear regression (least squares)."""
        return np.polyfit(np.arange(len(x)), x, 1)[0]

    def _create_lag_features(self, series: pd.Series) -> Dict[str, pd.Series]:
        """Generate rolling window lag features for a single series."""
        lag_one = series.shift(1)
        features = {f'{series.name}_sum_1_lag': lag_one}

        valid_funcs = [func for func in self.functions if func in self._function_map]

        for func in valid_funcs:
            func_operation = self._function_map[func]
            for win in self.windows:
                feature_values = lag_one.rolling(window=win, min_periods=2).apply(func_operation, raw=True)
                features[f'{series.name}_{func}_{win}_lag'] = feature_values

        return features

    def _create_lag_div_features(self, series: pd.Series, max_lag: int = 4) -> Dict[str, pd.Series]:
        """Create features based on the division between lagged values."""
        features = {}
        lags = {i: series.shift(i) for i in range(1, max_lag + 1)}

        for i in range(1, max_lag):
            for j in range(i + 1, max_lag + 1):
                pct_diff = lags[i] / lags[j].replace({0: np.nan})
                features[f'{series.name}_div_lag_{i}_vs_{j}'] = pct_diff

        return features

    def _create_lag_diff_features(self, series: pd.Series, max_lag: int = 4) -> Dict[str, pd.Series]:
        """Create features based on the difference between lagged values."""
        features = {}
        lags = {i: series.shift(i) for i in range(1, max_lag + 1)}

        for i in range(1, max_lag):
            for j in range(i + 1, max_lag + 1):
                diff = lags[i] - lags[j]
                features[f'{series.name}_diff_lag_{i}_vs_{j}'] = diff

        return features

    def create(self, df: pd.DataFrame, target: str, time: str, groupby_col: str) -> pd.DataFrame:
        """
        Apply lag-based feature engineering on a grouped time series.

        Args:
            df: Input DataFrame containing the time series data.
            target: Name of the target column for lag computation.
            time: Name of the time column used to sort data.
            groupby_col: Column name used for grouping (e.g., source/category).

        Returns:
            A new DataFrame including the generated lag features.
        """
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")

        if groupby_col not in df.columns:
            raise ValueError(f"Group-by column '{groupby_col}' not found in DataFrame")

        # Sort by group and time
        df = df.sort_values(by=[groupby_col, time])
        output_df = df.copy()

        # Process each group independently
        for group, group_df in df.groupby(groupby_col):
            lag_feats = self._create_lag_features(group_df[target])

            if self.add_div:
                max_lag = max(self.windows) if self.windows else 4
                lag_feats.update(self._create_lag_div_features(group_df[target], max_lag=max_lag))

            if self.add_diff:
                max_lag = max(self.windows) if self.windows else 4
                lag_feats.update(self._create_lag_diff_features(group_df[target], max_lag=max_lag))

            # Assign lag features to the correct rows using original indices
            for col, values in lag_feats.items():
                output_df.loc[group_df.index, col] = values

        return output_df