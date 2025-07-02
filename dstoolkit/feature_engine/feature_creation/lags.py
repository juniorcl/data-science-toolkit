import warnings

import numpy  as np
import pandas as pd

from typing import List, Union
from itertools import combinations

from scipy.stats import skew, kurtosis


class SimpleLagTimeFeatureCreator:

    def __init__(self, windows=[2, 3, 4], functions=["mean", "median", "max", "min"], add_div=True, add_diff=True):
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

    def _calc_slope(self, x):
        """Calculates the slope using least squares."""
        x = np.asarray(x)
        
        if len(x) < 2 or np.isnan(x).any() or np.isinf(x).any() or np.all(x == x[0]):
            return np.nan
        try:
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        except np.linalg.LinAlgError:
            return np.nan

    def _create_lag_features(self, series):
        """Creates all lagged features for a time series."""
        self.features = {f'{series.name}_lag_{i}': series.shift(i) for i in range(1, max(self.windows) + 1)}

        for func in self.functions:
            func_operation = self._function_map[func]
            for win in self.windows:
                feature_values = self.features[f'{series.name}_lag_1'].rolling(window=win, min_periods=2).apply(func_operation, raw=True)
                self.features[f'{series.name}_{func}_last_{win}_lags'] = feature_values

    def _create_lag_div_features(self, series, max_lag=4):
        """
        Creates div between all lag combinations: lag_i/lag_j for i < j
        """
        for i, j in combinations(range(1, max_lag + 1), 2):
            pct_diff = self.features[f'{series.name}_lag_{i}'] / self.features[f'{series.name}_lag_{j}'].replace({0: np.nan})  # avoid division by zero
            self.features[f'{series.name}_div_lag_{i}_vs_lag_{j}'] = pct_diff

    def _create_lag_diff_features(self, series, max_lag=4):
        """
        Creates differences between all lag combinations: lag_i - lag_j for i < j
        """
        for i, j in combinations(range(1, max_lag + 1), 2):
            diff = self.features[f'{series.name}_lag_{i}'] - self.features[f'{series.name}_lag_{j}']
            self.features[f'{series.name}_diff_lag_{i}_vs_lag_{j}'] = diff

    def create(self, df, target, time):
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

        self._create_lag_features(df[target])

        if self.add_div:
            self._create_lag_div_features(df[target], max_lag=max(self.windows))

        if self.add_diff:
            self._create_lag_diff_features(df[target], max_lag=max(self.windows))
    
        return df.assign(**self.features)
    
class GroupedLagTimeFeatureCreator:

    def __init__(self, windows=[2, 3, 4], functions=["mean", "median", "max", "min"], add_div=True, add_diff=True):
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
                "Window size 1 was found. Windows with size equal to 1 are not recommended for lag/rolling calculations.",
                UserWarning
            )
            self.windows.remove(1)

    def _calc_slope(self, x):
        x = np.asarray(x)
        if len(x) < 2 or np.isnan(x).any() or np.isinf(x).any() or np.all(x == x[0]):
            return np.nan
        try:
            return np.polyfit(np.arange(len(x)), x, 1)[0]
        except np.linalg.LinAlgError:
            return np.nan

    def _create_group_features(self, group_df, target):
        series = group_df[target]
        max_lag = max(self.windows)
        lagged_features = {f'{target}_lag_{i}': series.shift(i) for i in range(1, max_lag + 1)}
        
        for func in self.functions:
            func_operation = self._function_map[func]
            for win in self.windows:
                result = lagged_features[f'{target}_lag_1'].rolling(window=win, min_periods=2).apply(func_operation, raw=True)
                lagged_features[f'{target}_{func}_last_{win}_lags'] = result

        if self.add_div or self.add_diff:
            for i, j in combinations(range(1, max_lag + 1), 2):
                lag_i = lagged_features[f'{target}_lag_{i}']
                lag_j = lagged_features[f'{target}_lag_{j}']

                if self.add_div:
                    lagged_features[f'{target}_div_lag_{i}_vs_lag_{j}'] = lag_i / lag_j.replace({0: np.nan})
                if self.add_diff:
                    lagged_features[f'{target}_diff_lag_{i}_vs_lag_{j}'] = lag_i - lag_j

        return group_df.assign(**lagged_features)

    def create(self, df, group_cols, target, time):
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        if target not in df.columns:
            raise ValueError(f"Column '{target}' not found in the DataFrame")

        df = df.sort_values(by=group_cols + [time])
        result = (
            df
            .groupby(group_cols, group_keys=False)
            .apply(lambda group: self._create_group_features(group, target))
        )

        return result.reset_index(drop=True)

