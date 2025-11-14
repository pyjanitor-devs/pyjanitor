from __future__ import annotations
from typing import Iterable, Optional, Union, Callable
import pandas as pd, numpy as np

def _mad(series: pd.Series) -> float:
    med = series.median(skipna=True)
    return (series.sub(med).abs()).median(skipna=True)

def scale_mad(
    df: pd.DataFrame,
    columns: Optional[Union[Iterable[str], Callable[[pd.DataFrame], Iterable[str]]]] = None,
    clip: Optional[float] = None,
    zero_mad: str = "skip",  # 'skip' | 'one' | 'raise'
    suffix: Optional[str] = None,
) -> pd.DataFrame:
    """Robustly scale numeric columns using Median and MAD."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    out = df.copy()
    if columns is None:
        cols = out.select_dtypes(include=[np.number]).columns
    elif callable(columns):
        cols = list(columns(out))
    else:
        cols = list(columns)
    for col in cols:
        if col not in out.columns:
            continue
        s = out[col]
        if not np.issubdtype(s.dtype, np.number):
            continue
        med = s.median(skipna=True)
        mad = _mad(s)
        if mad == 0 or np.isnan(mad):
            if zero_mad == "skip":
                scaled = s
            elif zero_mad == "one":
                scaled = s - med
            else:
                raise ValueError(f"MAD is zero for column '{col}'")
        else:
            scaled = (s - med) / (mad * 1.4826)
        if clip is not None:
            scaled = scaled.clip(-clip, clip)
        out[f"{col}{suffix}" if suffix else col] = scaled
    return out
