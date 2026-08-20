import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor.functions.conditional_join import _create_frame


def _mixed_frame(prefix):
    """Return columns whose missing-value behavior is dtype-sensitive."""
    return pd.DataFrame(
        {
            f"{prefix}_int": np.array([1, 2, 3], dtype=np.int64),
            f"{prefix}_bool": np.array([True, False, True]),
            f"{prefix}_nullable_int": pd.array([1, 2, 3], dtype="Int64"),
            f"{prefix}_nullable_bool": pd.array([True, False, True], dtype="boolean"),
            f"{prefix}_category": pd.Categorical(
                ["a", "b", "a"], categories=["a", "b"]
            ),
            f"{prefix}_string": pd.array(["a", "b", "c"], dtype="string"),
            f"{prefix}_datetime": pd.date_range("2020-01-01", periods=3),
            f"{prefix}_datetime_tz": pd.date_range("2020-01-01", periods=3, tz="UTC"),
            f"{prefix}_timedelta": pd.to_timedelta([1, 2, 3], unit="D"),
        }
    )


def _missing_rows(frame, length):
    """Build missing rows using public pandas reindexing behavior."""
    return frame.iloc[:0].reindex(range(length))


def _expected_frame(left, right, left_index, right_index, how):
    """Assemble the expected blocks independently of `_create_frame`."""
    left_unmatched = np.setdiff1d(np.arange(len(left)), left_index)
    right_unmatched = np.setdiff1d(np.arange(len(right)), right_index)

    def pair(left_rows, right_rows):
        return pd.concat(
            [
                left_rows.reset_index(drop=True),
                right_rows.reset_index(drop=True),
            ],
            axis=1,
        )

    matched = pair(left.iloc[left_index], right.iloc[right_index])
    left_only = pair(
        left.iloc[left_unmatched], _missing_rows(right, left_unmatched.size)
    )
    right_only = pair(
        _missing_rows(left, right_unmatched.size), right.iloc[right_unmatched]
    )
    parts = {
        "inner": [(matched, "both")],
        "left": [(matched, "both"), (left_only, "left_only")],
        "right": [(matched, "both"), (right_only, "right_only")],
        "outer": [
            (matched, "both"),
            (left_only, "left_only"),
            (right_only, "right_only"),
        ],
        "left_anti": [(left_only, "left_only")],
        "right_anti": [(right_only, "right_only")],
    }[how]
    output = pd.concat([part for part, _ in parts], ignore_index=True)
    labels = [label for part, label in parts for _ in range(len(part))]
    output["_merge"] = pd.Categorical(
        labels, categories=["left_only", "right_only", "both"]
    )
    return output


@pytest.mark.parametrize(
    "how", ["inner", "left", "right", "outer", "left_anti", "right_anti"]
)
@pytest.mark.parametrize(
    ("left_index", "right_index"),
    [
        (np.array([0, 0, 2]), np.array([0, 1, 1])),
        (np.array([], dtype=np.intp), np.array([], dtype=np.intp)),
    ],
    ids=["mixed_matches", "no_matches"],
)
def test_create_frame_preserves_mixed_dtypes(how, left_index, right_index):
    """Result assembly preserves values, row order, and extension dtypes."""
    left = _mixed_frame("left")
    right = _mixed_frame("right")
    expected = _expected_frame(left, right, left_index, right_index, how=how)
    actual = _create_frame(
        df=left,
        right=right,
        left_index=left_index,
        right_index=right_index,
        how=how,
        df_columns=slice(None),
        right_columns=slice(None),
        indicator=True,
        include_join_positions=False,
    )
    assert_frame_equal(actual, expected)
