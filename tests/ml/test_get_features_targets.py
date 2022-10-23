import pytest
from hypothesis import given
from hypothesis import settings

import janitor.ml  # noqa: F401
from janitor.testing_utils.strategies import df_strategy


@pytest.mark.ml
@given(df=df_strategy())
@settings(deadline=None)
def test_get_features_targets(df):
    """Test one column returned as target and rest as features."""
    X, y = df.clean_names().get_features_targets(
        target_column_names="bell_chart"
    )
    assert X.shape[1] == 4
    assert len(y.shape) == 1


@pytest.mark.ml
@given(df=df_strategy())
@settings(deadline=None)
def test_get_features_targets_multi_features(df):
    """Test one column returned as target and two as features."""
    X, y = df.clean_names().get_features_targets(
        feature_column_names=["animals@#$%^", "cities"],
        target_column_names="bell_chart",
    )
    assert X.shape[1] == 2
    assert len(y.shape) == 1


@pytest.mark.ml
@given(df=df_strategy())
@settings(deadline=None)
def test_get_features_target_multi_columns(df):
    """Test two columns returned as target and rest as features."""
    X, y = df.clean_names().get_features_targets(
        target_column_names=["a", "bell_chart"]
    )
    assert X.shape[1] == 3
    assert y.shape[1] == 2
