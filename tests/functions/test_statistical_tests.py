"""Tests for statistical tests on tabyls."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import janitor  # noqa: F401 - needed to register DataFrame methods


@pytest.fixture
def simple_df():
    """Create a simple DataFrame for tabyl statistical tests."""
    return pd.DataFrame(
        {
            "row": ["A", "A", "B", "B"],
            "col": ["X", "Y", "X", "Y"],
        }
    )


@pytest.mark.functions
def test_chisq_test_basic_tabyl_results(simple_df):
    """Test chi-squared output formatting on a tabyl."""
    tabyl_df = simple_df.tabyl("row", "col")
    result = tabyl_df.chisq_test()

    expected = stats.chi2_contingency(np.array([[1, 1], [1, 1]]))
    assert result.statistic == pytest.approx(expected.statistic)
    assert result.dof == expected.dof
    assert result.pvalue == pytest.approx(expected.pvalue)

    expected_table = pd.DataFrame(
        {"row": ["A", "B"], "X": [1.0, 1.0], "Y": [1.0, 1.0]}
    )
    pd.testing.assert_frame_equal(result.expected, expected_table)
    pd.testing.assert_frame_equal(result.observed, tabyl_df)
    assert np.allclose(result.residuals[["X", "Y"]].to_numpy(), 0.0)
    assert np.allclose(result.stdres[["X", "Y"]].to_numpy(), 0.0)


@pytest.mark.functions
def test_chisq_test_with_totals_warns(simple_df):
    """Test totals are removed with a warning."""
    tabyl_df = simple_df.tabyl("row", "col").adorn_totals("both")
    base_result = simple_df.tabyl("row", "col").chisq_test()
    with pytest.warns(UserWarning, match="Totals"):
        result = tabyl_df.chisq_test()
    assert result.statistic == pytest.approx(base_result.statistic)


@pytest.mark.functions
def test_chisq_test_requires_two_way(simple_df):
    """Test chisq_test raises on a 1-way tabyl."""
    one_way = simple_df.tabyl("row")
    with pytest.raises(ValueError, match="2-way tabyl"):
        one_way.chisq_test()


@pytest.mark.functions
def test_fisher_test_basic(simple_df):
    """Test Fisher's exact test output."""
    tabyl_df = simple_df.tabyl("row", "col")
    result = tabyl_df.fisher_test()
    expected = stats.fisher_exact(np.array([[1, 1], [1, 1]]))
    expected_stat = (
        expected.statistic if hasattr(expected, "statistic") else expected[0]
    )
    expected_p = expected.pvalue if hasattr(expected, "pvalue") else expected[1]
    assert result.statistic == pytest.approx(expected_stat)
    assert result.pvalue == pytest.approx(expected_p)


@pytest.mark.functions
def test_fisher_test_requires_2x2():
    """Test fisher_test raises on non-2x2 tables."""
    df = pd.DataFrame(
        {
            "row": ["A", "A", "B", "B", "C", "C"],
            "col": ["X", "Y", "X", "Y", "X", "Y"],
        }
    )
    tabyl_df = df.tabyl("row", "col")
    with pytest.raises(ValueError, match="2x2"):
        tabyl_df.fisher_test()
