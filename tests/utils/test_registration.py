"""Regressions for the shared pandas method metadata boundary."""

from inspect import signature

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor.registration import preserve_dataframe_metadata


class MetadataFrame(pd.DataFrame):
    """Frame with an application-owned metadata field."""

    _metadata = ["source"]

    @property
    def _constructor(self):
        """Keep the application frame type through pandas operations."""
        return MetadataFrame


@pytest.fixture
def frame():
    """A source frame with both pandas and subclass metadata."""
    df = MetadataFrame({"group": [1, 1], "item": [1, 2]})
    df.source = "application"
    df.attrs = {"nested": {"revision": 1}}
    return df


def test_metadata_boundary_supports_keyword_and_grouped_sources(frame):
    """Both invocation styles finalize from the original frame."""

    @preserve_dataframe_metadata
    def recreate(df):
        """Deliberately return a base frame."""
        return pd.DataFrame({"new": [3]})

    for result in (recreate(df=frame), recreate(frame.groupby("group"))):
        assert isinstance(result, MetadataFrame)
        assert result.source == "application"
        assert result.attrs == frame.attrs
        result.attrs["nested"]["revision"] = 2
        assert frame.attrs["nested"]["revision"] == 1


def test_operation_attributes_take_precedence(frame):
    """Intentional result attributes survive inherited metadata."""

    @preserve_dataframe_metadata
    def transform(df):
        """Deliberately attach operation-specific metadata."""
        result = pd.DataFrame({"value": [2]})
        result.attrs = {"nested": {"revision": 7}, "operation": "transform"}
        return result

    result = transform(frame)
    assert result.attrs == {"nested": {"revision": 7}, "operation": "transform"}
    assert result.source == frame.source


def test_explicit_result_subtype_is_not_demoted(frame):
    """A transformation's intentional return type remains authoritative."""

    class ResultFrame(pd.DataFrame):
        """An explicitly chosen result type."""

    @preserve_dataframe_metadata
    def operation(df):
        """Return a different, explicit DataFrame subtype."""
        return ResultFrame({"value": [2]})

    assert isinstance(operation(frame), ResultFrame)
    source = pd.DataFrame(frame)
    source.attrs = {"source": "base"}
    result = operation(source)
    assert isinstance(result, ResultFrame)
    assert result.attrs == source.attrs


def test_non_frame_results_and_in_place_returns_keep_identity(frame):
    """Tuples, Series, scalars, and explicit in-place returns are untouched."""
    for expected in (frame, frame["item"], (frame, frame), 7, None):

        def operation(df):
            """Return the selected object without reconstruction."""
            return expected

        assert preserve_dataframe_metadata(operation)(frame) is expected


def test_registration_preserves_signature_and_wraps_only_once(frame):
    """Dual method registrations do not duplicate metadata finalization."""

    def operation(df, *, option=True):
        """Exercise signature retention."""
        return pd.DataFrame(df)

    wrapped = preserve_dataframe_metadata(operation)
    assert signature(wrapped) == signature(operation)
    assert wrapped.__name__ == operation.__name__
    assert preserve_dataframe_metadata(wrapped) is wrapped


def test_registered_in_place_finance_method_keeps_identity(frame, monkeypatch):
    """Finance registration can be checked without external pricing services."""
    import janitor.finance

    monkeypatch.setattr(janitor.finance, "_inflate_currency", lambda *args: 1.5)
    result = frame.inflate_currency(
        "item", country="USA", currency_year=2020, to_year=2021
    )
    assert result is frame
    assert result.source == "application"
    assert result["item"].tolist() == [1.5, 3.0]


@pytest.mark.parametrize(
    "name,args",
    [
        ("complete", ("group", "item")),
        ("expand", ("group", "item")),
        ("pivot_longer", ()),
    ],
)
def test_registered_transformations_keep_metadata(frame, name, args):
    """Representative constructor-losing methods share one policy."""
    import janitor

    original = frame.copy(deep=True)
    bound = getattr(frame, name)(*args)
    direct = getattr(janitor, name)(frame, *args)
    for result in (bound, direct):
        assert isinstance(result, MetadataFrame)
        assert result.source == frame.source
        assert result.attrs == frame.attrs
    assert_frame_equal(frame, original)
