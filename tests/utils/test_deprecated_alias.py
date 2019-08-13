import pytest

from janitor.utils import deprecated_alias


@deprecated_alias(a="alpha", b="beta")
def simple_sum(alpha, beta):
    gamma = alpha + beta
    return gamma


@pytest.mark.utils
def test_old_aliases():
    """
    Using old aliases should  result in `DeprecationWarning`
    """
    with pytest.warns(DeprecationWarning):
        simple_sum(a=2, b=6)


@pytest.mark.utils
def test_new_aliases():
    """
    Using new aliases should not result in errors or warnings
    """
    with pytest.warns(None) as record:
        simple_sum(alpha=2, beta=6)
    assert not record.list

    assert simple_sum(alpha=2, beta=6)


@pytest.mark.utils
def test_mixed_aliases():
    """
    Using mixed aliases should result in errors
    """
    with pytest.raises(TypeError):
        assert simple_sum(alpha=2, beta=6, a=5)
