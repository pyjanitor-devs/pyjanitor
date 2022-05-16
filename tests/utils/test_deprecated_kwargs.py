import pytest

from janitor.utils import deprecated_kwargs


@pytest.mark.utils
@pytest.mark.parametrize(
    "arguments, func_kwargs",
    [
        (["a"], dict(a=1)),
        (["b"], dict(b=2)),
        (["a", "b"], dict(a=1, b=2)),
        (["b", "a"], dict(a=1, b=2)),
    ],
)
def test_error(arguments, func_kwargs):
    @deprecated_kwargs(*arguments)
    def simple_sum(alpha, beta, a=0, b=0):
        return alpha + beta

    with pytest.raises(ValueError):
        simple_sum(1, 2, **func_kwargs)


@pytest.mark.utils
@pytest.mark.parametrize(
    "arguments, func_args, expected",
    [
        (["a"], [0, 0], 0),
        (["b"], [1, 1], 2),
        (["a", "b"], [0, 1], 1),
        (["b", "a"], [0, 1], 1),
    ],
)
def test_without_error(arguments, func_args, expected):
    @deprecated_kwargs(*arguments)
    def simple_sum(alpha, beta, a=0, b=0):
        return alpha + beta

    assert simple_sum(*func_args) == expected
