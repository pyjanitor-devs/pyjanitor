import pytest

from janitor.utils import deprecated_kwargs


@pytest.mark.utils
@pytest.mark.parametrize(
    "arguments, message, func_kwargs, msg_expected",
    [
        (
            ["a"],
            "The keyword argument '{argument}' of '{func_name}' is deprecated",
            dict(a=1),
            "The keyword argument 'a' of 'simple_sum' is deprecated",
        ),
        (
            ["b"],
            "The keyword argument '{argument}' of '{func_name}' is deprecated",
            dict(b=2),
            "The keyword argument 'b' of 'simple_sum' is deprecated",
        ),
        (
            ["a", "b"],
            "The option '{argument}' of '{func_name}' is deprecated.",
            dict(a=1, b=2),
            "The option 'a' of 'simple_sum' is deprecated.",
        ),
        (
            ["b", "a"],
            "The keyword of function is deprecated.",
            dict(a=1, b=2),
            "The keyword of function is deprecated.",
        ),
    ],
)
def test_error(arguments, message, func_kwargs, msg_expected):
    @deprecated_kwargs(*arguments, message=message)
    def simple_sum(alpha, beta, a=0, b=0):
        return alpha + beta

    with pytest.raises(ValueError, match=msg_expected):
        simple_sum(1, 2, **func_kwargs)


@pytest.mark.utils
@pytest.mark.parametrize(
    "arguments, message, func_kwargs, msg_expected",
    [
        (
            ["a"],
            "The keyword argument '{argument}' of '{func_name}' is deprecated",
            dict(a=1),
            "The keyword argument 'a' of 'simple_sum' is deprecated",
        ),
        (
            ["b"],
            "The keyword argument '{argument}' of '{func_name}' is deprecated",
            dict(b=2),
            "The keyword argument 'b' of 'simple_sum' is deprecated",
        ),
        (
            ["a", "b"],
            "The option '{argument}' of '{func_name}' is deprecated.",
            dict(a=1, b=2),
            "The option 'a' of 'simple_sum' is deprecated.",
        ),
        (
            ["b", "a"],
            "The keyword of function is deprecated.",
            dict(a=1, b=2),
            "The keyword of function is deprecated.",
        ),
    ],
)
def test_warning(arguments, message, func_kwargs, msg_expected):
    @deprecated_kwargs(*arguments, message=message, error=False)
    def simple_sum(alpha, beta, a=0, b=0):
        return alpha + beta

    with pytest.warns(DeprecationWarning, match=msg_expected):
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
