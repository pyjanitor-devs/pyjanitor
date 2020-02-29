import pandas as pd
import pytest


@pytest.mark.functions
def test_update_where(dataframe):
    """
    Test that it accepts conditional parameters
    """
    pd.testing.assert_frame_equal(
        dataframe.update_where(
            (dataframe["decorated-elephant"] == 1)
            & (dataframe["animals@#$%^"] == "rabbit"),
            "cities",
            "Durham",
        ),
        dataframe.replace("Cambridge", "Durham"),
    )
