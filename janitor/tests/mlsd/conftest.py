import pandas as pd
import numpy as np
import pytest

# import paso.pasoBase

from sklearn.datasets import fetch_california_housing


@pytest.fixture()
def Housing():
    dataset = fetch_california_housing()
    return pd.DataFrame(dataset.data, columns=dataset.feature_names)


arrb = [
    [True, False, 1],
    [False, True, 0],
    [False, 0, True],
    [1, True, 1],
    [False, True, False],
    [True, 0, 0],
    [True, False, 1],
    [False, True, 0],
    [False, 0, True],
    [1, True, 1],
    [False, True, False],
    [True, 0, 0],
    [True, False, 1],
    [False, True, 0],
    [False, 0, True],
    [1, True, 1],
    [False, True, False],
    [True, 0, 0],
]


@pytest.fixture()
def Xboolean():
    return pd.DataFrame(arrb)


arrob = np.array(
    [
        ["a1", "b", "c1"],
        ["a2", "b", "c2"],
        ["a3", "b", "c3"],
        ["a4", "b", "c4"],
        ["a5", "b", "c5"],
        ["a6", "b", "c6"],
        ["a7", "b", "c7"],
        ["a8", "b", "c8"],
        ["a9", "b", "c9"],
        ["a0", "b", "c01"],
        ["a01", "b", "c02"],
        ["a02", "b", "c03"],
    ]
)


@pytest.fixture()
def Xobject():
    return pd.DataFrame(arrob)


arrn = np.array(
    [
        [-4.2, -3.7, -8.9],
        [-3.1, -3.2, -0.5],
        [1.4, 0.9, 8.9],
        [5.8, 5.0, 2.4],
        [5.6, 7.8, 2.4],
        [0.1, 7.0, 0.2],
        [8.3, 1.9, 7.8],
        [3.8, 9.2, 2.8],
        [5.3, 5.7, 4.5],
        [6.8, 5.3, 3.2],
    ]
)


@pytest.fixture()
def Xneg():
    return arrn


arrz = np.array(
    [
        [0.0, 3.7, 8.9],
        [3.1, 3.2, 0.5],
        [1.4, 0.9, 8.9],
        [5.8, 5.0, 2.4],
        [5.6, 7.8, 2.4],
        [0.1, 7.0, 0.2],
        [8.3, 1.9, 7.8],
        [3.8, 9.2, 2.8],
        [5.3, 5.7, 4.5],
        [6.8, 5.3, 3.2],
    ]
)


@pytest.fixture()
def Xzero():
    return arrz


arr = np.array(
    [
        [4.2, 3.7, 8.9],
        [3.1, 3.2, 0.5],
        [1.4, 0.9, 8.9],
        [5.8, 5.0, 2.4],
        [5.6, 7.8, 2.4],
        [0.1, 7.0, 0.2],
        [8.3, 1.9, 7.8],
        [3.8, 9.2, 2.8],
        [5.3, 5.7, 4.5],
        [6.8, 5.3, 3.2],
    ]
)

arrx = np.array(
    [
        [4.2, 3.7, 8.9],
        [3.1, 3.2, 0.5],
        [1.4, 0.9, 8.9],
        [5.8, 5.0, 2.4],
        [5.6, 7.8, 2.4],
        [0.1, 7.0, 0.2],
        [8.3, 1.9, 7.8],
        [3.8, 9.2, 2.8],
        [5.3, 5.7, 4.5],
        [6.8, 5.3, 3.2],
    ]
)


@pytest.fixture()
def X():
    return arrx


arry = np.array([4, 1, 1, 1, 1, 4, 2, 3, 2, 1])


@pytest.fixture()
def y():
    return arry


arrs = np.array(["x", "y", "z"])


@pytest.fixture()
def ystr():
    return arrs


vn = np.array([4, 0, 1, 1, 1, 4, 2, 3, 2, 1])


@pytest.fixture()
def yz():
    return vn


@pytest.fixture()
def yn():
    return np.array([4, -1, 1, 1, 1, 4, 2, 3, 2, 1])


@pytest.fixture()
def z():
    return pd.DataFrame(Xxx(), columns=[cn()])


@pytest.fixture()
def cn():
    return ["Z", "k", "T"]


@pytest.fixture()
def cnv():
    return ["x", "y", "z"]


@pytest.fixture()
def cno():
    return ["RA", "B", "C"]


@pytest.fixture()
def df_type():
    return pd.DataFrame(arr, columns=cn)


@pytest.fixture()
def df_typeDup():
    df = pd.DataFrame(arr, columns=[cn])
    df["k"] = df["T"].values
    return df


@pytest.fixture()
def df_typeo():
    return pd.DataFrame(arr, columns=[cno])


@pytest.fixture()
def df_typeNA():
    return (pd.DataFrame(arr, columns=[cn])).replace(to_replace=5, value=np.nan)


@pytest.fixture()
def df_type_low_V():
    df = pd.DataFrame(arr, columns=[cn])
    df["lowV"] = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    return df


@pytest.fixture()
def df_type_low_V11():
    df = pd.DataFrame(arr)  # , columns=[cn])
    df["lowV"] = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    df.iloc[9, :] = [1, 1.1, 10.2]
    return df


@pytest.fixture()
def df_type_SV():
    return df_type_low_V11().replace(to_replace=0, value=1)


@pytest.fixture()
def df_small():
    df = pd.DataFrame(
        {
            "datetime_S_column": [
                "11/11/1906",
                "11/11/1906",
                "11/11/1906 12:13:14",
                "11/11/1906",
                "11/11/1906",
            ],
            "datetime_NY_column": [
                "11/11/1906",
                "11/11/1907",
                "11/11/1908",
                "11/11/1909",
                "11/11/1910",
            ],
            "datetime_ND_column": [
                "11/11/1906",
                "11/12/1907",
                "11/13/1908",
                "11/14/1909",
                "11/15/1910",
            ],
            "datetime_NA_column": [
                "11/11/1906",
                np.nan,
                "11/11/1908",
                "11/11/1909",
                "11/11/1910",
            ],
            "datetime_EU_column": [
                "21.01.1906",
                "21.11.1907",
                "14.11.1908",
                "13.11.1909",
                "11.10.1910",
            ],
            "obj_column": ["red", "blue", "green", "pink", np.nan],
            "booly": [True, False, True, False, True],
            "integer": [1, 2, 33, 44, 34],
            "float": [1.0, 2.0, 35.0, 46, 0.37],
        }
    )
    return df


@pytest.fixture()
def df_small_NFeatures(df_small):
    return df_small.shape[1]


@pytest.fixture()
def NComponentFeatures():
    dt_features = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Elapsed",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ]
    return len(dt_features)


@pytest.fixture()
def df_big():
    ld = [
        "11/11/1906",
        "11/11/1906",
        "11/11/1906 12:13:14",
        "11/11/1906",
        "11/11/1906",
    ] * 200000
    dfb = pd.DataFrame({"datetime_S_column": ld})
    return dfb


@pytest.fixture()
def df_big_dt():
    ld = [
        "11/11/1906",
        "11/11/1906",
        "11/11/1906 12:13:14",
        "11/11/1906",
        "11/11/1906",
    ] * 200000
    dfb = pd.DataFrame({"datetime_S_column": ld})
    return pd.DataFrame(
        pd.to_datetime(dfb["datetime_S_column"], infer_datetime_format=True)
    )


@pytest.fixture()
def df_small_NA(df_small):
    return df_small.replace(to_replace="11/11/1906", value=np.nan)


@pytest.fixture()
def df_small_no_NA(df_small):
    return df_small.replace(to_replace=np.nan, value="11/11/1906")


@pytest.fixture()
def City():
    from sklearn.datasets import load_boston

    boston = load_boston()
    City = pd.DataFrame(boston.data, columns=boston.feature_names)
    City["MEDV"] = boston.target
    return City.copy()


@pytest.fixture()
def flower():
    from sklearn.datasets import load_iris

    iris = load_iris()
    Flower = pd.DataFrame(iris.data, columns=iris.feature_names)
    Flower["TypeOf"] = iris.target
    return Flower.copy()


dates = [
    "6/7/05 7:00",
    "6/7/05 8:00",
    "6/7/05 9:00",
    "6/7/05 10:00",
    "6/7/05 11:00",
    "6/7/05 12:00",
    "6/7/05 13:00",
    "6/7/05 14:00",
    "6/7/05 15:00",
    "6/7/05 16:00",
    "6/7/05 17:00",
    "6/7/05 18:00",
    "6/7/05 19:00",
    "6/7/05 20:00",
    "6/7/05 21:00",
    "6/7/05 22:00",
    "6/7/05 23:00",
    "6/8/05 0:00",
    "6/8/05 1:00",
    "6/8/05 2:00",
    "6/8/05 3:00",
    "6/8/05 4:00",
    "6/8/05 5:00",
    "6/8/05 6:00",
    "6/8/05 7:00",
    "6/8/05 8:00",
    "6/8/05 9:00",
    "6/8/05 10:00",
    "6/8/05 11:00",
    "6/8/05 12:00",
    "6/8/05 13:00",
    "6/8/05 14:00",
    "6/8/05 15:00",
    "6/8/05 16:00",
    "6/8/05 17:00",
    "6/8/05 18:00",
    "6/8/05 19:00",
    "6/8/05 20:00",
    "6/8/05 21:00",
    "6/8/05 22:00",
    "6/8/05 23:00",
    "6/9/05 0:00",
    "6/9/05 1:00",
    "6/9/05 2:00",
    "6/9/05 3:00",
    "6/9/05 4:00",
    "6/9/05 5:00",
    "6/9/05 6:00",
    "6/9/05 7:00",
    "6/9/05 8:00",
    "6/9/05 9:00",
    "6/9/05 10:00",
    "6/9/05 11:00",
    "6/9/05 12:00",
    "6/9/05 13:00",
    "6/9/05 14:00",
    "6/9/05 15:00",
    "6/9/05 16:00",
    "6/9/05 17:00",
]

bits = [
    56718587433,
    76456162968,
    82534038485,
    88796995092,
    90247922345,
    90146117117,
    90457410673,
    89967660859,
    87211742250,
    73610634839,
    56695326238,
    48110601866,
    44406878766,
    43050199070,
    36961508495,
    31555022712,
    27467342825,
    25041832565,
    21160963384,
    19820358830,
    19088172849,
    23204098006,
    26290215100,
    42380697686,
    70062696507,
    80545779973,
    92844295517,
    96880965179,
    90932927706,
    91291134373,
    91713661478,
    90959467260,
    79385942062,
    62489604464,
    56243244765,
    50040519394,
    46874882633,
    41135776475,
    33502564447,
    28113068984,
    24938212968,
    21893717932,
    20344443618,
    19757922807,
    19588672617,
    21350728448,
    34965445638,
    58397356904,
    79198224742,
    89726734078,
    92701958472,
    89941544749,
    93225272251,
    93840519305,
    91765453742,
    84996841280,
    70779170417,
    53526087319,
    46784929778,
]


@pytest.fixture()
def df_internet_traffic():
    internet_traffic = pd.DataFrame(dates, columns=["date"])
    internet_traffic["bit"] = bits
    internet_traffic["byte"] = internet_traffic["bit"] / 8.0
    return internet_traffic.copy()
