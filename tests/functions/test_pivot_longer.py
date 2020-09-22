from itertools import product

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor import patterns

df_checks = pd.DataFrame(
    [
        {"region": "Pacific", "2007": 1039, "2009": 2587},
        {"region": "Southwest", "2007": 51, "2009": 176},
        {"region": "Rocky Mountains and Plains", "2007": 200, "2009": 338},
    ]
)


@pytest.fixture
def df_checks_output():
    return pd.DataFrame(
        {
            "region": [
                "Pacific",
                "Pacific",
                "Southwest",
                "Southwest",
                "Rocky Mountains and Plains",
                "Rocky Mountains and Plains",
            ],
            "year": [2007, 2009, 2007, 2009, 2007, 2009],
            "num_nests": [1039, 2587, 51, 176, 200, 338],
        }
    )


paired_columns_pattern = [
    (
        pd.DataFrame(
            [
                {
                    "id": 1,
                    "a1": "a",
                    "a2": "b",
                    "a3": "c",
                    "A1": "A",
                    "A2": "B",
                    "A3": "C",
                }
            ]
        ),
        pd.DataFrame(
            {
                "id": [1, 1, 1],
                "instance": [1, 2, 3],
                "a": ["a", "b", "c"],
                "A": ["A", "B", "C"],
            }
        ),
        "id",
        (".value", "instance"),
        r"(\w)(\d)",
    ),
    (
        pd.DataFrame(
            {
                "A1970": ["a", "b", "c"],
                "A1980": ["d", "e", "f"],
                "B1970": [2.5, 1.2, 0.7],
                "B1980": [3.2, 1.3, 0.1],
                "X": [-1.085631, 0.997345, 0.282978],
            }
        ),
        pd.DataFrame(
            {
                "X": [
                    -1.085631,
                    -1.085631,
                    0.997345,
                    0.997345,
                    0.282978,
                    0.282978,
                ],
                "year": [1970, 1980, 1970, 1980, 1970, 1980],
                "A": ["a", "d", "b", "e", "c", "f"],
                "B": [2.5, 3.2, 1.2, 1.3, 0.7, 0.1],
            }
        ),
        "X",
        (".value", "year"),
        "([A-Z])(.+)",
    ),
    (
        pd.DataFrame(
            {
                "id": ["A", "B", "C", "D", "E", "F"],
                "f_start": ["p", "i", "i", "p", "p", "i"],
                "d_start": [
                    "2018-01-01",
                    "2019-04-01",
                    "2018-06-01",
                    "2019-12-01",
                    "2019-02-01",
                    "2018-04-01",
                ],
                "f_end": ["p", "p", "i", "p", "p", "i"],
                "d_end": [
                    "2018-02-01",
                    "2020-01-01",
                    "2019-03-01",
                    "2020-05-01",
                    "2019-05-01",
                    "2018-07-01",
                ],
            }
        ),
        pd.DataFrame(
            [
                {"id": "A", "status": "start", "f": "p", "d": "2018-01-01"},
                {"id": "A", "status": "end", "f": "p", "d": "2018-02-01"},
                {"id": "B", "status": "start", "f": "i", "d": "2019-04-01"},
                {"id": "B", "status": "end", "f": "p", "d": "2020-01-01"},
                {"id": "C", "status": "start", "f": "i", "d": "2018-06-01"},
                {"id": "C", "status": "end", "f": "i", "d": "2019-03-01"},
                {"id": "D", "status": "start", "f": "p", "d": "2019-12-01"},
                {"id": "D", "status": "end", "f": "p", "d": "2020-05-01"},
                {"id": "E", "status": "start", "f": "p", "d": "2019-02-01"},
                {"id": "E", "status": "end", "f": "p", "d": "2019-05-01"},
                {"id": "F", "status": "start", "f": "i", "d": "2018-04-01"},
                {"id": "F", "status": "end", "f": "i", "d": "2018-07-01"},
            ]
        ),
        "id",
        (".value", "status"),
        "(.*)_(.*)",
    ),
]

paired_columns_sep = [
    (
        pd.DataFrame(
            {
                "index": [0, 1],
                "S_1": [1, 1],
                "S_2": [0, 1],
                "S_3": ["0", np.nan],
                "S_4": ["1", np.nan],
            }
        ),
        pd.DataFrame(
            {
                "index": [0, 0, 0, 0, 1, 1, 1, 1],
                "num": [1, 2, 3, 4, 1, 2, 3, 4],
                "S": [1, 0, 0, 1, 1, 1, np.nan, np.nan],
            }
        ),
        "index",
        (".value", "num"),
        "_",
    ),
    (
        pd.DataFrame(
            {
                "county": [1001, 1003, 1005],
                "area": [275, 394, 312],
                "pop_2006": [1037, 2399, 1638],
                "pop_2007": [1052, 2424, 1647],
                "pop_2008": [1102, 2438, 1660],
            }
        ),
        pd.DataFrame(
            {
                "county": [
                    1001,
                    1001,
                    1001,
                    1003,
                    1003,
                    1003,
                    1005,
                    1005,
                    1005,
                ],
                "area": [275, 275, 275, 394, 394, 394, 312, 312, 312],
                "year": [2006, 2007, 2008, 2006, 2007, 2008, 2006, 2007, 2008],
                "pop": [1037, 1052, 1102, 2399, 2424, 2438, 1638, 1647, 1660],
            }
        ),
        ["county", "area"],
        (".value", "year"),
        "_",
    ),
    (
        pd.DataFrame(
            {
                "family": [1, 2, 3, 4, 5],
                "dob_child1": [
                    "1998-11-26",
                    "1996-06-22",
                    "2002-07-11",
                    "2004-10-10",
                    "2000-12-05",
                ],
                "dob_child2": [
                    "2000-01-29",
                    np.nan,
                    "2004-04-05",
                    "2009-08-27",
                    "2005-02-28",
                ],
                "gender_child1": [1, 2, 2, 1, 2],
                "gender_child2": [2.0, np.nan, 2.0, 1.0, 1.0],
            }
        ),
        pd.DataFrame(
            [
                {
                    "family": 1,
                    "child": "child1",
                    "dob": "1998-11-26",
                    "gender": 1.0,
                },
                {
                    "family": 1,
                    "child": "child2",
                    "dob": "2000-01-29",
                    "gender": 2.0,
                },
                {
                    "family": 2,
                    "child": "child1",
                    "dob": "1996-06-22",
                    "gender": 2.0,
                },
                {
                    "family": 2,
                    "child": "child2",
                    "dob": np.nan,
                    "gender": np.nan,
                },
                {
                    "family": 3,
                    "child": "child1",
                    "dob": "2002-07-11",
                    "gender": 2.0,
                },
                {
                    "family": 3,
                    "child": "child2",
                    "dob": "2004-04-05",
                    "gender": 2.0,
                },
                {
                    "family": 4,
                    "child": "child1",
                    "dob": "2004-10-10",
                    "gender": 1.0,
                },
                {
                    "family": 4,
                    "child": "child2",
                    "dob": "2009-08-27",
                    "gender": 1.0,
                },
                {
                    "family": 5,
                    "child": "child1",
                    "dob": "2000-12-05",
                    "gender": 2.0,
                },
                {
                    "family": 5,
                    "child": "child2",
                    "dob": "2005-02-28",
                    "gender": 1.0,
                },
            ]
        ),
        "family",
        (".value", "child"),
        "_",
    ),
    (
        pd.DataFrame(
            [
                {
                    "id": "A",
                    "Q1r1_pepsi": 1,
                    "Q1r1_cola": 0,
                    "Q1r2_pepsi": 1,
                    "Q1r2_cola": 0,
                },
                {
                    "id": "B",
                    "Q1r1_pepsi": 0,
                    "Q1r1_cola": 0,
                    "Q1r2_pepsi": 1,
                    "Q1r2_cola": 1,
                },
                {
                    "id": "C",
                    "Q1r1_pepsi": 1,
                    "Q1r1_cola": 1,
                    "Q1r2_pepsi": 1,
                    "Q1r2_cola": 1,
                },
            ]
        ),
        pd.DataFrame(
            [
                {"id": "A", "brand": "pepsi", "Q1r1": 1, "Q1r2": 1},
                {"id": "A", "brand": "cola", "Q1r1": 0, "Q1r2": 0},
                {"id": "B", "brand": "pepsi", "Q1r1": 0, "Q1r2": 1},
                {"id": "B", "brand": "cola", "Q1r1": 0, "Q1r2": 1},
                {"id": "C", "brand": "pepsi", "Q1r1": 1, "Q1r2": 1},
                {"id": "C", "brand": "cola", "Q1r1": 1, "Q1r2": 1},
            ]
        ),
        "id",
        (".value", "brand"),
        "_",
    ),
    (
        pd.DataFrame(
            {
                "event": [1, 2],
                "url_1": ["g1", "g3"],
                "name_1": ["dc", "nyc"],
                "url_2": ["g2", "g4"],
                "name_2": ["sf", "la"],
            }
        ),
        pd.DataFrame(
            {
                "event": [1, 1, 2, 2],
                "item": [1, 2, 1, 2],
                "url": ["g1", "g2", "g3", "g4"],
                "name": ["dc", "sf", "nyc", "la"],
            }
        ),
        "event",
        (".value", "item"),
        "_",
    ),
]

multi_index_df = [
    pd.DataFrame(
        pd.DataFrame(
            {
                "name": {
                    (67, 56): "Wilbur",
                    (80, 90): "Petunia",
                    (64, 50): "Gregory",
                }
            }
        )
    ),
    pd.DataFrame(
        {
            ("name", "a"): {0: "Wilbur", 1: "Petunia", 2: "Gregory"},
            ("names", "aa"): {0: 67, 1: 80, 2: 64},
            ("more_names", "aaa"): {0: 56, 1: 90, 2: 50},
        }
    ),
    pd.DataFrame(
        {
            ("name", "a"): {
                (0, 2): "Wilbur",
                (1, 3): "Petunia",
                (2, 4): "Gregory",
            },
            ("names", "aa"): {(0, 2): 67, (1, 3): 80, (2, 4): 64},
            ("more_names", "aaa"): {(0, 2): 56, (1, 3): 90, (2, 4): 50},
        }
    ),
]

multiple_values_sep = [
    (
        pd.DataFrame(
            {
                "country": ["United States", "Russia", "China"],
                "vault_2012": [48.1, 46.4, 44.3],
                "floor_2012": [45.4, 41.6, 40.8],
                "vault_2016": [46.9, 45.7, 44.3],
                "floor_2016": [46.0, 42.0, 42.1],
            }
        ),
        pd.DataFrame(
            [
                {
                    "country": "United States",
                    "event": "vault",
                    "year": 2012,
                    "score": 48.1,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": 2012,
                    "score": 45.4,
                },
                {
                    "country": "United States",
                    "event": "vault",
                    "year": 2016,
                    "score": 46.9,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": 2016,
                    "score": 46.0,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": 2012,
                    "score": 46.4,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": 2012,
                    "score": 41.6,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": 2016,
                    "score": 45.7,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": 2016,
                    "score": 42.0,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": 2012,
                    "score": 44.3,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": 2012,
                    "score": 40.8,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": 2016,
                    "score": 44.3,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": 2016,
                    "score": 42.1,
                },
            ]
        ),
        "country",
        ("event", "year"),
        "_",
    ),
    (
        pd.DataFrame(
            {
                "country": ["United States", "Russia", "China"],
                "vault_2012_f": [
                    48.132,
                    46.36600000000001,
                    44.266000000000005,
                ],
                "vault_2012_m": [46.632, 46.86600000000001, 48.316],
                "vault_2016_f": [
                    46.86600000000001,
                    45.733000000000004,
                    44.332,
                ],
                "vault_2016_m": [45.865, 46.033, 45.0],
                "floor_2012_f": [45.36600000000001, 41.599, 40.833],
                "floor_2012_m": [45.266000000000005, 45.308, 45.133],
                "floor_2016_f": [45.998999999999995, 42.032, 42.066],
                "floor_2016_m": [43.757, 44.766000000000005, 43.799],
            }
        ),
        pd.DataFrame(
            [
                {
                    "country": "United States",
                    "event": "vault",
                    "year": 2012,
                    "gender": "f",
                    "score": 48.132,
                },
                {
                    "country": "United States",
                    "event": "vault",
                    "year": 2012,
                    "gender": "m",
                    "score": 46.632,
                },
                {
                    "country": "United States",
                    "event": "vault",
                    "year": 2016,
                    "gender": "f",
                    "score": 46.86600000000001,
                },
                {
                    "country": "United States",
                    "event": "vault",
                    "year": 2016,
                    "gender": "m",
                    "score": 45.865,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": 2012,
                    "gender": "f",
                    "score": 45.36600000000001,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": 2012,
                    "gender": "m",
                    "score": 45.266000000000005,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": 2016,
                    "gender": "f",
                    "score": 45.998999999999995,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": 2016,
                    "gender": "m",
                    "score": 43.757,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": 2012,
                    "gender": "f",
                    "score": 46.36600000000001,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": 2012,
                    "gender": "m",
                    "score": 46.86600000000001,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": 2016,
                    "gender": "f",
                    "score": 45.733000000000004,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": 2016,
                    "gender": "m",
                    "score": 46.033,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": 2012,
                    "gender": "f",
                    "score": 41.599,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": 2012,
                    "gender": "m",
                    "score": 45.308,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": 2016,
                    "gender": "f",
                    "score": 42.032,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": 2016,
                    "gender": "m",
                    "score": 44.766000000000005,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": 2012,
                    "gender": "f",
                    "score": 44.266000000000005,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": 2012,
                    "gender": "m",
                    "score": 48.316,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": 2016,
                    "gender": "f",
                    "score": 44.332,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": 2016,
                    "gender": "m",
                    "score": 45.0,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": 2012,
                    "gender": "f",
                    "score": 40.833,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": 2012,
                    "gender": "m",
                    "score": 45.133,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": 2016,
                    "gender": "f",
                    "score": 42.066,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": 2016,
                    "gender": "m",
                    "score": 43.799,
                },
            ]
        ),
        "country",
        ("event", "year", "gender"),
        "_",
    ),
]

multiple_values_pattern = [
    (
        pd.DataFrame(
            {
                "country": ["United States", "Russia", "China"],
                "vault2012": [48.132, 46.36600000000001, 44.266000000000005],
                "floor2012": [45.36600000000001, 41.599, 40.833],
                "vault2016": [46.86600000000001, 45.733000000000004, 44.332],
                "floor2016": [45.998999999999995, 42.032, 42.066],
            }
        ),
        pd.DataFrame(
            [
                {
                    "country": "United States",
                    "event": "vault",
                    "year": 2012,
                    "score": 48.132,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": 2012,
                    "score": 45.36600000000001,
                },
                {
                    "country": "United States",
                    "event": "vault",
                    "year": 2016,
                    "score": 46.86600000000001,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": 2016,
                    "score": 45.998999999999995,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": 2012,
                    "score": 46.36600000000001,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": 2012,
                    "score": 41.599,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": 2016,
                    "score": 45.733000000000004,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": 2016,
                    "score": 42.032,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": 2012,
                    "score": 44.266000000000005,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": 2012,
                    "score": 40.833,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": 2016,
                    "score": 44.332,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": 2016,
                    "score": 42.066,
                },
            ]
        ),
        "country",
        ("event", "year"),
        r"([A-Za-z]+)(\d+)",
    )
]


# https://community.rstudio.com/t/pivot-longer-on-multiple-column-sets-pairs/43958/10
paired_columns_no_index_pattern = [
    (
        pd.DataFrame(
            {
                "off_loc": ["A", "B", "C", "D", "E", "F"],
                "pt_loc": ["G", "H", "I", "J", "K", "L"],
                "pt_lat": [
                    100.07548220000001,
                    75.191326,
                    122.65134479999999,
                    124.13553329999999,
                    124.13553329999999,
                    124.01028909999998,
                ],
                "off_lat": [
                    121.271083,
                    75.93845266,
                    135.043791,
                    134.51128400000002,
                    134.484374,
                    137.962195,
                ],
                "pt_long": [
                    4.472089953,
                    -144.387785,
                    -40.45611048,
                    -46.07156181,
                    -46.07156181,
                    -46.01594293,
                ],
                "off_long": [
                    -7.188632000000001,
                    -143.2288569,
                    21.242563,
                    40.937416999999996,
                    40.78472,
                    22.905889000000002,
                ],
            }
        ),
        pd.DataFrame(
            [
                {
                    "set": "off",
                    "loc": "A",
                    "lat": 121.271083,
                    "long": -7.188632000000001,
                },
                {
                    "set": "pt",
                    "loc": "G",
                    "lat": 100.07548220000001,
                    "long": 4.472089953,
                },
                {
                    "set": "off",
                    "loc": "B",
                    "lat": 75.93845266,
                    "long": -143.2288569,
                },
                {
                    "set": "pt",
                    "loc": "H",
                    "lat": 75.191326,
                    "long": -144.387785,
                },
                {
                    "set": "off",
                    "loc": "C",
                    "lat": 135.043791,
                    "long": 21.242563,
                },
                {
                    "set": "pt",
                    "loc": "I",
                    "lat": 122.65134479999999,
                    "long": -40.45611048,
                },
                {
                    "set": "off",
                    "loc": "D",
                    "lat": 134.51128400000002,
                    "long": 40.937416999999996,
                },
                {
                    "set": "pt",
                    "loc": "J",
                    "lat": 124.13553329999999,
                    "long": -46.07156181,
                },
                {
                    "set": "off",
                    "loc": "E",
                    "lat": 134.484374,
                    "long": 40.78472,
                },
                {
                    "set": "pt",
                    "loc": "K",
                    "lat": 124.13553329999999,
                    "long": -46.07156181,
                },
                {
                    "set": "off",
                    "loc": "F",
                    "lat": 137.962195,
                    "long": 22.905889000000002,
                },
                {
                    "set": "pt",
                    "loc": "L",
                    "lat": 124.01028909999998,
                    "long": -46.01594293,
                },
            ]
        ),
        ("set", ".value"),
        "(.+)_(.+)",
    )
]

names_single_value = [
    (
        pd.DataFrame(
            {
                "event": [1, 2],
                "url_1": ["g1", "g3"],
                "name_1": ["dc", "nyc"],
                "url_2": ["g2", "g4"],
                "name_2": ["sf", "la"],
            }
        ),
        pd.DataFrame(
            {
                "event": [1, 1, 2, 2],
                "url": ["g1", "g2", "g3", "g4"],
                "name": ["dc", "sf", "nyc", "la"],
            }
        ),
        "event",
        "(.+)_.",
    ),
    (
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "x1": [4, 5, 6],
                "x2": [5, 6, 7],
                "y1": [7, 8, 9],
                "y2": [10, 11, 12],
            }
        ),
        pd.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "x": [4, 5, 5, 6, 6, 7],
                "y": [7, 10, 8, 11, 9, 12],
            }
        ),
        "id",
        "(.).",
    ),
]

index_labels = [pd.Index(["region"]), {"2007", "region"}]
column_labels = [{"region": 2007}, {"2007", "2009"}]
names_to_labels = [1, {12, "newnames"}]

index_does_not_exist = ["Region", [2007, "region"]]
column_does_not_exist = ["two thousand and seven", ("2007", 2009)]

index_type_checks = [
    (frame, index) for frame, index in product([df_checks], index_labels)
]
column_type_checks = [
    (frame, column_name)
    for frame, column_name in product([df_checks], column_labels)
]
names_to_type_checks = [
    (frame, names_to)
    for frame, names_to in product([df_checks], names_to_labels)
]

names_to_sub_type_checks = [
    (df_checks, (1, "rar")),
    (df_checks, [{"set"}, 20]),
]

index_presence_checks = [
    (frame, index)
    for frame, index in product([df_checks], index_does_not_exist)
]
column_presence_checks = [
    (frame, column_name)
    for frame, column_name in product([df_checks], column_does_not_exist)
]

names_sep_not_required = [
    (df_checks, "rar", "_"),
    (df_checks, ["blessed"], ","),
]

names_sep_type_check = [
    (df_checks, ["rar", "bar"], 1),
    (df_checks, ("rar", "ragnar"), ["\\d+"]),
]
names_pattern_type_check = [
    (df_checks, "rar", 1),
    (df_checks, ["rar"], ["\\d+"]),
]


@pytest.mark.parametrize("df,index", index_type_checks)
def test_type_index(df, index):
    """Raise TypeError if wrong type is provided for the `index`.'"""
    with pytest.raises(TypeError):
        df.pivot_longer(index=index)


@pytest.mark.parametrize("df,column", column_type_checks)
def test_type_column_names(df, column):
    """Raise TypeError if wrong type is provided for `column_names`.'"""
    with pytest.raises(TypeError):
        df.pivot_longer(column_names=column)


@pytest.mark.parametrize("df,names_to", names_to_type_checks)
def test_type_names_to(df, names_to):
    """Raise TypeError if wrong type is provided for `names_to`."""
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to)


@pytest.mark.parametrize("df,names_to", names_to_sub_type_checks)
def test_subtype_names_to(df, names_to):
    """
    Raise TypeError if `names_to` is a list/tuple and the wrong type is 
    provided for entries in `names_to`.
    """
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to)


@pytest.mark.parametrize("df,index", index_presence_checks)
def test_presence_index(df, index):
    """Raise ValueError if labels in `index` do not exist."""
    with pytest.raises(ValueError):
        df.pivot_longer(index=index)


@pytest.mark.parametrize("df,column", column_presence_checks)
def test_presence_columns(df, column):
    """Raise ValueError if labels in `column_names` do not exist."""
    with pytest.raises(ValueError):
        df.pivot_longer(column_names=column)


@pytest.mark.parametrize("df,names_to, names_sep", names_sep_not_required)
def test_name_sep_names_to_len(df, names_to, names_sep):
    """
    Raise ValueError if the `names_to` is a string, or `names_to` is a
    list/tuple, its length is one, and `names_sep` is provided."""
    with pytest.raises(ValueError):
        df.pivot_longer(names_to=names_to, names_sep=names_sep)


@pytest.mark.parametrize("df,names_to, names_sep", names_sep_type_check)
def test_name_sep_wrong_type(df, names_to, names_sep):
    """
    Raise TypeError if the wrong type provided for `names_sep`."""
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to, names_sep=names_sep)

@pytest.mark.parametrize(
    "df,names_to, names_pattern", names_pattern_type_check
)
def test_name_pattern_wrong_type(df, names_to, names_pattern):
    """
    Raise TypeError if the wrong type provided for `names_pattern`."""
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to, names_pattern=names_pattern)

@pytest.mark.parametrize("df", multi_index_df)
def test_warning_multi_index(df):
    """Raise Warning if dataframe is a MultiIndex."""
    with pytest.warns(UserWarning):
        df.pivot_longer()

def test_both_names_sep_and_pattern():
    """Raise ValueError if both `names_sep` and `names_pattern` is provided."""
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            names_to=["rar", "bar"], names_sep="-", names_pattern=r"\\d+"
        )


def test_values_to():
    """Raise TypeError if the wrong type is provided for `values_to`."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(values_to=["salvo"])


def test_pivot_no_args_passed():
    """Test output if no arguments are passed."""
    df_no_args = pd.DataFrame({"name": ["Wilbur", "Petunia", "Gregory"]})
    df_no_args_output = pd.DataFrame(
        {
            "variable": ["name", "name", "name"],
            "value": ["Wilbur", "Petunia", "Gregory"],
        }
    )
    result = df_no_args.pivot_longer()

    assert_frame_equal(result, df_no_args_output)


def test_pivot_index_only(df_checks_output):
    """Test output if only `index` is passed."""
    result = df_checks.pivot_longer(
        index="region", names_to="year", values_to="num_nests"
    )
    assert_frame_equal(result, df_checks_output)


def test_pivot_column_only(df_checks_output):
    """Test output if only `column_names` is passed."""
    result = df_checks.pivot_longer(
        column_names=["2007", "2009"], names_to="year", values_to="num_nests"
    )
    assert_frame_equal(result, df_checks_output)


def test_pivot_index_patterns_only(df_checks_output):
    """Test output if the `patterns` function is passed to `index`."""
    result = df_checks.pivot_longer(
        index=patterns(r"[^\d+]"), names_to="year", values_to="num_nests"
    )
    assert_frame_equal(result, df_checks_output)


def test_pivot_columns_patterns_only(df_checks_output):
    """Test output if the `patterns` function is passed to `column_names`."""
    result = df_checks.pivot_longer(
        column_names=patterns(r"\d+"), names_to="year", values_to="num_nests"
    )
    assert_frame_equal(result, df_checks_output)


@pytest.mark.parametrize(
    "df_in,df_out,index,names_to,names_pattern", paired_columns_pattern
)
def test_extract_column_names_pattern(
    df_in, df_out, index, names_to, names_pattern
):
    """
    Test output if `.value` is in the `names_to` argument and
    names_pattern is used.
    """
    result = df_in.pivot_longer(
        index=index, names_to=names_to, names_pattern=names_pattern
    )
    assert_frame_equal(result, df_out)


@pytest.mark.parametrize(
    "df_in,df_out,index,names_to,names_sep", paired_columns_sep
)
def test_extract_column_names_sep(df_in, df_out, index, names_to, names_sep):
    """
    Test output if `.value` is in the `names_to` argument and names_sep
    is used.
    """
    result = df_in.pivot_longer(
        index=index, names_to=names_to, names_sep=names_sep
    )
    assert_frame_equal(result, df_out)

@pytest.mark.parametrize(
    "df_in,df_out,index,names_to,names_sep", multiple_values_sep
)
def test_multiple_values_sep(df_in, df_out, index, names_to, names_sep):
    """
    Test function to extract multiple columns, using the `names_to` and
    names_sep arguments.
    """
    result = df_in.pivot_longer(
        index=index, names_to=names_to, names_sep=names_sep, values_to="score"
    )
    assert_frame_equal(result, df_out)


@pytest.mark.parametrize(
    "df_in,df_out,index,names_to,names_pattern", multiple_values_pattern
)
def test_multiple_values_pattern(
    df_in, df_out, index, names_to, names_pattern
):
    """
    Test function to extract multiple columns, using the `names_to` and
    names_pattern arguments.
    """
    result = df_in.pivot_longer(
        index=index,
        names_to=names_to,
        names_pattern=names_pattern,
        values_to="score",
    )
    assert_frame_equal(result, df_out)


@pytest.mark.parametrize(
    "df_in,df_out,names_to,names_pattern", paired_columns_no_index_pattern
)
def test_paired_columns_no_index_pattern(
    df_in, df_out, names_to, names_pattern
):
    """
    Test function where `.value` is in the `names_to` argument, names_pattern
    is used and no index is supplied.
    """
    result = df_in.pivot_longer(names_to=names_to, names_pattern=names_pattern)
    assert_frame_equal(result, df_out)


@pytest.mark.parametrize(
    "df_in,df_out,index, names_pattern", names_single_value
)
def test_single_value(df_in, df_out, index, names_pattern):
    """Test function where names_to is a string and == `.value`."""
    result = df_in.pivot_longer(
        index=index, names_to=".value", names_pattern=names_pattern
    )
    assert_frame_equal(result, df_out)
