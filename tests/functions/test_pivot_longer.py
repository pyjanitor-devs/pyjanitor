import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import janitor

# df1 = pd.DataFrame(
#    [
#        {"region": "Pacific", "2007": 1039, "2009": 2587},
#        {"region": "Southwest", "2007": 51, "2009": 176},
#        {"region": "Rocky Mountains and Plains", "2007": 200, "2009": 338},
#    ]
# )

# print(df1.pivot_longer(names_to=["year"], index=janitor.patterns(r"[^\d+]")))


# df2 = pd.DataFrame(
#     {
#         "country": ["United States", "Russia", "China"],
#         "vault_2012": [48.132, 46.36600000000001, 44.266000000000005],
#         "floor_2012": [45.36600000000001, 41.599, 40.833],
#         "vault_2016": [46.86600000000001, 45.733000000000004, 44.332],
#         "floor_2016": [45.998999999999995, 42.032, 42.066],
#     }
# )

# print(df2)

# print(
#     df2.pivot_longer(
#         index="country", names_to=("event", "year"), names_sep="_"
#     )
# )

# df3 = pd.DataFrame({'country': ['United States', 'Russia', 'China'],
# 'vault_2012_f': [48.132, 46.36600000000001, 44.266000000000005],
# 'vault_2012_m': [46.632, 46.86600000000001, 48.316],
# 'vault_2016_f': [46.86600000000001, 45.733000000000004, 44.332],
# 'vault_2016_m': [45.865, 46.033, 45.0],
# 'floor_2012_f': [45.36600000000001, 41.599, 40.833],
# 'floor_2012_m': [45.266000000000005, 45.308, 45.133],
# 'floor_2016_f': [45.998999999999995, 42.032, 42.066],
# 'floor_2016_m': [43.757, 44.766000000000005, 43.799]})

# print(df3)

# print(
#    df3.pivot_longer(
#        index="country", names_to=("event", "year", "gender"), names_sep="_"
#    )
# )

# df4 = pd.DataFrame(
#    {
#       "country": ["United States", "Russia", "China"],
#     "floor2012": [45.36600000000001, 41.599, 40.833],
#     "vault2016": [46.86600000000001, 45.733000000000004, 44.332],
#    "floor2016": [45.998999999999995, 42.032, 42.066],
# }
# )

# print(df4)

# print(
#    df4.pivot_longer(
#        index="country",
#        names_to=("event", "year"),
#        names_sep="_",
#        names_pattern="([A-Za-z]+)(\\d+)",
#    )
# )

# df5 = pd.DataFrame({'country': ['United States', 'Russia', 'China'],
# 'score_vault': [46.86600000000001, 45.733000000000004, 44.332],
# 'score_floor': [45.998999999999995, 42.032, 42.066]})

# print(df5)

# print(df5.pivot_longer(index='country', names_to='event', names_sep="_"))
