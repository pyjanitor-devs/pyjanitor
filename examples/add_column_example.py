import pandas as pd
import janitor


data = {
    "a": [1, 2, 3] * 3,
    "Bell__Chart": [1, 2, 3] * 3,
    "decorated-elephant": [1, 2, 3] * 3,
    "animals": ["rabbit", "leopard", "lion"] * 3,
    "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
}


df = pd.DataFrame(data)


df.add_column('city_population', 100000)
df.add_column('city_population2', range(3), fill_remaining=True)


df.add_column('city_population1', df.city_population-2*df.city_population2)
df.add_column('city_population3', df.city_population-2*df.city_population3)
df.add_column('city_population4', df.city_population3-2*df.city_population1)



