import pandas as pd
import janitor

data_dict = {
    'really_long_name_for_a_column': range(10),
    'another_really_long_name_for_a_column': [2*item for item in range(10)],
    'another_really_longer_name_for_a_column': list('lllongname'),
    'this_is_getting_out_of_hand': list('longername')
}

example_dataframe = pd.DataFrame(data_dict)

example_dataframe.limit_column_characters(7)

# And if you like a different separator character:

example_dataframe2 = pd.DataFrame(data_dict)

example_dataframe2.limit_column_characters(7, '.')


