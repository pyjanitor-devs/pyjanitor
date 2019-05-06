df.add_column()
Description
This method adds a column to the dataframe. It is intended to be the method-chaining alternative to: `df[colname] = value`.

df.convert_currency()
Description
This method converts a column from one currency to another, with an option to convert based on historical exchange values.

df.filter_date()
Description
This method modifies a column of floats to a nearest fraction, given a certain denominator.

df.limit_column_characters()
Description
This method truncates column names to a given character length. In the case of duplicated column names, numbers are appended to the columns with a character separator (default is "_").

df.make_currency_column_numeric()
Description
This method allows one to take a column containing currency values, inadvertently imported as a string, and cast it as a float. This is usually the case when reading CSV files that were modified in Excel. Empty strings (i.e. '') are retained as NaN values.

df.round_to_fraction()
Description
This method modifies a column of floats to a nearest fraction, given a certain denominator.

df.row_to_names()
Description
This method elevates a row to be the column names of a DataFrame. It contains parameters to remove the elevated row from the DataFrame along with removing the rows above the selected row.

df.then()
Description
This method allows for arbitrary functions to be used in the pyJanitor method-chaining fluent API.




=====================  ===========================================================================================================
Method                  Description
---------------------  ------------------------------------------------------------------------------------------------------------
df.add_column()         This method adds a column to the dataframe. It is intended to be the method-chaining alternative to: 
                        `df[colname] = value`.
---------------------  -----------------------------------------------------------------------------------------------------------
df.convert_currency()   This method converts a column from one currency to another, with an option to convert based on historical 
                        exchange values.

=====================  ===========================================================================================================
