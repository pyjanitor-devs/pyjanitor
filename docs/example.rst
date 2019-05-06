Example Usage
=============

This example explicitly uses the `dirty_data.xlsx` file from the `janitor`_ repository.

.. _janitor: https://github.com/sfirke/janitor

Introduction to Dirty Data
--------------------------

Here's what the dirty dataframe looks like.

.. code-block:: python

  import pandas as pd
  import janitor

  df = pd.read_excel('examples/dirty_data.xlsx')
  print(df)

.. code-block:: none

  First Name Last Name Employee Status     Subject  Hire Date  \
  0          Jason    Bourne         Teacher          PE    39690.0
  1          Jason    Bourne         Teacher    Drafting    39690.0
  2         Alicia      Keys         Teacher       Music    37118.0
  3            Ada  Lovelace         Teacher         NaN    27515.0
  4          Desus      Nice  Administration        Dean    41431.0
  5   Chien-Shiung        Wu         Teacher     Physics    11037.0
  6   Chien-Shiung        Wu         Teacher   Chemistry    11037.0
  7            NaN       NaN             NaN         NaN        NaN
  8          James     Joyce         Teacher     English    32994.0
  9           Hedy    Lamarr         Teacher     Science    27919.0
  10        Carlos    Boozer           Coach  Basketball    42221.0
  11         Young    Boozer           Coach         NaN    34700.0
  12       Micheal    Larsen         Teacher     English    40071.0

      % Allocated Full time?  do not edit! ---> Certification Certification.1  \
  0          0.75        Yes                NaN   Physical ed         Theater
  1          0.25        Yes                NaN   Physical ed         Theater
  2          1.00        Yes                NaN  Instr. music     Vocal music
  3          1.00        Yes                NaN       PENDING       Computers
  4          1.00        Yes                NaN       PENDING             NaN
  5          0.50        Yes                NaN  Science 6-12         Physics
  6          0.50        Yes                NaN  Science 6-12         Physics
  7           NaN        NaN                NaN           NaN             NaN
  8          0.50         No                NaN           NaN    English 6-12
  9          0.50         No                NaN       PENDING             NaN
  10          NaN         No                NaN   Physical ed             NaN
  11          NaN         No                NaN           NaN  Political sci.
  12         0.80         No                NaN   Vocal music         English

      Certification.2
  0               NaN
  1               NaN
  2               NaN
  3               NaN
  4               NaN
  5               NaN
  6               NaN
  7               NaN
  8               NaN
  9               NaN
  10              NaN
  11              NaN
  12              NaN

Cleaning Column Names
---------------------

There's a bunch of problems with this data. Firstly, the column names are not lowercase, and they have spaces. This will make it cumbersome to use in a programmatic function. To solve this, we can use the :py:meth:`clean_names` method.

.. code-block:: python

  df_clean = df.clean_names()
  print(df_clean.head(2))

Notice now how the column names have been made better.

.. code-block:: none

    first_name last_name employee_status   subject  hire_date  %_allocated  \
    0      Jason    Bourne         Teacher        PE    39690.0         0.75
    1      Jason    Bourne         Teacher  Drafting    39690.0         0.25

      full_time?  do_not_edit!_---> certification certification.1  certification.2
    0        Yes                NaN   Physical ed         Theater              NaN
    1        Yes                NaN   Physical ed         Theater              NaN

If you squint at the unclean dataset, you'll notice one row and one column of data that are missing. We can also fix this! Building on top of the code block from above, let's now remove those empty columns using the :py:meth:`remove_empty()` method:

.. code-block:: python

    df_clean = df.clean_names().remove_empty()
    print(df_clean.head(5))

.. code-block:: none

    first_name last_name employee_status   subject  hire_date  %_allocated  \
    0      Jason    Bourne         Teacher        PE    39690.0         0.75
    1      Jason    Bourne         Teacher  Drafting    39690.0         0.25
    2     Alicia      Keys         Teacher     Music    37118.0         1.00
    3        Ada  Lovelace         Teacher       NaN    27515.0         1.00
    4      Desus      Nice  Administration      Dean    41431.0         1.00

    full_time? certification certification.1
    0        Yes   Physical ed         Theater
    1        Yes   Physical ed         Theater
    2        Yes  Instr. music     Vocal music
    3        Yes       PENDING       Computers
    4        Yes       PENDING             NaN

Now this is starting to shape up well!

Renaming Individual Columns
---------------------------

Next, let's rename some of the columns. `%_allocated` and `full_time?` contain non-alphanumeric characters, so they make it a bit harder to use. We can rename them using the :py:meth:`rename_column()` method:

.. code-block:: python

    df_clean = (df.clean_names()
                .remove_empty()
                .rename_column("%_allocated", "percent_allocated")
                .rename_column("full_time?", "full_time"))

    print(df_clean.head(5))

.. code-block:: none

    first_name last_name employee_status   subject  hire_date  \
    0      Jason    Bourne         Teacher        PE    39690.0
    1      Jason    Bourne         Teacher  Drafting    39690.0
    2     Alicia      Keys         Teacher     Music    37118.0
    3        Ada  Lovelace         Teacher       NaN    27515.0
    4      Desus      Nice  Administration      Dean    41431.0

     percent_allocated full_time certification certification.1
    0               0.75       Yes   Physical ed         Theater
    1               0.25       Yes   Physical ed         Theater
    2               1.00       Yes  Instr. music     Vocal music
    3               1.00       Yes       PENDING       Computers
    4               1.00       Yes       PENDING             NaN


Note how now we have really nice column names! You might be wondering why I'm not modifying the two certifiation columns -- that is the next thing we'll tackle.

Coalescing Columns
------------------

If we look more closely at the two `certification` columns, we'll see that they look like this:

.. code-block:: python

    print(df_clean[['certification', 'certification.1']])

.. code-block:: none

    certification certification.1
    0    Physical ed         Theater
    1    Physical ed         Theater
    2   Instr. music     Vocal music
    3        PENDING       Computers
    4        PENDING             NaN
    5   Science 6-12         Physics
    6   Science 6-12         Physics
    8            NaN    English 6-12
    9        PENDING             NaN
    10   Physical ed             NaN
    11           NaN  Political sci.
    12   Vocal music         English

Rows 8 and 11 have NaN in the left certification column, but have a value in the right certification column. Let's assume for a moment that the left certification column is intended to record the first certification that a teacher had obtained. In this case, the values in the right certification column on rows 8 and 11 should be moved to the first column. Let's do that with Janitor, using the :py:meth:`coalesce()` method, which does the following:

.. code-block:: python

    df_clean = (df.clean_names()
                .remove_empty()
                .rename_column("%_allocated", "percent_allocated")
                .rename_column("full_time?", "full_time")
                .coalesce(columns=['certification', 'certification.1'], new_column_name='certification'))

    print(df_clean)

.. code-block:: none

    first_name last_name employee_status     subject  hire_date  \
    0          Jason    Bourne         Teacher          PE    39690.0
    1          Jason    Bourne         Teacher    Drafting    39690.0
    2         Alicia      Keys         Teacher       Music    37118.0
    3            Ada  Lovelace         Teacher         NaN    27515.0
    4          Desus      Nice  Administration        Dean    41431.0
    5   Chien-Shiung        Wu         Teacher     Physics    11037.0
    6   Chien-Shiung        Wu         Teacher   Chemistry    11037.0
    8          James     Joyce         Teacher     English    32994.0
    9           Hedy    Lamarr         Teacher     Science    27919.0
    10        Carlos    Boozer           Coach  Basketball    42221.0
    11         Young    Boozer           Coach         NaN    34700.0
    12       Micheal    Larsen         Teacher     English    40071.0

        percent_allocated full_time   certification
    0                0.75       Yes     Physical ed
    1                0.25       Yes     Physical ed
    2                1.00       Yes    Instr. music
    3                1.00       Yes         PENDING
    4                1.00       Yes         PENDING
    5                0.50       Yes    Science 6-12
    6                0.50       Yes    Science 6-12
    8                0.50        No    English 6-12
    9                0.50        No         PENDING
    10                NaN        No     Physical ed
    11                NaN        No  Political sci.
    12               0.80        No     Vocal music

Awesome stuff! Now we don't have two columns of scattered data, we have one column of densely populated data.

Dealing with Excel Dates
------------------------

Finally, notice how the `hire_date` column isn't date formatted. It's got this weird Excel serialization.
To clean up this data, we can use the :py:meth:`convert_excel_date` method.

.. code-block:: python

  df_clean = (df.clean_names()
              .remove_empty()
              .rename_column('%_allocated', 'percent_allocated')
              .rename_column('full_time?', 'full_time')
              .coalesce(['certification', 'certification.1'], 'certification')
              .convert_excel_date('hire_date'))

This gives the output:

.. code-block:: none

  first_name last_name employee_status     subject  hire_date  \
  0          Jason    Bourne         Teacher          PE 2008-08-30
  1          Jason    Bourne         Teacher    Drafting 2008-08-30
  2         Alicia      Keys         Teacher       Music 2001-08-15
  3            Ada  Lovelace         Teacher         NaN 1975-05-01
  4          Desus      Nice  Administration        Dean 2013-06-06
  5   Chien-Shiung        Wu         Teacher     Physics 1930-03-20
  6   Chien-Shiung        Wu         Teacher   Chemistry 1930-03-20
  8          James     Joyce         Teacher     English 1990-05-01
  9           Hedy    Lamarr         Teacher     Science 1976-06-08
  10        Carlos    Boozer           Coach  Basketball 2015-08-05
  11         Young    Boozer           Coach         NaN 1995-01-01
  12       Micheal    Larsen         Teacher     English 2009-09-15

    percent_allocated full_time   certification
  0                0.75       Yes     Physical ed
  1                0.25       Yes     Physical ed
  2                1.00       Yes    Instr. music
  3                1.00       Yes         PENDING
  4                1.00       Yes         PENDING
  5                0.50       Yes    Science 6-12
  6                0.50       Yes    Science 6-12
  8                0.50        No    English 6-12
  9                0.50        No         PENDING
  10                NaN        No     Physical ed
  11                NaN        No  Political sci.
  12               0.80        No     Vocal music

We have a cleaned dataframe!
