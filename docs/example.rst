Example Usage
=============

This example explicitly uses the `dirty_data.xlsx` file from the `janitor`_ repository.

.. _janitor: https://github.com/sfirke/janitor

Here's what the dirty dataframe looks like.

.. code-block:: python

  import pandas as pd
  import janitor as jn

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

Notice how there's an entire row of null values (row 7), as well as two columns of null values (`do not edit! --->` and `Certification.2`).

To clean up this data, we can use pyjanitor's functions (which are shamelessly copied from the R package).

.. code-block:: python

  (jn.DataFrame(df)
     .clean_names()
     .remove_empty()
     .rename_column('%_allocated', 'percent_allocated')
     .rename_column('full_time?', 'full_time')
     .coalesce(['certification', 'certification.1'], 'certification')
     .encode_categorical(['subject', 'employee_status', 'full_time'])
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
