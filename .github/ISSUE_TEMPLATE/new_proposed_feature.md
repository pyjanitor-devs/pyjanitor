---
name: Propose New Feature
about: If you'd like to propose a new feature, please use this template.
---

# Brief Description

<!-- Please provide a brief description of what you'd like to propose. -->

I would like to propose...

# Example API

<!-- One of the selling points of pyjanitor is the API. Hence, we guard the API very carefully, and want to
make sure that it is accessible and understandable to many people. Please provide a few examples of what the API
of the new function you're proposing will look like. We have provided an example that you should modify. -->

Please modify the example API below to illustrate your proposed API, and then delete this sentence.

```python
# transform only one column, while creating a new column name for it.
df.transform_columns(column_name=['col1'], function=np.abs, new_column_name=['col1_abs'])

# transform multiple columns by the same function, without creating a new column name.
df.transform_columns(column_name=['col1', 'col2'], function=np.abs)

# more examples below
# ...
```
