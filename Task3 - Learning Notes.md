#### [pandas.DataFrame.stack](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.stack.html)

DataFrame.stack(level=- 1, dropna=True)

Return a reshaped DataFrame or Series having a multi-level index with one or more new inner-most levels compared to the current DataFrame. The new inner-most levels are created by pivoting the columns of the current dataframe:

if the columns have a single level, the output is a Series;

if the columns have multiple levels, the new index level(s) is (are) taken from the prescribed level(s) and the output is a DataFrame.

Parameters
levelint, str, list, default -1
Level(s) to stack from the column axis onto the index axis, defined as one index or label, or a list of indices or labels.

dropnabool, default True
Whether to drop rows in the resulting Frame/Series with missing values. Stacking a column level onto the index axis can create combinations of index and column values that are missing from the original dataframe. See Examples section.

Returns
DataFrame or Series
Stacked dataframe or series.

* example Before apply stack()

![image](https://user-images.githubusercontent.com/39177230/111866213-d9684d80-89a6-11eb-9056-5a8746bb6e74.png)

* after apply stack()


