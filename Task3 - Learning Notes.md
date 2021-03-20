#### [pandas.DataFrame.stack](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.stack.html)

DataFrame.stack(level=- 1, dropna=True)

Return a reshaped DataFrame or Series having a multi-level index with one or more new inner-most levels compared to the current DataFrame. The new inner-most levels are created by pivoting the columns of the current dataframe:

if the columns have a single level, the output is a Series;
if the columns have multiple levels, the new index level(s) is (are) taken from the prescribed level(s) and the output is a DataFrame.

Returns
DataFrame or Series
Stacked dataframe or series.

* example Before apply stack()

![image](https://user-images.githubusercontent.com/39177230/111866213-d9684d80-89a6-11eb-9056-5a8746bb6e74.png)

* after apply stack()

![image](https://user-images.githubusercontent.com/39177230/111866476-ade66280-89a8-11eb-8bd0-b2a305fa17c6.png)

#### [pandas.DataFrame.reset_index](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html)

Reset the index, or a level of it.

Reset the index of the DataFrame, and use the default one instead. If the DataFrame has a MultiIndex, this method can remove one or more levels.
Returns
DataFrame or None
DataFrame with the new index or None if inplace=True.

* After reset_index()
![image](https://user-images.githubusercontent.com/39177230/111866554-4a106980-89a9-11eb-92d1-44b0fd5f55b1.png)

* After set_index()
![image](https://user-images.githubusercontent.com/39177230/111866590-8217ac80-89a9-11eb-9851-6f1a4f396539.png)





