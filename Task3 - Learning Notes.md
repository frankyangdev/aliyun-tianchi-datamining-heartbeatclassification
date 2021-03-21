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


* Rename index column name to null and other column names to time and signals, chagne column signal type to float
![image](https://user-images.githubusercontent.com/39177230/111866779-dc653d00-89aa-11eb-914b-8388000feadf.png)


#### [pandas.DataFrame.drop](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)

Drop specified labels from rows or columns.

Remove rows or columns by specifying label names and corresponding axis, or by specifying directly index or column names. When using a multi-index, labels on different levels can be removed by specifying the level.

Parameters
labelssingle label or list-like
Index or column labels to drop.

axis{0 or ‘index’, 1 or ‘columns’}, default 0
Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).

indexsingle label or list-like
Alternative to specifying axis (labels, axis=0 is equivalent to index=labels).

columnssingle label or list-like
Alternative to specifying axis (labels, axis=1 is equivalent to columns=labels).

#### [tsfresh quick start](https://tsfresh.readthedocs.io/en/latest/text/quick_start.html)

### Code sample [Multiclass Selection Example.ipynb](https://github.com/frankyangdev/tsfresh/blob/main/notebooks/examples/04%20Multiclass%20Selection%20Example.ipynb)

![image](https://user-images.githubusercontent.com/39177230/111891082-c5662f80-8a2a-11eb-9b1c-011b03394315.png)

#### extract_features()

#### impute()

#### select_features()

#### [Benjamini Yekutieli procedure](https://en.wikipedia.org/wiki/False_discovery_rate#Benjamini%E2%80%93Yekutieli_procedure)

![image](https://user-images.githubusercontent.com/39177230/111891167-baf86580-8a2b-11eb-86d6-d871ba454a95.png)










 





