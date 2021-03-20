##### 查看当前kernel下已安装的包  list packages
!pip list --format=columns

#### Notebook [T2 - EDA.ipynb](https://github.com/frankyangdev/aliyun-tianchi-datamining-heartbeatclassification/blob/main/T2%20-%20EDA.ipynb)

#### import package:

#coding:utf-8
#导入warnings包，利用过滤器来实现忽略警告语句。
import warnings
warnings.filterwarnings('ignore')

[import missingno as msno](https://libraries.io/pypi/missingno)

[import matplotlib.pyplot as plt](https://matplotlib.org/2.0.2/api/pyplot_api.html)

[import seaborn as sns](https://seaborn.pydata.org/introduction.html)



#### pd.options.display.max_colwidth = 250 extend column size to view more value in same column, change the value according to your column length
import pandas as pd

![image](https://user-images.githubusercontent.com/39177230/111588106-13e2b680-87fe-11eb-88d0-fbc4d1da9567.png)

![image](https://user-images.githubusercontent.com/39177230/111588192-2b21a400-87fe-11eb-9538-d25ceba70251.png)

#### df.shape will show the reocrds and size of dataframe
![image](https://user-images.githubusercontent.com/39177230/111588729-e77b6a00-87fe-11eb-8f8b-ca6c447e1766.png)

#### df.head().append(Train_data.tail()) will combine 1st 5 records and last five records together

![image](https://user-images.githubusercontent.com/39177230/111588923-27425180-87ff-11eb-8d93-7e22d4901614.png)

#### df.describe() Generate descriptive statistics. Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
![image](https://user-images.githubusercontent.com/39177230/111589012-45a84d00-87ff-11eb-9f67-f3ad909f2e33.png)

#### df.info() show column type and the index dtype and column dtypes, non-null values and memory usage.
![image](https://user-images.githubusercontent.com/39177230/111589429-ccf5c080-87ff-11eb-93e2-4e8c58e21675.png)

#### df.isnull().sum() check sum of null value
![image](https://user-images.githubusercontent.com/39177230/111589528-eb5bbc00-87ff-11eb-8168-c29083e07eb6.png)

#### df.count() check each count of column
![image](https://user-images.githubusercontent.com/39177230/111589842-59a07e80-8800-11eb-90b0-577790eb49b9.png)


#### df['label'] display value of column label
![image](https://user-images.githubusercontent.com/39177230/111590166-bdc34280-8800-11eb-9ec7-1a21ce53001d.png)

#### df['label'].value_counts()  count group by different value
![image](https://user-images.githubusercontent.com/39177230/111590344-f3682b80-8800-11eb-93a5-51f4462f81f6.png)

![image](https://user-images.githubusercontent.com/39177230/111590530-2ca09b80-8801-11eb-91c9-0d6c754c7642.png)

![image](https://user-images.githubusercontent.com/39177230/111590621-4cd05a80-8801-11eb-8ebe-f3c9d46307a8.png)

![image](https://user-images.githubusercontent.com/39177230/111590648-55c12c00-8801-11eb-914a-13fa3e1cf34c.png)

#### [Skew()](https://pythontic.com/pandas/dataframe-computations/skew) function in Pandas. Skewness is a measure of asymmetry(不对称) of a distribution. Another measure that describes the shape of a distribution is kurtosis.

#### [Kurt()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.kurt.html) Return unbiased kurtosis over requested axis.


![image](https://user-images.githubusercontent.com/39177230/111590700-6bceec80-8801-11eb-8b0c-539b0b2aaa34.png)

![image](https://user-images.githubusercontent.com/39177230/111590792-899c5180-8801-11eb-89a2-ea9dbdc1de9b.png)

![image](https://user-images.githubusercontent.com/39177230/111590832-9456e680-8801-11eb-8fc7-6fa6443e4b94.png)

![image](https://user-images.githubusercontent.com/39177230/111590938-b6e8ff80-8801-11eb-8406-8b741c53813b.png)

#### [Profiling report](https://github.com/frankyangdev/aliyun-tianchi-datamining-heartbeatclassification/blob/main/T2%20-%20Train%20data%20profiling%20example.html)

![image](https://user-images.githubusercontent.com/39177230/111591050-e1d35380-8801-11eb-89c4-56fe49309c0b.png)


