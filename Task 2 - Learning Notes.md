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











