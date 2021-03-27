[Task 5 模型融合代码运行](https://github.com/frankyangdev/aliyun-tianchi-datamining-heartbeatclassification/blob/main/T5%20-%20HeartbeatClassification-Ensambling.ipynb)

[sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

```python
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.datasets import make_regression
>>> X, y = make_regression(n_features=4, n_informative=2,
...                        random_state=0, shuffle=False)
>>> regr = RandomForestRegressor(max_depth=2, random_state=0)
>>> regr.fit(X, y)
RandomForestRegressor(...)
>>> print(regr.predict([[0, 0, 0, 0]]))
[-8.32987858]

```
![image](https://user-images.githubusercontent.com/39177230/112706453-300de400-8edf-11eb-9306-90200134c5fa.png)


[sklearn.neural_network.MLPRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

```python
>>> from sklearn.neural_network import MLPRegressor
>>> from sklearn.datasets import make_regression
>>> from sklearn.model_selection import train_test_split
>>> X, y = make_regression(n_samples=200, random_state=1)
>>> X_train, X_test, y_train, y_test = train_test_split(X, y,
...                                                     random_state=1)
>>> regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
>>> regr.predict(X_test[:2])
array([-0.9..., -7.1...])
>>> regr.score(X_test, y_test)
0.4...
```



![image](https://user-images.githubusercontent.com/39177230/112706461-3c923c80-8edf-11eb-8524-6860c8663d5e.png)

[sklearn.metrics.mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)

```python
>>> from sklearn.metrics import mean_absolute_error
>>> y_true = [3, -0.5, 2, 7]
>>> y_pred = [2.5, 0.0, 2, 8]
>>> mean_absolute_error(y_true, y_pred)
0.5
>>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
>>> y_pred = [[0, 2], [-1, 2], [8, -5]]
>>> mean_absolute_error(y_true, y_pred)
0.75
>>> mean_absolute_error(y_true, y_pred, multioutput='raw_values')
array([0.5, 1. ])
>>> mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
0.85...
```

![image](https://user-images.githubusercontent.com/39177230/112707558-4750cf80-8ee7-11eb-9dcc-519fe358e3c7.png)

### Stacking

#### 假设有3个模型M1、M2、M3
1. 基模型M1，对训练集train训练，然后用于预测train和test的标签列，分别是P1，T1

![image](https://user-images.githubusercontent.com/39177230/112707638-c5ad7180-8ee7-11eb-9698-12aebca0368f.png)

#### 分别把P1,P2,P3以及T1,T2,T3合并，得到一个新的训练集和测试集train2,test2

![image](https://user-images.githubusercontent.com/39177230/112707648-dd84f580-8ee7-11eb-97d6-685f618a2102.png) ![image](https://user-images.githubusercontent.com/39177230/112707652-e675c700-8ee7-11eb-8691-a0850bae3cd2.png)

#### 再用第二层的模型M4训练train2,预测test2,得到最终的标签列
![image](https://user-images.githubusercontent.com/39177230/112707670-fb525a80-8ee7-11eb-863c-4d4bf86895cc.png

#### 用整个训练集训练的模型反过来去预测训练集的标签，毫无疑问过拟合是非常非常严重的，因此现在的问题变成了如何在解决过拟合的前提下得到P1、P2、P3，这就变成了熟悉的节奏——K折交叉验证。我们以2折交叉验证得到P1为例,假设训练集为4行3列

![image](https://user-images.githubusercontent.com/39177230/112707707-2ccb2600-8ee8-11eb-85f5-65b3e3f082b2.png)

对于测试集T1的得到，有两种方法。注意到刚刚是2折交叉验证，M1相当于训练了2次，所以一种方法是每一次训练M1，可以直接对整个test进行预测，这样2折交叉验证后测试集相当于预测了2次，然后对这两列求平均得到T1。
或者直接对测试集只用M1预测一次直接得到T1。
P1、T1得到之后，P2、T2、P3、T3也就是同样的方法。理解了2折交叉验证，对于K折的情况也就理解也就非常顺利了。所以最终的代码是两层循环，第一层循环控制基模型的数目，每一个基模型要这样去得到P1，T1，第二层循环控制的是交叉验证的次数K，对每一个基模型，会训练K次最后拼接得到P1，取平均得到T1。

![image](https://user-images.githubusercontent.com/39177230/112707743-74ea4880-8ee8-11eb-8e8b-ef7de55e3896.png)

该图是一个基模型得到P1和T1的过程，采用的是5折交叉验证，所以循环了5次，拼接得到P1，测试集预测了5次，取平均得到T1。而这仅仅只是第二层输入的一列/一个特征，并不是整个训练集。

```pyhton
def get_oof(clf, x_train, y_train, x_test):
 oof_train = np.zeros((ntrain,))  
 oof_test = np.zeros((ntest,))
 oof_test_skf = np.empty((NFOLDS, ntest))  #NFOLDS行，ntest列的二维array
 for i, (train_index, test_index) in enumerate(kf): #循环NFOLDS次
     x_tr = x_train[train_index]
     y_tr = y_train[train_index]
     x_te = x_train[test_index]
     clf.fit(x_tr, y_tr)
     oof_train[test_index] = clf.predict(x_te)
     oof_test_skf[i, :] = clf.predict(x_test)  #固定行填充，循环一次，填充一行
 oof_test[:] = oof_test_skf.mean(axis=0)  #axis=0,按列求平均，最后保留一行
 return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)  #转置，从一行变为一列
```








[模型融合方法概述](https://blog.csdn.net/muyimo/article/details/80066449?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161681205016780269821374%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161681205016780269821374&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-80066449.first_rank_v2_pc_rank_v29&utm_term=%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88%E7%9A%84%E4%B8%89%E7%A7%8D%E6%96%B9%E5%BC%8F&spm=1018.2226.3001.4187)

