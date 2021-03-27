[Task 5 模型融合代码运行](https://github.com/frankyangdev/aliyun-tianchi-datamining-heartbeatclassification/blob/main/T5%20-%20HeartbeatClassification-Ensambling.ipynb)

### Voting

假设对于一个二分类问题，有3个基础模型，那么就采取投票制的方法，投票多者确定为最终的分类。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)])
eclf = eclf1.fit(x_train,y_train)
print(eclf1.predict(x_test))
```

### Averaging

对于回归问题，一个简单直接的思路是取平均。稍稍改进的方法是进行加权平均。权值可以用排序的方法确定，举个例子，比如A、B、C三种基本模型，模型效果进行排名，假设排名分别是1，2，3，那么给这三个模型赋予的权值分别是3/6、2/6、1/6 
这两种方法看似简单，其实后面的高级算法也可以说是基于此而产生的，Bagging或者Boosting都是一种把许多弱分类器这样融合成强分类器的思想

### Bagging

Bagging就是采用有放回的方式进行抽样，用抽样的样本建立子模型,对子模型进行训练，这个过程重复多次，最后进行融合。大概分为这样两步：
#### 重复K次
* 有放回地重复抽样建模
* 训练子模型

#### 模型融合 
* 分类问题：voting
* 回归问题：average

Bagging算法不用我们自己实现，随机森林就是基于Bagging算法的一个典型例子，采用的基分类器是决策树。R和python都集成好了，直接调用。

随机森林实际上就是Bagging算法的进化版，不同于Bagging算法的是，Bagging产生不同数据集的方式只是对行利用有放回的随机抽样，而随机森林产生不同数据集的方式不仅对行随机抽样也对列进行随机抽样。

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

### Boosting

Boosting的思想是一种迭代的方法，每一次训练的时候都更加关心分类错误的样例，给这些分类错误的样例增加更大的权重，下一次迭代的目标就是能够更容易辨别出上一轮分类错误的样例。最终将这些弱分类器进行加权相加

![image](https://user-images.githubusercontent.com/39177230/112707902-b29ba100-8ee9-11eb-823d-9f5049b34cb1.png)

![image](https://user-images.githubusercontent.com/39177230/112707911-be876300-8ee9-11eb-9ddd-51586baf11a6.png)


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

### Blending

* 将数据划分为训练集和测试集(test_set)，其中训练集需要再次划分为训练集(train_set)和验证集(val_set)；
* 创建第一层的多个模型，这些模型可以使同质的也可以是异质的；
* 使用train_set训练步骤2中的多个模型，然后用训练好的模型预测val_set和test_set得到val_predict, test_predict1；
* 创建第二层的模型,使用val_predict作为训练集训练第二层的模型；
* 使用第二层训练好的模型对第二层测试集test_predict1进行预测，该结果为整个测试集的结果

![image](https://user-images.githubusercontent.com/39177230/112708018-5c7b2d80-8eea-11eb-978e-e191fa4cf297.png)


### Blending与Stacking对比
Blending的优点在于：

1.比stacking简单（因为不用进行k次的交叉验证来获得stacker feature）

2.避开了一个信息泄露问题：generlizers和stacker使用了不一样的数据集

3.在团队建模过程中，不需要给队友分享自己的随机种子

而缺点在于：

1.使用了很少的数据（是划分hold-out作为测试集，并非cv）

2.blender可能会过拟合（其实大概率是第一点导致的）

3.stacking使用多次的CV会比较稳健


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

### Ref:
[模型融合在kaggle比赛中的几种常见应用](https://blog.csdn.net/sinat_26811377/article/details/98495425?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161681205016780266220058%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=161681205016780266220058&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-5-98495425.first_rank_v2_pc_rank_v29&utm_term=%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88%E7%9A%84%E4%B8%89%E7%A7%8D%E6%96%B9%E5%BC%8F&spm=1018.2226.3001.4187)

[模型融合方法概述](https://blog.csdn.net/muyimo/article/details/80066449?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161681205016780269821374%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161681205016780269821374&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-80066449.first_rank_v2_pc_rank_v29&utm_term=%E6%A8%A1%E5%9E%8B%E8%9E%8D%E5%90%88%E7%9A%84%E4%B8%89%E7%A7%8D%E6%96%B9%E5%BC%8F&spm=1018.2226.3001.4187)

