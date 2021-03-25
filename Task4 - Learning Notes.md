### [逻辑回归模型](https://blog.csdn.net/han_xiaoyang/article/details/49123419)

它将数据拟合到一个logit函数(或者叫做logistic函数)中，从而能够完成对事件发生的概率进行预测。
如果线性回归的结果输出是一个连续值，而值的范围是无法限定的，那我们有没有办法把这个结果值映射为可以帮助我们判断的结果呢。而如果输出结果是 (0,1) 的一个概率值.
![image](https://user-images.githubusercontent.com/39177230/112442333-af3bd480-8d86-11eb-8fb1-a235480d4e82.png)

从函数图上可以看出，函数y=g(z)在z=0的时候取值为1/2，而随着z逐渐变小，函数值趋于0，z逐渐变大的同时函数值逐渐趋于1，而这正是一个概率的范围。

所以我们定义线性回归的预测函数为Y=WTX，那么逻辑回归的输出Y= g(WTX)，其中y=g(z)函数正是上述sigmoid函数(或者简单叫做S形函数)。

所谓的代价函数Cost Function，其实是一种衡量我们在这组参数下预估的结果和实际结果差距的函数，比如说线性回归的代价函数定义为:
![image](https://user-images.githubusercontent.com/39177230/112442759-26716880-8d87-11eb-83b5-1e036532cbf9.png)

查看数据在空间的分布 
```python
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
 
#load the dataset
data = loadtxt('/home/HanXiaoyang/data/data1.txt', delimiter=',')
 
X = data[:, 0:2]
y = data[:, 2]
 
pos = where(y == 1)
neg = where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Feature1/Exam 1 score')
ylabel('Feature2/Exam 2 score')
legend(['Fail', 'Pass'])
show()

```
![image](https://user-images.githubusercontent.com/39177230/112443647-2b82e780-8d88-11eb-9201-e85bf4c0d7da.png)


写好计算sigmoid函数、代价函数、和梯度下降的程序
```python
 
 def sigmoid(X):
    '''Compute sigmoid function '''
    den =1.0+ e **(-1.0* X)
    gz =1.0/ den
    return gz
def compute_cost(theta,X,y):
    '''computes cost given predicted and actual values'''
    m = X.shape[0]#number of training examples
    theta = reshape(theta,(len(theta),1))
    
    J =(1./m)*(-transpose(y).dot(log(sigmoid(X.dot(theta))))- transpose(1-y).dot(log(1-sigmoid(X.dot(theta)))))
    
    grad = transpose((1./m)*transpose(sigmoid(X.dot(theta))- y).dot(X))
    #optimize.fmin expects a single value, so cannot return grad
    return J[0][0]#,grad
def compute_grad(theta, X, y):
    '''compute gradient'''
    theta.shape =(1,3)
    grad = zeros(3)
    h = sigmoid(X.dot(theta.T))
    delta = h - y
    l = grad.size
    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i]=(1.0/ m)* sumdelta *-1
    theta.shape =(3,)
    return  grad
```

梯度下降算法得到的结果判定边界是如下的样子:

![image](https://user-images.githubusercontent.com/39177230/112444153-b7950f00-8d88-11eb-9089-d9f8f89886cd.png)

使用我们的判定边界对training data做一个预测，然后比对一下准确率：
计算出来的结果是89.2%

```python
def predict(theta, X):
    '''Predict label using learned logistic regression parameters'''
    m, n = X.shape
    p = zeros(shape=(m,1))
    h = sigmoid(X.dot(theta.T))
    for it in range(0, h.shape[0]):
        if h[it]>0.5:
            p[it,0]=1
        else:
            p[it,0]=0
    return p
#Compute accuracy on our training set
p = predict(array(theta), it)
print'Train Accuracy: %f'%((y[where(p == y)].size / float(y.size))*100.0)
```


### [决策树模型](https://blog.csdn.net/c406495762/article/details/76262487)

分类决策树模型是一种描述对实例进行分类的树形结构。决策树由结点(node)和有向边(directed edge)组成。结点有两种类型：内部结点(internal node)和叶结点(leaf node)。内部结点表示一个特征或属性，叶结点表示一个类。

#### 决策树的一些优点：

易于理解和解释，决策树可以可视化。
几乎不需要数据预处理。其他方法经常需要数据标准化，创建虚拟变量和删除缺失值。决策树还不支持缺失值。
使用树的花费（例如预测数据）是训练数据点(data points)数量的对数。
可以同时处理数值变量和分类变量。其他方法大都适用于分析一种变量的集合。
可以处理多值输出变量问题。
使用白盒模型。如果一个情况被观察到，使用逻辑判断容易表示这种规则。相反，如果是黑盒模型（例如人工神经网络），结果会非常难解释。
即使对真实模型来说，假设无效的情况下，也可以较好的适用。

#### 决策树的一些缺点：

决策树学习可能创建一个过于复杂的树，并不能很好的预测数据。也就是过拟合。修剪机制（现在不支持），设置一个叶子节点需要的最小样本数量，或者数的最大深度，可以避免过拟合。
决策树可能是不稳定的，因为即使非常小的变异，可能会产生一颗完全不同的树。这个问题通过decision trees with an ensemble来缓解。
学习一颗最优的决策树是一个NP-完全问题under several aspects of optimality and even for simple concepts。因此，传统决策树算法基于启发式算法，例如贪婪算法，即每个节点创建最优决策。这些算法不能产生一个全家最优的决策树。对样本和特征随机抽样可以降低整体效果偏差。
概念难以学习，因为决策树没有很好的解释他们，例如，XOR, parity or multiplexer problems.
如果某些分类占优势，决策树将会创建一棵有偏差的树。因此，建议在训练之前，先抽样使样本均衡。

#### 特征选择
    特征选择在于选取对训练数据具有分类能力的特征。这样可以提高决策树学习的效率，如果利用一个特征进行分类的结果与随机分类的结果没有很大差别，则称这个特征是没有分类能力的。经验上扔掉这样的特征对决策树学习的精度影响不大。通常特征选择的标准是信息增益(information gain)或信息增益比
    
#### 香农熵
熵定义为信息的期望值。在信息论与概率统计中，熵是表示随机变量不确定性的度量。如果待分类的事务可能划分在多个分类之中，则符号xi的信息定义为

![image](https://user-images.githubusercontent.com/39177230/112446910-aa2d5400-8d8b-11eb-8d63-08caa1bb14c1.png)

![image](https://user-images.githubusercontent.com/39177230/112446950-b87b7000-8d8b-11eb-9dba-c8b09f856216.png)


### [GBDT模型](https://zhuanlan.zhihu.com/p/45145899)

梯度提升树GBDT

#### CART回归树
GBDT是一个集成模型，可以看做是很多个基模型的线性相加，其中的基模型就是CART回归树。

CART树是一个决策树模型，与普通的ID3，C4.5相比，CART树的主要特征是，他是一颗二分树，每个节点特征取值为“是”和“不是”。举个例子，在ID3中如果天气是一个特征，那么基于此的节点特征取值为“晴天”、“阴天”、“雨天”，而CART树中就是“不是晴天”与“是晴天”。

这样的决策树递归的划分每个特征，并且在输入空间的每个划分单元中确定唯一的输出

![image](https://user-images.githubusercontent.com/39177230/112447936-d5fd0980-8d8c-11eb-908e-32455914be37.png)

![image](https://user-images.githubusercontent.com/39177230/112447978-e0b79e80-8d8c-11eb-950d-8441051e0823.png)

####  GBDT模型

GBDT模型是一个集成模型，是很多CART树的线性相加。

![image](https://user-images.githubusercontent.com/39177230/112448157-13619700-8d8d-11eb-9c5d-52f6ff8912b9.png)


### [XGBoost模型](https://blog.csdn.net/wuzhongqiang/article/details/104854890)

常见的机器学习算法：

* 监督学习算法：逻辑回归，线性回归，决策树，朴素贝叶斯，K近邻，支持向量机，集成算法Adaboost等
* 无监督算法：聚类，降维，关联规则, PageRank等

根据各个弱分类器之间有无依赖关系，分为Boosting和Bagging

* Boosting流派，各分类器之间有依赖关系，必须串行，比如Adaboost、GBDT(Gradient Boosting Decision Tree)、Xgboost
* Bagging流派，各分类器之间没有依赖关系，可各自并行，比如随机森林（Random Forest）

AdaBoost，是英文"Adaptive Boosting"（自适应增强），它的自适应在于：前一个基本分类器分错的样本会得到加强，加权后的全体样本再次被用来训练下一个基本分类器。同时，在每一轮中加入一个新的弱分类器，直到达到某个预定的足够小的错误率或达到预先指定的最大迭代次数。白话的讲，就是它在训练弱分类器之前，会给每个样本一个权重，训练完了一个分类器，就会调整样本的权重，前一个分类器分错的样本权重会加大，这样后面再训练分类器的时候，就会更加注重前面分错的样本， 然后一步一步的训练出很多个弱分类器，最后，根据弱分类器的表现给它们加上权重，组合成一个强大的分类器，就足可以应付整个数据集了。 这就是AdaBoost， 它强调自适应，不断修改样本权重， 不断加入弱分类器进行boosting。

GBDT(Gradient Boost Decision Tree)就是另一种boosting的方式， 上面说到AdaBoost训练弱分类器关注的是那些被分错的样本，AdaBoost每一次训练都是为了减少错误分类的样本。 而GBDT训练弱分类器关注的是残差，也就是上一个弱分类器的表现与完美答案之间的差距，GBDT每一次训练分类器，都是为了减少这个差距


### LightGBM模型

### Catboost模型

### 时间序列模型


