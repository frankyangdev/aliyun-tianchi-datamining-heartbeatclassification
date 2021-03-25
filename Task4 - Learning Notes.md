### 逻辑回归模型

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


### 决策树模型

### GBDT模型

### XGBoost模型

### LightGBM模型

### Catboost模型

### 时间序列模型


