import random
import torch

from d2l import torch as d2l


## 生成数据集和标签

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 均值为0，方差为1， n行 w长度列
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 添加噪声
    return X, y.reshape((-1, 1))  # 将y变成列向量

## 处理小批量的数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 打乱索引以便于后面取batch_size大小的随机批量
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

## 定义模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

## 定义损失函数
def squared_loss(y_hat, y):
    return(y_hat-y.reshape(y_hat.shape))**2/2


## 定义优化算法(梯度下降)
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr *param.grad / batch_size
            # 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
            param.grad.zero_() #清零， 下次计算不受影响


## 生成数据集
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print(features[:3, :])
print(labels[:3])

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()

##测试生成小批量
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 定义初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True) #需要计算记录梯度!!
b = torch.zeros(1, requires_grad=True)


# 训练 初始化超参数 和模型网络使用统一化用名 方便后期使用其他名字
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss


for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        #计算[w, b]的梯度 求和成为标量，使其能够访问梯度，
        l.sum().backward()
        ## 前面若没取平均可在此处取 (l.sum()/batch_size).backward()
        sgd([w, b], lr, batch_size)

    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch+1}, loss{float(train_l.mean()):f}')


## 自己生成的数据集 最开始知道 w,b 是多少 计算下实际参数误差

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')





