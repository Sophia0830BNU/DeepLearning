import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
from torch import nn


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


net = nn.Sequential(nn.Linear(2,1)) ##单层 两个输入一个输出
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels =d2l.synthetic_data(true_w, true_b, 1000)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

##使用iter构造Python迭代器，并使用next从迭代器中获取第一项
next(iter(data_iter))

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step() #更新参数 原实现过程在sgd内部更新
    l = loss(net(features), labels)
    print(f'epoch {epoch+1}, loss{l:f}')