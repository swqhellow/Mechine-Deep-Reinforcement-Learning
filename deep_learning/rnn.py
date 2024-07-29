# -*- coding :UTF-8 -*-
"""
@作者：shangwenqing
@文件名:rnn.py
@时间:2024:07:26:13:29
@IDE:PyCharm
@copyright:WenQing Shang
"""
import numpy as np

# 定义数据和参数
data = "During my undergraduate studies, I worked hard and achieved a GPA of 3.85 of 4 in the first five semesters, ranking first among 74 students in the department. Previously I served as the Technical Director of the Robotics Association and Class Leader of my major organizing and participating in activities such as the Global Campus Artificial Intelligence Competition. At the same time, I actively participate in innovation and entrepreneurship projects for college students, such as Yi language recognition based on deep learning and Intelligent voice recycling trash can. I am mainly responsible for building deep learning models and developing voice recognition hardware. My paper was included in the EI International Conference ITQM and I won three awards, including the second prize at the national level in the Global Campus Artificial Intelligence Competition. In terms of competition, I have participated in the RoboMaster National College Robot Competition for two years and won the first and second prizes at the national level, respectively"
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# 超参数
hidden_size = 200  # 隐藏层神经元数
seq_length = 20  # 序列长度
learning_rate = 1e-1

# 模型参数
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # 输入到隐藏层
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层到隐藏层
Why = np.random.randn(vocab_size, hidden_size) * 0.01  # 隐藏层到输出
bh = np.zeros((hidden_size, 1))  # 隐藏层偏置
by = np.zeros((vocab_size, 1))  # 输出层偏置


def lossFun(inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0

    # 前向传播
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(ps[t][targets[t], 0])

    # 反向传播
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


# 训练
n, p = 0, 0
hprev = np.zeros((hidden_size, 1))
while n < 10000:
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1))
        p = 0

    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)

    for param, dparam in zip([Wxh, Whh, Why, bh, by],
                             [dWxh, dWhh, dWhy, dbh, dby]):
        param -= learning_rate * dparam

    p += seq_length
    n += 1

    if n % 100 == 0:
        print(f'Iteration {n}, loss: {loss}')


# 测试生成文本
def sample(h, seed_ix, n):
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes


sample_ix = sample(hprev, char_to_ix['h'], 200)
txt = ''.join(ix_to_char[ix] for ix in sample_ix)
print(f'----\n {txt} \n----')
