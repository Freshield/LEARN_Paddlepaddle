#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b4_line_model.py
@Time: 2020-01-08 15:38
@Last_update: 2020-01-08 15:38
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import paddle.fluid as fluid
import numpy as np

np.random.seed(0)
outputs = np.random.randint(5, size=(10, 4))
res = []

for i in range(10):
    y = 4*outputs[i][0] + 6*outputs[i][1] + 7*outputs[i][2] + 2*outputs[i][3]
    res.append([y])

train_data = np.array(outputs).astype('float32')
y_true = np.array(res).astype('float32')

x = fluid.layers.data(name='x', shape=[4], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None)

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.05)
sgd_optimizer.minimize(avg_cost)

cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_main_program())

for i in range(500):
    outs = exe.run(
        feed={'x':train_data, 'y':y_true},
        fetch_list=[y_predict.name, avg_cost.name]
    )

    if i % 50 == 0:
        print('iter=%d, cost=.2f' % (i, outs[1][0]))