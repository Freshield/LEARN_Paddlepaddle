#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b1_linear_model.py
@Time: 2020-01-13 12:08
@Last_update: 2020-01-13 12:08
@Desc: None
@==============================================@
@      _____             _   _     _   _       @
@     |   __|___ ___ ___| |_|_|___| |_| |      @
@     |   __|  _| -_|_ -|   | | -_| | . |      @
@     |__|  |_| |___|___|_|_|_|___|_|___|      @
@                                    Freshield @
@==============================================@
"""
import paddle
import paddle.fluid as fluid
import numpy as np
import math
import sys
import six

# Data part
BATCH_SIZE = 20
feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'convert'
]
feature_num = len(feature_names)
data = np.fromfile('data/housing.txt', sep=' ')
data = data.reshape(data.shape[0] // feature_num, feature_num)
maxi, mini, avgs = data.max(axis=0), data.min(axis=0), data.sum(axis=0)/data.shape[0]

for i in range(feature_num-1):
    data[:, i] = ((data[:,i] - avgs[i]) / (maxi[i] - mini[i]))

ratio = 0.8
offset = int(data.shape[0] * ratio)
train_data = data[:offset]
test_data = data[offset:]


def reader_creator(train_data):
    def reader():
        for d in train_data:
            yield d[:-1], d[-1:]
    return reader


train_reader = paddle.batch(
    paddle.reader.shuffle(
        reader_creator(train_data), buf_size=500),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        reader_creator(test_data), buf_size=500),
    batch_size=BATCH_SIZE)


# Net part

x = fluid.data('x', [None, 13], 'float32')
y = fluid.data('y', [None, 1], 'float32')
y_predict = fluid.layers.fc(x, 1, act=None)

main_program = fluid.default_main_program()
startup_program = fluid.default_startup_program()

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_loss = fluid.layers.mean(cost)

test_program = main_program.clone(for_test=True)

sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

# # Train
# num_epochs = 100
#
#
# def train_test(executor, program, reader, feeder, fetch_list):
#     accumulated = 1 * [0]
#     count = 0
#     for data_test in reader():
#         outs = executor.run(program=program,
#                             feed=feeder.feed(data_test),
#                             fetch_list=fetch_list)
#         accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)] # 累加测试过程中的损失值
#         count += 1 # 累加测试集中的样本数量
#     return [x_d / count for x_d in accumulated] # 计算平均损失
#
#
params_dirname = 'data/inference_model'
# feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
# exe.run(startup_program)
# train_prompt = "train cost"
# test_prompt = "test cost"
# from paddle.utils.plot import Ploter
# plot_prompt = Ploter(train_prompt, test_prompt)
# step = 0
#
# exe_test = fluid.Executor(place)
#
# # Loop begin
# for pass_id in range(num_epochs):
#     for data_train in train_reader():
#         avg_loss_value, = exe.run(main_program, feed=feeder.feed(data_train), fetch_list=[avg_loss])
#
#     if step % 10 == 0:
#         # plot_prompt.append(train_prompt, step, avg_loss_value[0])
#         # plot_prompt.plot()
#         print('%s, Step %d, Cost %f' % (train_prompt, step, avg_loss_value[0]))
#
#     if step % 100 == 0:
#         test_metics = train_test(executor=exe_test,
#                                  program=test_program,
#                                  reader=test_reader,
#                                  fetch_list=[avg_loss.name],
#                                  feeder=feeder)
#         # plot_prompt.append(test_prompt, step, test_metics[0])
#         # plot_prompt.plot()
#         print('%s, Step %d, Cost %f' % (test_prompt, step, test_metics[0]))
#
#         if test_metics[0] < 10.0:
#             break
#
#     if math.isnan(float(avg_loss_value[0])):
#         sys.exit('got NaN loss, training failed.')
#
#     if params_dirname is not None:
#         fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)

infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()

def save_result(points1, points2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('data/image/prediction_gt.png')


with fluid.scope_guard(inference_scope):
    infer_program, feed_target_names, fetch_targets = fluid.io.load_inference_model(params_dirname, infer_exe)
    batch_size = 10

    infer_reader = paddle.batch(
        reader_creator(test_data), batch_size=batch_size)
    
    infer_data = next(infer_reader())
    infer_feat = np.array(
        [data[0] for data in infer_data]).astype('float32')
    infer_label = np.array(
        [data[1] for data in infer_data]).astype('float32')
    
    assert feed_target_names[0] == 'x'
    results = infer_exe.run(infer_program, feed={feed_target_names[0]: np.array(infer_feat)},
                            fetch_list=fetch_targets)

    print('infer results: (House Price)')
    for idx, val in enumerate(results[0]):
        print('%d: %.2f' % (idx, val))

    print('\nground truth:')
    for idx, val in enumerate(infer_label):
        print('%d: %.2f' % (idx, val))

    save_result(results[0], infer_label)