#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b2_array_add.py
@Time: 2020-01-10 15:22
@Last_update: 2020-01-10 15:22
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

data = fluid.layers.ones([5], 'int64')

add = fluid.layers.elementwise_add(data, data)
place = fluid.CPUPlace()
exe = fluid.Executor(place)

add_result = exe.run(fluid.default_main_program(),
                     fetch_list=[add],
                     return_numpy=True)

print(add_result)