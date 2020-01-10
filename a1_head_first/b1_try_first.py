#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b1_try_first.py
@Time: 2020-01-10 15:03
@Last_update: 2020-01-10 15:03
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

data = fluid.layers.ones(shape=[5], dtype='int64')

place = fluid.CPUPlace()

exe = fluid.Executor(place)

ones_result = exe.run(fluid.default_main_program(),
                      fetch_list=[data],
                      return_numpy=True)

print(ones_result)