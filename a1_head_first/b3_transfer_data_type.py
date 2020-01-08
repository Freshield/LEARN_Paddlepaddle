#coding=utf-8
"""
@Author: Freshield
@Contact: yangyufresh@163.com
@File: b3_transfer_data_type.py
@Time: 2020-01-09 15:26
@Last_update: 2020-01-09 15:26
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
cast = fluid.layers.cast(data, 'float64')

place = fluid.CPUPlace()
exe = fluid.Executor(place)
cast_result = exe.run(fluid.default_main_program(),
                      fetch_list=[cast],
                      return_numpy=True)

print(cast_result)