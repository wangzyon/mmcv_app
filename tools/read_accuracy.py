#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import matplotlib.pyplot as plt
import argparse
'''
解析参数
'''
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default='val')
parser.add_argument("--json_path", type=str, default='20220721_020123.log.json')
parser.add_argument("--out_dir", type=str,default='/volume/singal/SignalSeparation/workdir/result/')
args = parser.parse_args()


select = ['n_cluster_accuracy','need_separation_accuracy', 'avg_rand_score'] # 选择绘图的指标
mode = args.mode  # 选择log文件中的模式
json_path = "/volume/singal/SignalSeparation/workdir/SeparationII_Sta_nomask/"+args.json_path
ext = os.path.splitext(args.json_path)

x = []  # 存放epoch
y_cluster = []  # 存放指标
y_cluster_min = 1000000  # 存放指标最大值   ap不会超过1000000  绘制loss可自由更改
y_cluster_max = -1  # 存放指标最小值   ap不会小于-1   绘制loss可自由更改
x_cluster_min = 0  # 出现最小值的epoch
x_cluster_max = 0  # 出现最大值的epoch

y_separation = []
y_separation_min = 1000000  # 存放指标最大值   ap不会超过1000000  绘制loss可自由更改
y_separation_max = -1  # 存放指标最小值   ap不会小于-1   绘制loss可自由更改
x_separation_min = 0  # 出现最小值的epoch
x_separation_max = 0  # 出现最大值的epoch

y_rand_score = []
y_rand_score_min = 1000000  # 存放指标最大值   ap不会超过1000000  绘制loss可自由更改
y_rand_score_max = -1  # 存放指标最小值   ap不会小于-1   绘制loss可自由更改
x_rand_score_min = 0  # 出现最小值的epoch
x_rand_score_max = 0  # 出现最大值的epoch

isFirst = True
with open(json_path, 'r') as f:
    for jsonstr in f.readlines():
        if isFirst:  # mmdetection生成的log  json文件第一行是配置信息  跳过
            isFirst = False
            continue
        row_data = json.loads(jsonstr)
        if row_data['mode'] == mode:  # 选择train或者val模式中的指标数据
            item_cluster_select = float(row_data['n_cluster_accuracy'])
            x_select = int(row_data['epoch'])
            x.append(x_select)
            y_cluster.append(item_cluster_select)
            if item_cluster_select >= y_cluster_max:  # 选择最大值  为什么不用numpy.argmax呢？  因为epoch可# 能不从1开始  xmin和ymin会匹配错误  处理麻烦
                y_cluster_max = item_cluster_select
                x_cluster_max = x_select
            if item_cluster_select <= y_cluster_min:  # 选择最小值
                y_cluster_min = item_cluster_select
                x_cluster_min = x_select

            item_separation_select = float(row_data['need_separation_accuracy'])
            x_select = int(row_data['epoch'])
            y_separation.append(item_separation_select)
            if item_separation_select >= y_separation_max:  # 选择最大值
                y_separation_max = item_separation_select
                x_separation_max = x_select
            if item_separation_select <= y_separation_min:  # 选择最小值
                y_separation_min = item_separation_select
                x_separation_min = x_select

            item_rand_score_select = float(row_data['avg_rand_score'])
            x_select = int(row_data['epoch'])
            y_rand_score.append(item_rand_score_select)
            if item_rand_score_select >= y_rand_score_max:  # 选择最大值  为什么不用numpy.argmax呢？  因为epoch可# 能不从1开始  xmin和ymin会匹配错误  处理麻烦
                y_rand_score_max = item_rand_score_select
                x_rand_score_max = x_select
            if item_rand_score_select <= y_rand_score_min:  # 选择最小值
                y_rand_score_min = item_rand_score_select
                x_rand_score_min = x_select

    print(y_cluster,"\n",
          y_separation,"\n",
            y_rand_score)


plt.figure(figsize=(12, 8), dpi=300)
plt.grid(True, linestyle='--', alpha=0.5)
# plt.ylim(0.8, 1.0)  # 设置y轴坐标范围
plt.xlabel('epoch')
plt.ylabel("accuracy")


plt.plot(x, y_separation)
plt.plot(x_separation_min,y_separation_min,'g-p',x_separation_max,y_separation_max,'r-p')
show_min = '[' + str(x_separation_min) + ' , ' + str(y_separation_min) + ']'
show_max = '[' + str(x_separation_max) + ' , ' + str(y_separation_max) + ']'
plt.annotate(show_min, xy=(x_separation_min, y_separation_min), xytext=(x_separation_min, y_separation_min))
plt.annotate(show_max, xy=(x_separation_max, y_separation_max), xytext=(x_separation_max, y_separation_max))

plt.plot(x, y_cluster)
plt.plot(x_cluster_min,y_cluster_min,'g-p',x_cluster_max,y_cluster_max,'r-p')
show_min = '[' + str(x_cluster_min) + ' , ' + str(y_cluster_min) + ']'
show_max = '[' + str(x_cluster_max) + ' , ' + str(y_cluster_max) + ']'
plt.annotate(show_min, xy=(x_cluster_min, y_cluster_min), xytext=(x_cluster_min, y_cluster_min))
plt.annotate(show_max, xy=(x_cluster_max, y_cluster_max), xytext=(x_cluster_max, y_cluster_max))

plt.plot(x, y_rand_score)
plt.plot(x_rand_score_min,y_cluster_min,'g-p',x_rand_score_max,y_rand_score_max,'r-p')
show_min = '[' + str(x_rand_score_min) + ' , ' + str(y_rand_score_min) + ']'
show_max = '[' + str(x_rand_score_max) + ' , ' + str(y_rand_score_max) + ']'
plt.annotate(show_min, xy=(x_rand_score_min, y_rand_score_min), xytext=(x_rand_score_min, y_rand_score_min))
plt.annotate(show_max, xy=(x_rand_score_max, y_rand_score_max), xytext=(x_rand_score_max, y_rand_score_max))

plt.legend(['n_cluster_accuracy', 'need_separation_accuracy', 'avg_rand_score'], loc='lower right')

plt.savefig(args.out_dir+'/'+str(ext[0])+args.mode+'.jpg', dpi=300)
