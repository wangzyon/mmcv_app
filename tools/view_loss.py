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
parser.add_argument("--mode", type=str, default='train')
parser.add_argument("--json_path", type=str, default='20220715_011955.log.json')
parser.add_argument("--out_dir", type=str,default='/volume/singal/SignalSeparation/workdir/result/')
args = parser.parse_args()

select = ['contrast_loss','classify_loss', 'loss'] # 选择绘图的指标
mode = args.mode  # 选择log文件中的模式
json_path = "/volume/singal/SignalSeparation/workdir/SeparationII_Con_mask/"+args.json_path
ext = os.path.splitext(args.json_path)

x = []  # 存放epoch
y_contrast_loss = []  # 存放指标
y_classify_loss = []  # 存放指标
y_loss = []

isFirst = True
with open(json_path, 'r') as f:
    for jsonstr in f.readlines():
        if isFirst:  # mmdetection生成的log  json文件第一行是配置信息  跳过
            isFirst = False
            continue
        row_data = json.loads(jsonstr)
        if row_data['mode'] == mode:  # 选择train或者val模式中的指标数据
            item_contrast_loss = float(row_data['contrast_loss'])
            x_select = int(row_data['epoch'])
            x.append(x_select)
            y_contrast_loss.append(item_contrast_loss)

            item_classify_loss = float(row_data['classify_loss'])
            y_classify_loss.append(item_classify_loss)

            item_loss = float(row_data['loss'])
            y_loss.append(item_loss)

    # print(y_contrast_loss,"/n",
    #           y_classify_loss,"/n",
    #           y_loss)


plt.figure(figsize=(12, 8), dpi=300)
plt.grid(True, linestyle='--', alpha=0.5)
# plt.ylim(0.8, 1.0)  # 设置y轴坐标范围
plt.xlabel('epoch')
plt.ylabel("loss")

plt.plot(x, y_contrast_loss)
plt.plot(x, y_classify_loss)
plt.plot(x, y_loss)

plt.legend(['contrast_loss', 'classify_loss', 'loss'], loc='upper right')

plt.savefig(args.out_dir+'/'+str(ext[0])+args.mode+'loss'+'.jpg', dpi=300)
