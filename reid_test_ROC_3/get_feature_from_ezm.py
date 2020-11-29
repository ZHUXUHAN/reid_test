#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: xiaozhang
# Created Time : 2019-09-28
# File Name: get_feature.py
# Description:
"""
import mxnet as mx
from sklearn import preprocessing
import pickle
import sys
from easydict import EasyDict as edict
import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
import os
#import mxnet as mx
import time
import json 
import math
#import time
t0 = time.time()
print(t0)

cfg = edict(json.loads(open(sys.argv[1]).read()))

def process_feat_infos(imgname_feat, feat_infos, img_datasets_path):
    for feat_info in feat_infos:
        feat_info_split = feat_info.strip().split(' ')
        imgname = os.path.join(img_datasets_path, feat_info_split[0].strip().split('/')[-1])
        feat = np.array([float(i) for i in feat_info_split[2:]])
        imgname_feat[imgname] = feat
    return imgname_feat

if 1:
    query_feat_infos = open(sys.argv[2], encoding='utf-8').readlines()
    distractor_feat_infos = open(sys.argv[3], encoding='utf-8').readlines()
    img_datasets_path = sys.argv[4]
    imgname_feat = {}
    imgname_feat = process_feat_infos(imgname_feat, query_feat_infos, img_datasets_path+'/query')
    imgname_feat = process_feat_infos(imgname_feat, distractor_feat_infos, img_datasets_path+'/distractor')
    img_infos = open(cfg.img_info, 'r')
    features = []

    for img_info in img_infos:
        imgname = img_info.strip().split()[0]
        features.append(imgname_feat[imgname])
    features = np.array(features).astype(np.float32)
    np.save(cfg.feat_path, features)
t1 = time.time()
print('extracted all features costs', t1-t0)
