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
#import mxnet as mx
import time
import json 
import math
#import time
t0 = time.time()
print(t0)

cfg = edict(json.loads(open(sys.argv[1]).read()))
ctxs = [mx.gpu(i) for i in cfg.gpus]
cfg.ctxs = ctxs
cfg.test_batch_size = cfg.batch_size*len(ctxs)
prefix, epoch = cfg.model_prefix.split(',')
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, int(epoch))
all_layers = sym.get_internals()
sym = all_layers[cfg.output_name+'_output']
model = mx.mod.Module(symbol=sym, context=cfg.ctxs, label_names = None)
model.bind(data_shapes=[('data', (cfg.test_batch_size,*cfg.input_shape))],for_training=False)

model.set_params(arg_params, aux_params)
print("load model")

def get_feature(cfg,model,all_data):
    if cfg.RGB_BGR=='RGB':
        all_data = all_data[:,(2,1,0),:,:]
    datalen = all_data.shape[0]
    #datas = np.ones((cfg.test_batch_size, *cfg.input_shape))
    num_batches=math.ceil(datalen/cfg.test_batch_size)
    print('num_batches',num_batches)
    #feature_keys ={}
    embeddings = []
    for n in range(num_batches):
        print(n)
        idx1 = n*cfg.test_batch_size
        idx2 = (n+1)*cfg.test_batch_size
        datas = all_data[idx1:idx2]
        #print(datas.shape)
        datas_nd = mx.nd.array(datas)
        db = mx.io.DataBatch(data=(datas_nd,))
        model.forward(db, is_train=False)
        embedding = model.get_outputs()[0].asnumpy()
        embeddings.append(embedding)
    embeddings = np.concatenate(embeddings,axis=0)[:datalen]
    print('embedding.shape', embeddings.shape)
    return embeddings

if 1:
    img_data = np.load(cfg.img_path)
    embeddings = get_feature(cfg,model,img_data)
    embeddings = preprocessing.normalize(embeddings)
    imgnames = np.array(pickle.load(open(cfg.imagename, 'rb'), encoding='utf8'))
    imgname_feat = {}
    for i, imgname in enumerate(imgnames):
        imgname_feat[imgname] = embeddings[i] #按名称存了feature
    img_infos = open(cfg.img_info, 'r')
    features = []
    for img_info in img_infos:
        imgname = img_info.strip().split()[0]
        features.append(imgname_feat[imgname]) #按libs里的图片名称顺序写入feature
    features = np.array(features)
    np.save(cfg.feat_path, features)
t1 = time.time()
print('extracted all features costs', t1-t0)
