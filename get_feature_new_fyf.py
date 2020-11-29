#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: xiaozhang
# Created Time : 2019-09-28
# File Name: get_feature.py
# Description:
"""
import mxnet as mx
from sklearn.decomposition import PCA
import pickle
import sys
from easydict import EasyDict as edict
import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
#import mxnet as mx
import time
#import sklearn
#from sklearn.decomposition import PCA
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
model.bind(data_shapes=[('data', (cfg.test_batch_size, *cfg.input_shape))], for_training=False)

model.set_params(arg_params, aux_params)
print("load model")
#print(1)
#query_hasplate_lst = open(cfg.query_list_hasplate).readlines()
#query_noplate_lst = open(cfg.query_list_noplate).readlines()
#distractor_lst = open(cfg.distractor_list).readlines()
#print(2)


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
        #datas = np.ones((cfg.test_batch_size, *cfg.input_shape))
        #idxs=[]
        #for i in range(cfg.test_batch_size):
        #   idx = i + n*cfg.test_batch_size
        #   idxs.append(idx)
        #   img =  one_sample(idx, datalen, lst, cfg)
        #   datas[i,:,:,:]=img
        idx1 = n*cfg.test_batch_size
        idx2 = (n+1)*cfg.test_batch_size
        datas = all_data[idx1:idx2]
        #print(datas.shape)
        datas_nd = mx.nd.array(datas)
        db = mx.io.DataBatch(data=(datas_nd,))
        model.forward(db, is_train=False)
        embedding = model.get_outputs()[0].asnumpy()
        #print('embedding.shape', embedding.shape)
        #for m,idx in enumerate(idxs):
        #    if idx<datalen:
        #       feature_keys[lst[idx]]=embedding[m]
        #       #print(embedding[m].shape)
        embeddings.append(embedding)
    embeddings = np.concatenate(embeddings,axis=0)[:datalen]
    print('embedding.shape', embeddings.shape)
    return embeddings
if cfg.dataset=="carback" or cfg.dataset=="carfront":
    distractor_data = np.load(cfg.distractor_img_path)
    distractor_feature = get_feature(cfg,model,distractor_data)
    np.save(cfg.distractor_feat_path, distractor_feature)

    query_hasplate_data = np.load(cfg.query_img_hasplate_path)
    query_hasplate_feature = get_feature(cfg,model,query_hasplate_data)
    np.save(cfg.query_feat_hasplate_path, query_hasplate_feature)

    query_noplate_data = np.load(cfg.query_img_noplate_path)
    query_noplate_feature = get_feature(cfg,model,query_noplate_data)
    np.save(cfg.query_feat_noplate_path, query_noplate_feature)
else:
    distractor_data = np.load(cfg.distractor_img_path)
    distractor_feature = get_feature(cfg,model,distractor_data)
    print('@@@@@@@@@',cfg.distractor_feat_path)
    np.save(cfg.distractor_feat_path, distractor_feature) 

    query_data = np.load(cfg.query_img_path)
    query_feature = get_feature(cfg,model,query_data)
    np.save(cfg.query_feat_path, query_feature)
"""
query_hasplate_feature = get_feature(query_hasplate_lst,cfg,model)
query_noplate_feature = get_feature(query_noplate_lst,cfg,model)
distractor_feature = get_feature(distractor_lst,cfg,model)

pickle.dump(query_hasplate_feature, open(cfg.query_feat_hasplate_path, 'wb'))
pickle.dump(query_noplate_feature, open(cfg.query_feat_noplate_path, 'wb'))
pickle.dump(distractor_feature, open(cfg.distractor_feat_path, 'wb'))
"""

print('@@@@@@@@@',cfg.distractor_feat_path)
t1 = time.time()
print('extracted all features costs', t1-t0)
