# coding=UTF-8
import pickle
import sys
import numpy as np 
from easydict import EasyDict as edict
import time
import json

from sklearn import preprocessing

t0 = time.time()
cfg = edict(json.loads(open(sys.argv[1]).read()))

def process(id_key,lst):
    ids = []
    for key in lst:
        ids.append(int(id_key[key]))
    ids = np.array(ids)
    return ids

def cal_topn(q_feats,q_ids,d_feats,d_ids, max_rank=100):
    sim = np.dot(q_feats, d_feats.T)
    num_q, num_d = sim.shape
    print('num_q', num_q, 'num_d', num_d)
    indices = np.argsort(-sim, axis=1)
    matches = (d_ids[indices] == q_ids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    distractor_index = np.array(range(num_d))+1
    #import pdb;pdb.set_trace()
    for q_idx in range(num_q):
        #q_id = q_ids[q_idx]
        #order = indices[q_idx]

        orig_cmc = matches[q_idx]
        num_rel = orig_cmc.sum()
        if num_rel==0: print(q_idx,q_ids[q_idx],'num_rel', num_rel)
        cmc = orig_cmc.cumsum()
        tmp_cmc = orig_cmc.cumsum() #cmc
        #print(cmc[:10])
        cmc[cmc > 1] = 1
        #print(cmc)
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        #num_rel = orig_cmc.sum()
        #tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        #tmp_cmc = tmp_cmc[orig_cmc==1]
        tmp_cmc = np.asarray(tmp_cmc) 
        
        select_distractor_index = np.array(distractor_index) #distractor_index[orig_cmc==1]
        tmp_cmc = tmp_cmc/select_distractor_index
        #print(orig_cmc==1)
        #print('tmp_cmc.shape',tmp_cmc.shape)
        #print('select_distractor_index.shape',select_distractor_index.shape)
        #tmp_cmc = tmp_cmc/select_distractor_index
        #print('tmp_cmc.shape',tmp_cmc.shape)
        #print(tmp_cmc)
        tmp_cmc = tmp_cmc * orig_cmc
        
        if num_rel==0: print(q_idx,q_ids[q_idx],'num_rel', num_rel,)
        AP = tmp_cmc.sum() / num_rel
        #print('num_rel', num_rel)
        #print(tmp_cmc.sum())
        #print(AP)
        #print(q_idx, tmp_cmc.sum(), num_rel,AP)
        all_AP.append(AP)

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    
    mAP = np.mean(all_AP)
    print('top1', all_cmc[0])
    print('top5', all_cmc[4])
    print('top10', all_cmc[9])
    print('map', round(mAP,4))

distractor_feature = preprocessing.normalize(np.load(cfg.distractor_feat_path))
distarctor_imagename = pickle.load(open(cfg.distractor_imagename,'rb'),encoding='utf8')
distractor_id_key = pickle.load(open(cfg.distractor_id_key,'rb'),encoding='utf8')
distractor_id  = process(distractor_id_key, distarctor_imagename)

if cfg.dataset=="carback" or cfg.dataset=="carfront":
    query_hasplate_feature = preprocessing.normalize(np.load(cfg.query_feat_hasplate_path))
    query_noplate_feature = preprocessing.normalize(np.load(cfg.query_feat_noplate_path))
    
    query_imagename_hasplate = pickle.load(open(cfg.query_imagename_hasplate,'rb'),encoding='utf8')
    query_imagename_noplate = pickle.load(open(cfg.query_imagename_noplate,'rb'),encoding='utf8')

    query_id_key_hasplate = pickle.load(open(cfg.query_id_key_hasplate,'rb'),encoding='utf8')
    query_id_key_noplate = pickle.load(open(cfg.query_id_key_noplate,'rb'),encoding='utf8')
    
    query_id_hasplate = process(query_id_key_hasplate, query_imagename_hasplate)
    query_id_noplate = process(query_id_key_noplate, query_imagename_noplate)

    print("has_plate")
    cal_topn(query_hasplate_feature,query_id_hasplate,distractor_feature,distractor_id)
    print("no_plate")
    cal_topn(query_noplate_feature,query_id_noplate,distractor_feature,distractor_id)
else:
    query_feature = preprocessing.normalize(np.load(cfg.query_feat_path))
    query_imagename = pickle.load(open(cfg.query_imagename,'rb'),encoding='utf8')
    query_id_key = pickle.load(open(cfg.query_id_key,'rb'),encoding='utf8')
    query_id = process(query_id_key, query_imagename)
    print("cal_topn")
    cal_topn(query_feature,query_id,distractor_feature,distractor_id)


t1 = time.time()
print('cal topn costs', t1-t0)
