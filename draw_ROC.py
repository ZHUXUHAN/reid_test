# coding=UTF-8
import pickle
import sys
import numpy as np 
from easydict import EasyDict as edict
import time
import json
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
plt.switch_backend('agg')
plt.figure(figsize=(10,10), facecolor=(1, 1, 1))
lw = 2
plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('FPR', fontsize=20)
plt.ylabel('TPR', fontsize=20)
plt.title('direction_ce_ROC_curve')


t0 = time.time()

print(sys.argv[1])
model_name = sys.argv[1].split(",")

def process(id_key,lst):
        ids = []
        for key in lst:
            ids.append(id_key[key])
        ids = np.array(ids)
        return ids
    
def cal_topn(q_feats,q_ids,d_feats,d_ids,name,idx,max_rank=100):
    sim = np.dot(q_feats,d_feats.T)
    num_q, num_d = sim.shape
    print('num_q',num_q,'num_d',num_d)
    sim_score = np.max(sim,axis=1)
    indices = np.argmax(sim,axis=1)
    matches = (d_ids[indices]==q_ids).astype(np.int32)
    print(matches.shape)
    print(sim_score.shape)
    fpr,tpr, _thresholds = metrics.roc_curve(matches,sim_score)
    
    print(metrics.auc(fpr,tpr))
    auc = ",auc:"+str(round(metrics.auc(fpr,tpr),4))
    
    plt.plot(fpr, tpr,linewidth=2,label=name+auc)
    plt.legend(loc='best')
    #print("-------",name)
    


for i in range(len(model_name)):
    cfg = edict(json.loads(open(model_name[i]).read()))
    print(str(i)+cfg.distractor_feat_path) 

    #name = model_name[i].split(".")[0].split("_")[1]
    name = model_name[i].split("/")[-1].split(".")[0].split("_")[-1]
    
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
        cal_topn(query_hasplate_feature,query_id_hasplate,distractor_feature,distractor_id,name)
        print("no_plate")
        cal_topn(query_noplate_feature,query_id_noplate,distractor_feature,distractor_id,name)
    else:
        query_feature = preprocessing.normalize(np.load(cfg.query_feat_path))
        query_imagename = pickle.load(open(cfg.query_imagename,'rb'),encoding='utf8')
        query_id_key = pickle.load(open(cfg.query_id_key,'rb'),encoding='utf8')
        query_id = process(query_id_key, query_imagename)
        print("cal_topn")
        cal_topn(query_feature,query_id,distractor_feature,distractor_id,name,i)
    
    
    t1 = time.time()
    print('cal topn costs', t1-t0)



plt.savefig('./roc_png/ROC_direction_ce.png')