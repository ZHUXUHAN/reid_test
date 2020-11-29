# coding=UTF-8
import pickle
import sys
import numpy as np 
from easydict import EasyDict as edict
import time
import json
import cv2

from sklearn import preprocessing

t0 = time.time()
cfg = edict(json.loads(open(sys.argv[1]).read()))

def process(id_key,lst):
    ids = []
    for key in lst:
        ids.append(id_key[key])
    ids = np.array(ids)
    return ids

def cal_topn(q_feats,q_ids,d_feats,d_ids, cfg, max_rank=100):
    sim = np.dot(q_feats,d_feats.T)
    num_q, num_d = sim.shape
    print('num_q',num_q,'num_d',num_d)
    indices = np.argsort(-sim,axis=1)
    matches = (d_ids[indices]==q_ids[:,np.newaxis]).astype(np.int32)
    
    all_cmc = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        q_id = q_ids[q_idx]
        order = indices[q_idx]
        
        orig_cmc = matches[q_idx]
        cmc = orig_cmc.cumsum()
        cmc[cmc>1]=1
        
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    print('top1',all_cmc[0])
    print('top5',all_cmc[4])
    print('top10',all_cmc[9])
    
    distractor_data = np.load(cfg.distractor_img_path)
    distractor_data = np.transpose(distractor_data,(0,2,3,1))
    distractor_data = distractor_data[:,:,:,[2,1,0]]
    query_data = np.load(cfg.query_img_path)
    query_data = np.transpose(query_data,(0,2,3,1))
    query_data = query_data[:,:,:,[2,1,0]] 


    if 1:
        for i in range(num_q):
            q_id = q_ids[i]
            sim_q = sim[i,:]
            sim_temp ={}
            for j in range(num_d):
                d_id = d_ids[j]
                if d_id not in sim_temp.keys():
                    sim_temp[d_id] = [sim_q[j],j]
                elif sim_temp[d_id][0]<sim_q[j]:
                    sim_temp[d_id] = [sim_q[j],j]
            sorted_simi = sorted(sim_temp.items(),key=lambda x:x[1][0], reverse=True)

            #save_img
            q_img = query_data[i].copy()
            cv2.putText(q_img, str(q_id),(2,12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            #cv2.imwrite('./demo_imgs/'+'ori.jpg',q_img)
            demo_imgs_h=[q_img]
            demo_imgs = []
            for ii in range(10):
                if ii==0:
                    jj_range_start = 0
                    jj_range_end = 9
                else:
                    jj_range_start = (ii-1)*10+9
                    jj_range_end = ii*10+9
                for jj in range(jj_range_start, jj_range_end):
                    d_img = distractor_data[sorted_simi[jj][1][1]].copy()
                    text1 = "top"+str(jj+1)+": "+str(sorted_simi[jj][0])
                    text2 = "sim: "+str(round(sorted_simi[jj][1][0],3))
                    cv2.putText(d_img, text1, (2,12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                    cv2.putText(d_img, text2, (2,24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                    demo_imgs_h.append(d_img)
                demo_imgs.append(np.hstack(demo_imgs_h))
                demo_imgs_h = []
            final_demo_imgs = np.vstack(demo_imgs)
            cv2.imwrite('./demo_imgs/new_demo/'+str(i)+'.png',final_demo_imgs)










distractor_feature = preprocessing.normalize(np.load(cfg.distractor_feat_path))
#distarctor_imagename = pickle.load(open(cfg.distractor_imagename,'rb'),encoding='utf8')
distractor_id = np.array(pickle.load(open(cfg.distractor_id,'rb'),encoding='utf8'))
#distractor_id  = process(distractor_id_key, distarctor_imagename)

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
    #query_imagename = pickle.load(open(cfg.query_imagename,'rb'),encoding='utf8')
    query_id = np.array(pickle.load(open(cfg.query_id,'rb'),encoding='utf8'))
    #query_id = process(query_id_key, query_imagename)
    print("cal_topn")
    cal_topn(query_feature,query_id,distractor_feature,distractor_id,cfg)


t1 = time.time()
print('cal topn costs', t1-t0)
