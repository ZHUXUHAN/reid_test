import sys
import os
import random
import pickle

old_list_path = sys.argv[1] #'query.list'
old_list = [line.strip().split('\t') for line in open(old_list_path)]
sorted(old_list,key=lambda a:a[1])
query_list_path = sys.argv[2] #'img_query.list'
gallery_list_path = sys.argv[3] #'img_query.list'
import pdb;pdb.set_trace()
def write_list_func(old_list,query_list_path,gallery_list_path):
    fq = open(query_list_path,'w+')
    fg = open(gallery_list_path,'w+')
    img_num= len(old_list)
    img_id_key = {}
    flag = '-1'
    for i in range(img_num):
        img_info = old_list[i]
        img_path = os.path.join(img_info[0])
        if not img_id_key.has_key(img_info[1]):
            img_id_key[img_info[1]]=[]
        img_id_key[img_info[1]].append(img_info[0])
    for k,v in img_id_key.items():
        try:
            sample = random.sample(v,10)
        except:
            sample = v
        for fid in sample[:1]:
            fg.write(fid+'\t'+str(k)+'\n')
        for fid in sample[1:]:
            fq.write(fid+'\t'+str(k)+'\n')

write_list_func(old_list,query_list_path,gallery_list_path)
