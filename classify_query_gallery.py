import sys
import os
import random
import pickle

old_list_path = sys.argv[1] #'query.list'
old_list = [line.strip() for line in open(old_list_path)]
sorted(old_list,key=lambda a:a[1])
query_list_path = '/mnt/sdd1/fyf/reid_test_haiqiang_new/test_datasets/person_ReID_test/reid_directions/dif_front_query/query.lst' #'img_query.list'
gallery_list_path = '/mnt/sdd1/fyf/reid_test_haiqiang_new/test_datasets/person_ReID_test/reid_directions/dif_front_query/gallery.lst' #'img_query.list'
def write_list_func(old_list,query_list_path,gallery_list_path):
    fq = open(query_list_path,'w+')
    fg = open(gallery_list_path,'w+')
    img_num= len(old_list)
    img_id_key = {}
    for i in range(img_num):
        img_info = old_list[i]
        img_path = img_info
        if not img_id_key.has_key(img_info.split('/')[-2]):
            img_id_key[img_info.split('/')[-2]]=[]
        img_id_key[img_info.split('/')[-2]].append(img_info)
    for k,v in img_id_key.items():
        sample = v
        flag = 0
        for fid in sample:
            if 'front' in fid and not flag:
                fg.write(fid+'\t'+str(k)+'\n')
                flag = 1
                continue
            if not 'front' in fid:
                fg.write(fid+'\t'+str(k)+'\n')
            else:
                fq.write(fid+'\t'+str(k)+'\n')

write_list_func(old_list,query_list_path,gallery_list_path)
