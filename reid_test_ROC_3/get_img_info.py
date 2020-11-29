import sys
import os
import random
import pickle
import json
from easydict import EasyDict as edict

cfg = edict(json.loads(open(sys.argv[1]).read()))

query_list_path = cfg.query_list #'query.list'
distractor_list_path = cfg.distractor_list

new_list_path = cfg.img_info #'img_query.list'
new_list = open(new_list_path,'w')

def write_list_func1(old_list_path, new_list, query_or_distractor):
    old_list = [line.strip().split() for line in open(old_list_path)]
    img_num= len(old_list)
    for i in range(img_num):
        img_info = old_list[i]
        img_path = img_info[0]
        if query_or_distractor=='1':
            img_path = os.path.join(os.path.dirname(old_list_path),'query',img_path)
        else:
            img_path = os.path.join(os.path.dirname(old_list_path),'distractor',img_path)
        line = img_path + " " + img_info[1] + " " + query_or_distractor+"\n"
        new_list.write(line)
        
write_list_func1(query_list_path, new_list, "1")
write_list_func1(distractor_list_path, new_list, "0")

#lbs