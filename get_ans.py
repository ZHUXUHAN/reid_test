# _*_ coding:utf-8 _*_
import matplotlib.pyplot as plt
filelst = './reid_directions_lst'
import shutil
import os
f = open(filelst)
label_dic = {}
for line in f.readlines():
        label = line.split('/')[-2]
        label_v = line.split('/')[-1].split('_')[-1].split('.')[0]
        if label_v == 'left' or label_v=='right':
            label_v = 'ce'
	#if not label_dic.has_key(label):
        if not label in label_dic:
            label_dic[label]=set()
        label_dic[label].add(label_v)
ancle_dic = {}
num=0
for k,v in label_dic.items():
    if 'front' in v and 'back' in v:
        num+=1
    key = len(v)
    if key == 1:
        continue
        try:
            shutil.rmtree(os.path.join('./reid_directions_datasets',str(k)))
        except Exception as e:
            print(str(e))
            continue
    if not key in  ancle_dic:
    #if not ancle_dic.has_key(key):
        ancle_dic[key]=0
    ancle_dic[key] += 1
print (num)
import pdb;pdb.set_trace()
count = ancle_dic.keys()
dis_c = ancle_dic.values()
plt.bar(count,dis_c)
plt.savefig('./save.jpg')
import pdb;pdb.set_trace()
