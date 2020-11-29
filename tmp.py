#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: xiaozhang
# Created Time : 2020-05-27
# File Name: tmp.py
# Description:
"""
import shutil
import os
source_file = './reid_directions_dataset/img.list'
dataset_dir = './reid_directions_datasets'
data_lst='reid_direction_lst'
#f = open(data_lst,'w+')
flist = open(source_file,'r').readlines()
for lin in flist:
    try:
        lin = lin.strip()
        ids = os.path.basename(lin).split('_')[-2]
        id_path = os.path.join(dataset_dir,ids)
        if not os.path.exists(id_path):
            os.makedirs(id_path)
        direcs = lin.split('/')[-2]
        new_name = id_path+'/'+'_'.join(os.path.basename(lin).split('_')[:3])+'_'+direcs+'.jpg'
        shutil.copyfile(lin,new_name)
        print(new_name)
      #  print (ids,direcs)
    except Exception as e:
        print(lin,str(e))
        continue
