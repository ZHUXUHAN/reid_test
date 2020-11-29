import sys
import os
import random
import pickle

old_list_path = sys.argv[1] #'query.list'
old_list = [line.strip().split('\t') for line in open(old_list_path)]

new_list_path = sys.argv[2] #'img_query.list'
new_list = open(new_list_path,'w')

def write_list_func(old_list, new_list, save_img_id_key_path):
    img_num= len(old_list)
    img_id_key = {}
    for i in range(img_num):
        print(i)
        img_info = old_list[i]
        img_path = os.path.join(img_info[0])
        img_id_key[img_path]=img_info[1]
        if i ==0 :
            line = img_path
        else:
            line = '\n'+img_path
        new_list.write(line)
    pickle.dump(img_id_key,open(save_img_id_key_path,'wb'))
 
save_img_id_key_path = sys.argv[3] 
write_list_func(old_list,new_list, save_img_id_key_path)

#save_img_id_key_path = sys.arg[3]
