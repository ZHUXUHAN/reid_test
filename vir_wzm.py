import pickle
import sys
import numpy as np 
import time
import json
from skimage import io
from sklearn import preprocessing
import cv2
import random
def process(id_key,lst):
    ids = []
    for key in lst:
        ids.append(id_key[key])
    ids = np.array(ids)
    return ids

#distractor_feature = preprocessing.normalize(np.load(".personD_distractor.npy"))
distarctor_imagename = pickle.load(open("./img_info/imgs_name_distractor_personD",'rb'),encoding='utf8')
distractor_id_key = pickle.load(open("./test_generate/imgid_distractor_personD",'rb'),encoding='utf8')
distractor_id  = process(distractor_id_key, distarctor_imagename)

#print(distractor_id_key)

#query_feature = preprocessing.normalize(np.load(".personD_query.npy"))
#query_imagename = pickle.load(open("./img_info/imgs_name_query_personD",'rb'),encoding='utf8')
#query_id_key = pickle.load(open("./test_generate/imgid_query_personD",'rb'),encoding='utf8')
#query_id = process(query_id_key, query_imagename)
#print(query_id_key)

#print(len(query_feature))
#print(len(query_imagename))
#print(len(query_id_key))
#print(len(query_id))
ran = random.randint(1,10)
e_id =distractor_id[0]
output_dir = "/mnt/sdb1/wzm/reid/gallery/"

#for i in range(len(query_imagename)):
#	print("-------"+str(i)+"-------")
#	print(query_imagename[i])
#	#print(distractor_id_key[i])
#	print(query_id[i])
#	label = query_id[i]
#	camid = query_imagename[i].split("/")[-1].split("-")[1]
#	pre = query_imagename[i].split("/")[-1].split("-")[-1]
#	fpath = output_dir + str(label) +"_c"+ str(camid) + "r1_"+pre
#	print(fpath)
#	img = cv2.imread(query_imagename[i])
#	io.imsave(fpath, img)


for i in range(len(distarctor_imagename)):
    
    print("-------"+str(i)+"-------")
    print(distarctor_imagename[i])
    #print(distractor_id_key[i])
    print(distractor_id[i])
    label = distractor_id[i]
    if label != e_id:
        ran = random.randint(1,10)
        e_id = label
    label = distractor_id[i]
    img_name = distarctor_imagename[i].split("/")[-1]
    if len(img_name)>10:
    	img_name = distarctor_imagename[i].split("/")[-1].split("-")[-1]
    fpath = output_dir + str(label) +"_c"+ str(ran) + "r1_"+img_name
    print(fpath)
    img = cv2.imread(distarctor_imagename[i])
    io.imsave(fpath, img)