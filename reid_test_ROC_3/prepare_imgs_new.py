# coding=UTF-8
import cv2
import sys
import threading
import time
import numpy as np
import pickle
import time
import json
from easydict import EasyDict as edict 

#从线程里按顺序从libs文件里读，然后存图片和名称

t0 = time.time()
cfg = edict(json.loads(open(sys.argv[1]).read()))

thread_num = 20
img_names=[]
imgs=[]

def process_func(imgfn):
    #print(imgfn)
    imgfn = imgfn.strip().split()[0]
    img = cv2.imread(imgfn)
    #print(img.shape)
    img = cv2.resize(img, (cfg.input_shape[2],cfg.input_shape[1]))
    img = np.transpose(img, (2,0,1))#hwc to chw
    #print('img3',img.shape)
    img_names.append(imgfn)
    imgs.append(img)
    #print(img_names)



def get_img_path_generate(filename):
    #filename=sys.argv[1]
    with open(filename, 'r') as f:
        index = 0
        for line in f:
            index += 1
            if index % 500 == 0:
                print( 'execute %s line at %s' % (index, time.time()))
            if not line:
                print( 'line %s is empty "\t"' % index)
                continue
            yield line

lock = threading.Lock()
def loop(line):
    print('thread %s is running...' % threading.current_thread().name)

    while True:
        try:
            with lock:
                img_path =  next(line)
        except StopIteration:
            break
        try:
            process_func(img_path)
        except:
            print('exceptfail\t%s' % img_path)
    print('thread %s is end...' % threading.current_thread().name)

def run_func(filename):
    img_gen = get_img_path_generate(filename)
    t_objs = []
    for i in range(0, thread_num):
        t = threading.Thread(target=loop, name='LoopThread %s' % i, args=(img_gen,))
        t.start()
        t_objs.append(t)

    for t in t_objs:
        t.join()

#imgs=[]
#img_names=[]
def save_func(imgs, img_names,save_img_path,save_img_names_path):
    print('len(img_name)',len(img_names))
    print('len(imgs)',len(imgs))
    imgs = np.array(imgs) 
    print(imgs.shape)
    np.save(save_img_path,imgs)
    pickle.dump(img_names,open(save_img_names_path,'wb'))

if 1:
    run_func(cfg.img_info)
    save_func(imgs,img_names,cfg.img_path,cfg.imagename)
