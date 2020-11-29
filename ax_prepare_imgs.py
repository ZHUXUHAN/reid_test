import cv2
import sys
import threading
import time
import numpy as np
import pickle
import time
import json
from easydict import EasyDict as edict 
t0 = time.time()
cfg = edict(json.loads(open(sys.argv[1]).read()))

thread_num = 20
img_names=[]
imgs=[]

def process_func(imgfn):
    img = cv2.imread(imgfn.strip())
    img = cv2.resize(img, (cfg.input_shape[2],cfg.input_shape[1]))

    # add by anxiang
    # img[:, :, 0] = img[:, :, 0] - 103.
    # img[:, :, 1] = img[:, :, 1] - 116.
    # img[:, :, 2] = img[:, :, 2] - 123.
    # img = img * 0.01

    img = np.transpose(img, (2,0,1))#hwc to chw
    img_names.append(imgfn.strip())
    imgs.append(img)



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

    #print('len(img_name)',len(img_names))
    #print('len(imgs)',len(imgs))
    #imgs = np.array(imgs)
    #print(imgs.shape)
    #np.save(save_img_path,imgs)
    #pickle.dump(img_names,open(save_img_names_path,'wb'))

#imgs=[]
#img_names=[]
def save_func(imgs, img_names,save_img_path,save_img_names_path):
    print('len(img_name)',len(img_names))
    print('len(imgs)',len(imgs))
    imgs = np.array(imgs) 
    print(imgs.shape)
    np.save(save_img_path,imgs)
    pickle.dump(img_names,open(save_img_names_path,'wb'))


if cfg.dataset=="carback" or cfg.dataset=="carfront":
    run_func(cfg.query_list_hasplate)
    save_func(imgs,img_names,cfg.query_img_hasplate_path,cfg.query_imagename_hasplate)

    imgs=[]
    img_names=[]
    run_func(cfg.query_list_noplate)
    save_func(imgs,img_names,cfg.query_img_noplate_path,cfg.query_imagename_noplate)

    imgs=[]
    img_names=[]
    run_func(cfg.distractor_list)
    save_func(imgs,img_names,cfg.distractor_img_path,cfg.distractor_imagename)
else:
    run_func(cfg.query_list)
    save_func(imgs,img_names,cfg.query_img_path,cfg.query_imagename)

    imgs=[]
    img_names=[] 
    run_func(cfg.distractor_list)
    save_func(imgs,img_names,cfg.distractor_img_path,cfg.distractor_imagename)

t1 = time.time() 
print('prepeare_img costs', t1-t0)
