#!/usr/bin/env python
# -*- coding=utf8 -*-

import numpy as np
from scipy import interp, sparse
import matplotlib as mpl
from PIL import ImageFont, Image, ImageDraw

mpl.use('Agg')
import faiss
import matplotlib.pyplot as plt
# from cython.parallel import prange, parallel, threadid
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_curve, auc
# import ImageFont,Image,ImageDraw
import sys, os, datetime, random, math
from sklearn.preprocessing import normalize
from ctypes import *
import argparse
from multiprocessing.dummy import Pool as ThreadPool

file_path = os.path.abspath(__file__)
dir_name = os.path.dirname(file_path)
print(dir_name)
libranker = CDLL(os.path.join(dir_name, 'libranker3.so'))


def time():
    return datetime.datetime.now()


def todict(array, kind, dest=[]):
    if kind == 1:
        return dict(zip(array, [1] * len(array)))
    elif kind == 0:
        return dict(zip(array, range(len(array))))
    else:
        return dict(zip(array, dest))


def checkexists(v, dic):
    return dic[v] if v in dic else -1


def getlabels(file):
    global GroundTruth
    lines = open(file).read().replace("\n", " ").split()
    size = len(lines)
    labels = np.array(lines).reshape((-1, 3))
    gallery = np.where(labels[:, 2] == "0")[0]
    query = np.where(labels[:, 2] == "1")[0]
    dic = todict(labels[gallery][:, 1], 0)
    vfunc = np.vectorize(checkexists)
    GroundTruth = vfunc(labels[query][:, 1], dic)
    print("gallery", len(gallery), "query", len(query))
    return labels, gallery, query


def readFeatures(filename, size, dtype="cos"):
    if ".npy" in filename:
        feats = np.load(filename)
    elif ".bin" in filename:
        # loader=CDLL('./featloader.so')
        shape = np.zeros(10).astype("i")
        libranker.BinFeatGetInfo(filename, shape.ctypes.data_as(
            POINTER(c_int)))  # extern "C"  void getinfo( const char* filename,int* shape )
        if shape[0] != size:
            print("feature size not match!")
            sys.exit()
        length = 1 * shape[0] * shape[1]
        feats = np.zeros(length).astype("f")
        libranker.BinFeatReadData(filename, 0, size, feats.ctypes.data_as(
            POINTER(c_float)))  # extern "C"  void read( const char* filename, float* feats)
        feats = feats.reshape((size, -1))
    else:
        flines = open(filename).read()
        fs = np.fromstring(flines, "f", -1, " ")
        feats = fs.reshape((size, -1))
    print("distance type =", dtype)
    feats = normalize(feats) if dtype == "cos" else feats

    return feats


def score(dist):
    return -(dist ** 2 - 2.0) / 2


def saveData(tpr, fpr, thresholds, filename):
    np.savez(filename, tpr=tpr, fpr=fpr, thresholds=thresholds)


def loadData(filename):
    try:
        npzfiles = np.load(filename)
        tpr = npzfiles["tpr"]
        fpr = npzfiles["fpr"]
        thresholds = npzfiles["thresholds"]
        return tpr, fpr, thresholds
    except:
        print("Need to update npz file for newer version")
        sys.exit()


def getScores(features, labels, gallery, query, kind, GPUNO):
    Query = features[query]
    Gallery = features[gallery]
    querys = labels[query]
    gallerys = labels[gallery]
    print("feature read!", time())

    dim = Query.shape[1]
    #    topk = 25 #min(int(10**6/len(Query)),len(Gallery)-1)
    if kind == 0:
        topk = len(Gallery)
    else:
        topk = 25
    particle = 1000
    #    index=np.zeros(len(Query)*topk,"i")
    #    mat=np.zeros(len(Query)*topk,"f")# return value is row major while numpy deal with it in colomn major way
    fpr = np.zeros(2 * particle + 5, "f")
    tpr = np.zeros(2 * particle + 5, "f")
    thresholds = np.zeros(2 * particle + 5, "f")
    qlabels = np.array(querys[:, 1], "int32")
    glabels = np.array(gallerys[:, 1], "int32")
    #    print qlabels.shape,glabels.shape
    #    print qlabels[:10],glabels[:10]
    print("ready for gpu calculating", time())
    t1 = time()
    #    print qlabels
    libranker.matrixMulTopK(Query.reshape(-1).ctypes.data_as(POINTER(c_float)),
                            Gallery.reshape(-1).ctypes.data_as(POINTER(c_float)),
                            len(query), len(gallery), dim, topk, qlabels.ctypes.data_as(POINTER(c_int)),
                            glabels.ctypes.data_as(POINTER(c_int)),
                            tpr.ctypes.data_as(POINTER(c_float)), fpr.ctypes.data_as(POINTER(c_float)),
                            thresholds.ctypes.data_as(POINTER(c_float)), particle, GPUNO)
    t2 = time()
    print(t2 - t1)
    print("scores calculated", time())
    # shape=(len(query),len(gallery))

    return tpr, fpr, thresholds


def getAwhenB(bvalue, A, B, order=0):
    if order == 0:
        pos = np.searchsorted(B, bvalue)
        print('pos',pos)
    else:
        pos = len(A) - 1 - np.searchsorted(B, bvalue, sorter=range(len(B), 0, -1))
    if pos >= len(A) or pos < 0:
        print(A, bvalue)
    print('pos',pos)
    return A[pos], pos


def plot(tpr, fpr, thresholds, title, style, params, name):
    firstvalue = np.searchsorted(fpr, 0, side="right")
    # if params % 10 == 1:
    #     score = getAwhenB(1e-6, tpr, fpr)
    # else:
    #score, index = getAwhenB(1e-2, tpr, fpr)

    print("fpr tpr calculated", time())
    # plt.subplot(params)
    plt.title(name)
    #print('firstvalue',firstvalue)
    #print(fpr[firstvalue], fpr[(firstvalue-10):(firstvalue+1000)])
    #plt.plot([fpr[firstvalue] / 10, fpr[firstvalue]], [tpr[firstvalue], tpr[firstvalue]], style)
    plt.plot(fpr[:-5], tpr[:-5], style) #,
    #         label=title + " FPR=1e-2 TPR={}, thresh:{:.4f}".format(str(round(score, 3)), thresholds[index]))
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlim([1e-8, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(name)
    plt.legend(loc="best")
    plt.grid(True)

    print(fpr, tpr)
    print(title, ":")


def getTitle(lbfn, kind):
    key = lbfn.split("_")[-1]
    GName = {"xj": "xinjiang", "4w": "HQXJ_4w", "sc": "shangchao", "cleaned8": "xinjiangHQ"}
    # tname = GName[key] if GName.has_key(key) else key
    tname = GName[key] if key in GName else key
    flag = "N" if kind == 1 else "1"
    title = tname + ' 1:' + flag
    return title


def getsavename(kind):
    name = "top1" if kind == 1 else "allpairs"
    suffix = 0
    while os.path.exists(name + "_" + str(suffix) + ".png"):
        suffix += 1
    savename = name + "_" + str(suffix) + ".png"
    return savename


def drawbadcase(features, labels, g, q, kind, GPUNO, drawlow, drawrecall):
    global GroundTruth
    topk = 1000
    query = features[q]
    gallery = features[g]
    labels_g, labels_q = labels[g][:, 1], labels[q][:, 1]  # 筛选出label
    path_g, path_q = labels[g][:, 0], labels[q][:, 0]  # 筛选出文件路径
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = GPUNO
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, query.shape[1], flat_config)
    index.add(gallery)
    mat, index = index.search(query, topk)
    mat = 1 - mat / 2  # 将L2距离转化为余弦距离
    pairs = []
    cnt = 0
    gdic = GroundTruth
    if drawrecall:
        top1 = index[:, 0]
        wh = np.where(labels_g[top1] != labels_q[np.arange(q.shape[0])])[0]
        top1 = index[:, 0][wh]
        for i in range(wh.shape[0]):
            if GroundTruth[wh[i]] == -1:
                continue
            p1 = path_q[wh[i]]
            p2 = path_g[gdic[wh[i]]]
            s2 = cos_dist(query[wh[i]], gallery[gdic[wh[i]]]) if gdic[wh[i]] != -1 else 0
            p3 = path_g[top1[i]]
            s3 = mat[i][0]
            s3 = cos_dist(query[wh[i]], gallery[top1[i]])
            merge(p1, p2, p3, s2, s3, "FN_error_%d.jpg" % i)

    if drawlow:
        wh = np.where(
            (mat > drawlow) & (labels_q[np.tile(np.arange(labels_q.shape[0]), (topk, 1)).T] != labels_g[index]))
        ys = index[wh]
        scores = mat[wh]
        if os.path.exists("error_file.log"):
            os.remove("error_file.log")
        for i in range(wh[0].shape[0]):
            p1 = path_q[wh[0][i]]
            p2 = path_g[gdic[wh[0][i]]] if gdic[wh[0][i]] != -1 else ""
            s2 = cos_dist(query[wh[0][i]], gallery[gdic[wh[0][i]]]) if gdic[wh[0][i]] != -1 else 0
            # if i == 5960:
            #     print(p1, p2, wh[0][i])
            p3 = path_g[ys[i]]
            merge(p1, p2, p3, s2, scores[i], "FP_error_%d.jpg" % i)
            with open("error_file.log", "a") as f:
                f.write("\n".join([p1, p2, p3]))
                f.write("\n***********************FP_error_%d.jpg************************\n\n" % i)


def cos_dist(a, b):
    return np.sum(a * b)


def merge(pic1, pic2, pic3, info1, info2, output):
    img1 = Image.open(pic1)
    img2 = Image.open(pic2)
    img3 = Image.open(pic3)

    height = max(img1.size[1], img2.size[1], img3.size[1], 200)
    # weight = max(img1.size[0], img2.size[0], img3.size[0], 200)

    merge_img = Image.new('RGB', (1000, height + 50), 0xffffff)
    merge_img = paste(merge_img, pic1, (50, 20))
    merge_img = paste(merge_img, pic2, (450, 20))
    merge_img = paste(merge_img, pic3, (850, 20))
    draw = ImageDraw.Draw(merge_img)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf', 15)
    draw.text((25, 0), "query", fill="red", font=font)
    draw.text((25, height + 20), os.path.basename(pic1), fill="red", font=font)

    draw.text((400, 0), "groundtruth: " + str(round(info1, 3)), fill="red", font=font)
    draw.text((400, height + 20), os.path.basename(pic2), fill="red", font=font)

    draw.text((800, 0), "predict : " + str(round(info2, 3)), fill="red", font=font)
    draw.text((800, height + 20), os.path.basename(pic3), fill="red", font=font)
    merge_img.save(output, quality=100)


def paste(img, pic, pos):
    if len(pic) == 0:
        return img
    try:
        img1 = Image.open(pic)
        img.paste(img1, pos)
    except:
        print("can not find img", pic)
        return img
    return img


def show(lbfn, kind):
    #    fig=plt.gcf()
    #    fig.set_size_inches(24,12)
    #    fig.savefig(getsavename(kind),dpi=100)
    plt.savefig(getsavename(kind), dpi=100)
    plt.show()


def getStyle(cnt):
    color = ["r", "b", "g", "c", "y", "k", "m"]
    marker = ["o", "v", "s", "x", "+", "."]
    linestyle = ["", "--", ":"]
    return color[cnt % 7] + linestyle[cnt % 3]


def getlabelfile():
    lists = os.popen('ls |grep labels_').read().splitlines()

    if len(lists) == 1:
        return lists[0]
    elif len(lists) == 0:
        print("no label file found in this directory")
        return ""
    else:
        return sorted(lists)[-1]


def main():
    print("program started!", time())
    # labelfile = getlabelfile()
    parser = argparse.ArgumentParser(description='shows the roc curve with provided feature files')
    parser.add_argument('--feature_list', type=str, required=True, nargs='+', help='list of feature files to evaluate.')
    parser.add_argument('-k', '--kind', type=int, required=True,
                        help='choose with kind to draw. 0 stands for 1:1 pairs and 1 stands for 1:n pairs. Default 1')
    parser.add_argument('-l', '--label_list', type=str, required=True, help='label file to judge upon. Default ')
    parser.add_argument('-d', '--drawlow', nargs='?', const=1e-8, type=float,
                        help='false accepting cases at given fpr rate')
    parser.add_argument('-r', '--drawrecall', nargs='?', type=int, const=1, help='draw wrong top1 cases.')
    parser.add_argument('-g', '--gpu', type=int, help='choose whether to use gpu for calculation. Default use gpu 0. ')
    parser.add_argument('-m', '--merge', nargs='?', type=str, const="a",
                        help='choose whether to show merged roc curve. default: a, which means every pairwise combination')
    parser.set_defaults(kind=0, drawlow=0, drawrecall=0, gpu=3, merge="")

    args = parser.parse_args()
    print(args)
    print("finish arg parsing!", time())
    labelfile = args.label_list
    featlist = args.feature_list
    kind = args.kind
    drawlow = args.drawlow
    drawrecall = args.drawrecall
    merge = args.merge
    GPUNO = args.gpu
    print("label =", labelfile)
    print("GPU =", GPUNO)
    # subplot = 221
    need_draw = drawlow or drawrecall
    labels, gallery, query = getlabels(labelfile)
    # races = np.array(open("ins_races").readlines(), "int32")
    plt.figure(figsize=(20, 12), dpi=100)
    # names = ["all races 1:1", "the yellow 1:1", "the black 1:1", "the white 1:1"]
    names = ["1:1"]

    for cnt, filename in enumerate(featlist):
        features = None
        for i, name in enumerate(names):
            print(i, name)
            rocfile = filename + "_" + name + ".npz"
            if os.path.exists(rocfile) and not need_draw:
                npzfile = np.load(rocfile)
                tpr, fpr, thresholds = loadData(rocfile)
            else:
                features = readFeatures(filename, len(labels)) if features is None else features
                np.save(filename, features)
                # if i != 0:
                #     g = gallery[np.where(races[gallery] == i)]
                #     q = query[np.where(races[query] == i)]
                # else:
                g = gallery
                q = query

                tpr, fpr, thresholds = getScores(features, labels, g, q, kind, GPUNO)

                if drawlow:
                    drawlow, _ = getAwhenB(drawlow, thresholds, fpr) if drawlow else 0
                    print("drawlow:{}, threshold:{}, fpr:{}".format(drawlow, thresholds, fpr))
                    drawbadcase(features, labels, g, q, kind, GPUNO, drawlow, drawrecall)
                if drawrecall:
                    print("drawrecall:{}, threshold:{}, fpr:{}".format(drawrecall, thresholds, fpr))
                    drawbadcase(features, labels, g, q, kind, GPUNO, drawlow, drawrecall)

                saveData(tpr, fpr, thresholds, rocfile)

            # print(subplot + i)
            plot(tpr, fpr, thresholds, str(cnt) + ":" + filename, getStyle(cnt), i, name)

    show(labelfile, kind)


if __name__ == '__main__':
    main()
