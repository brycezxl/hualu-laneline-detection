import os
import tqdm
import cv2
import numpy as np
import random


def print_label(label, com):
    print(com, end=" ")
    total = 0
    for m in range(len(label)):
        total += label[m]
    for m in range(len(label)):
        print("%d: %.6f%% " % (m, label[m] / total * 100), end="")
    print("")


# path = r"D:\Projects\hualu-laneline-detection\dataset\hualu_laneline\testB\test"
path1 = r"/home/zxl/hualu-laneline-detection/data/train_label"
path2 = r"/home/zxl/hualu-laneline-detection/data/train_pre_tag"
label_dic = np.zeros(20)
occur_dic = np.zeros(20)

for (path, pic_list) in [(path1, os.listdir(path1)), (path2, os.listdir(path2))]:
    random.shuffle(pic_list)
    for n, p in tqdm.tqdm(enumerate(pic_list), total=len(pic_list)):
        image = cv2.imread(os.path.join(path, p), cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (int(image.shape[0] / 2), int(image.shape[1] / 2)), interpolation=cv2.INTER_CUBIC)
        tmp = np.zeros(20)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                tmp[image[i, j]] += 1
        label_dic += tmp
        occur_dic += [1 if tmp[i] > 0 else 0 for i in range(20)]
        if (n + 1) % 200 == 0:
            print("%d / %d" % (n + 1, len(pic_list)))
            print_label(label_dic, "[pixel]")
            print_label(occur_dic, "[occur]")
