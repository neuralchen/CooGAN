#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: SelectImages.py
# Created Date: Wednesday March 4th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 13th March 2020 2:39:50 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import cv2
import os
import numpy as np

import argparse

keytable={
    "left":2424832,
    "right":2555904,
    "esc":27,
    "space":32
}

def str2bool(v):
    return v.lower() in ('true')

def getParameters():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--path1', type=str, default="D:\\PatchFace\\PleaseWork\\test_logs\\SRU_128_modify_leakyrelu\\samples")
    parser.add_argument('--path2', type=str, default="D:\\PatchFace\\PleaseWork\\output\\128\\sample_testing")
    parser.add_argument('--path3', type=str, default="F:\\attgan_samples\\sample_testing_bai")
    parser.add_argument('--saveWhich', type=str, default="path1",choices=['path1', 'path2', 'path3'])
    parser.add_argument('--savePath', type=str, default="./comparison_results")
    parser.add_argument('--row', type=int, default=3)
    parser.add_argument('--winWidth', type=int, default=1932)
    parser.add_argument('--winHeight', type=int, default=128)
    return parser.parse_args()

if __name__ == "__main__":
    config = getParameters()
    savePath = config.savePath
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    height = config.winHeight * config.row
    cv2.namedWindow("Comparison",0)
    cv2.resizeWindow("Comparison", config.winWidth, height)
    imglist1 = []
    root1= ""
    for root,dirs,files in os.walk(config.path1):
        imglist1 = files
        root1= root
    imglist2 = []
    root2= ""
    for root,dirs,files in os.walk(config.path2):
        imglist2 = files
        root2= root
    
    imglist3 = []
    root3= ""
    for root,dirs,files in os.walk(config.path3):
        imglist3 = files
        root3= root
        
    font = cv2.FONT_HERSHEY_SIMPLEX

    
    total = len(imglist1)
    index = 0
    
    img1 = cv2.imread(os.path.join(root1, imglist1[index]))
    img2 = cv2.imread(os.path.join(root2, imglist2[index]))
    img3 = cv2.imread(os.path.join(root3, imglist3[index]))
    img3 = cv2.resize(img3, (config.winWidth, config.winHeight))
    # delta = img1 - img2
    # imgshow = np.concatenate((img1, img2,img2),axis=0)
    imgshow = np.concatenate((img1, img2, img3),axis=0)
    imgshow = cv2.putText(imgshow, str(imglist1[index]), (0, 30), font, 1, (255, 255, 255), 2)
    while(1):
        cv2.imshow('Comparison',imgshow)
        waitkey_num = cv2.waitKeyEx(20)
        # if waitkey_num != -1:
        #     print(waitkey_num)
        if waitkey_num == keytable["left"]:
            # print("Left")
            index -= 1
            if index<0:
                index = 0
            img1 = cv2.imread(os.path.join(root1, imglist1[index]))
            img2 = cv2.imread(os.path.join(root2, imglist2[index]))
            img3 = cv2.imread(os.path.join(root3, imglist3[index]))
            img3 = cv2.resize(img3, (config.winWidth, config.winHeight))
            # delta = img1 - img2
            # imgshow = np.concatenate((img1, img2, delta),axis=0)
            imgshow = np.concatenate((img1, img2, img3),axis=0)
            imgshow = cv2.putText(imgshow, str(imglist1[index]), (0, 30), font, 1, (255, 255, 255), 2)
        if waitkey_num == keytable["right"]:
            # print("Right")
            index += 1
            if index>= total:
                index = total-1
            img1 = cv2.imread(os.path.join(root1, imglist1[index]))
            img2 = cv2.imread(os.path.join(root2, imglist2[index]))
            img3 = cv2.imread(os.path.join(root3, imglist3[index]))
            img3 = cv2.resize(img3, (config.winWidth, config.winHeight))
            # delta = img1 - img2
            # imgshow = np.concatenate((img1, img2, delta),axis=0)
            imgshow = np.concatenate((img1, img2, img3),axis=0)
            imgshow = cv2.putText(imgshow, str(imglist1[index]), (0, 30), font, 1, (255, 255, 255), 2)
        if waitkey_num == keytable["space"]:
            print("Save image")
            cv2.imwrite(os.path.join(savePath, imglist1[index]),img1)
        if waitkey_num == keytable["esc"]:
            break
    cv2.destroyAllWindows()