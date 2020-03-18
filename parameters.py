#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: parameter.py
# Created Date: 2020.2.24
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 14th March 2020 12:49:20 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

import argparse

def str2bool(v):
    return v.lower() in ('true')

def getParameters():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--mode', type=str, default="test", choices=['train', 'finetune','test','debug'])
    parser.add_argument('--cuda', type=int, default=0)
    # training
    parser.add_argument('--version', type=str, default='SRU_256_modify_relu')
    parser.add_argument('--experimentDescription', type=str, default="continue to train 256 modify_relu")
    parser.add_argument('--trainYaml', type=str, default="train_256.yaml")

    # finetune
    parser.add_argument('--finetuneCheckpoint', type=int, default=95)

    # test
    parser.add_argument('--testVersion', type=str, default='SRU_128_modify_leakyrelu')
    parser.add_argument('--testScriptsName', type=str, default='common')
    parser.add_argument('--nodeName', type=str, default='4card',choices=['localhost', '4card', '8card','lyh','loc','localhost'])
    parser.add_argument('--testCheckpointEpoch', type=int, default=126) #822000 972000 906000
    parser.add_argument('--testBatchSize', type=int, default=16)
    parser.add_argument('--totalImg', type=int, default=2000)
    parser.add_argument('--saveTestImg', type=str2bool, default=True)
    parser.add_argument('--enableThresIntSetting', type=str2bool, default=True)
    parser.add_argument('--testThresInt', type=float, default=2)
    parser.add_argument('--specifiedTestImages', nargs='+', help='selected images for validation', 
            # '000121.jpg','000124.jpg','000129.jpg','000132.jpg','000135.jpg','001210.jpg','001316.jpg', 
            default=[183947])
    parser.add_argument('--useSpecifiedImage', type=str2bool, default=True)
    return parser.parse_args()

if __name__ == "__main__":
    from main import main
    config = getParameters()
    # start = config.test_chechpoint_step
    for i in range(1):
        print("chechpoint step %d"%config.test_chechpoint_step)
        main(config)
        config.test_chechpoint_step += 6000