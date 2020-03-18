#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: path_test.py
# Created Date: Sunday March 8th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 8th March 2020 2:27:30 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
from pathlib import Path

if __name__ == "__main__":
    print(Path("wocao/nidaye/wocao").parent)