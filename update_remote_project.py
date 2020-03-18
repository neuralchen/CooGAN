#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: update_remote_project.py
# Created Date: Wednesday February 26th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 10th March 2020 1:11:22 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
import os
from pathlib import Path
import json
from utilities.sshupload import fileUploaderClass

scan_config={
    "ignore_dir":["test_logs","train_logs",".vscode","att_classification","comparison_results"],
    "ignore_file":[".gitignore","LICENSE","README.md","*.jpg","*.png","*.JPG","*.JPEG","*.PNG","*.pyc"],

}

if __name__ == "__main__":
    
    # wocao = os.stat("./train.py")
    # print(wocao.st_mtime)
    # # first time run this script.
    # # scan the root dir
    # for root,dirs,files in os.walk("./"):
    #     print(dirs)
    #     print(files)
    #     pass
    path = "./file_sync/filestate.json"
    files_dict = {}
    last_state = {}
    changed_files = []
    # if not Path("./file_sync/").exists():
    #     Path("./file_sync/").mkdir(parents=True)
    # else:
    #     if Path(path).exists():
    #         with open(path,'r') as cf:
    #             nodelocaltionstr = cf.read()
    #             last_state = json.loads(nodelocaltionstr)
    all_py = []           
    files = Path('.').glob('*.py') # */
    for item in files:
        temp = item.stat().st_mtime
        if item._str in last_state:
            last_mtime = last_state[item._str]
            if last_mtime != temp:
                changed_files.append(item._str)
                last_state[item._str] = temp
        else:
            changed_files.append(item._str)
            last_state[item._str] = temp
    
    with open(path, 'w') as cf:
        configjson  = json.dumps(last_state, indent=4)
        cf.writelines(configjson)
    
    print(changed_files)
    root_path = "/home/gdp/CXH/file_syntest/"

    wocaonima = fileUploaderClass("192.168.101.57","gdp","glass123456")
    for item in changed_files:
        localfile = item
        print("here %s"%item)
        remotefile = Path(root_path,item).as_posix()
        
        # wocaonima.sshScpPut(localfile,remotefile)