#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: main.py
# Created Date: 2020.2.24
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 16th March 2020 1:02:26 am
# Modified By: Chen Xuanhong
# Copyright (c) 2019 Shanghai Jiao Tong University
#############################################################

from    parameters import getParameters
from    utilities.reporter import Reporter
import  platform
import  os
import  json
import  shutil
from    utilities.json_config import *
from    utilities.yaml_config import getConfigYaml
from    utilities.sshupload import fileUploaderClass

def create_dirs(sys_state):
    # the base dir
    if not os.path.exists(sys_state["logRootPath"]):
            os.makedirs(sys_state["logRootPath"])

    # create dirs
    sys_state["projectRoot"]        = os.path.join(sys_state["logRootPath"], sys_state["version"])
    if not os.path.exists(sys_state["projectRoot"]):
        os.makedirs(sys_state["projectRoot"])
    
    sys_state["projectSummary"]     = os.path.join(sys_state["projectRoot"], "summary")
    if not os.path.exists(sys_state["projectSummary"]):
        os.makedirs(sys_state["projectSummary"])

    sys_state["projectCheckpoints"] = os.path.join(sys_state["projectRoot"], "checkpoints")
    if not os.path.exists(sys_state["projectCheckpoints"]):
        os.makedirs(sys_state["projectCheckpoints"])

    sys_state["projectSamples"]     = os.path.join(sys_state["projectRoot"], "samples")
    if not os.path.exists(sys_state["projectSamples"]):
        os.makedirs(sys_state["projectSamples"])

    sys_state["projectScripts"]     = os.path.join(sys_state["projectRoot"], "scripts")
    if not os.path.exists(sys_state["projectScripts"]):
        os.makedirs(sys_state["projectScripts"])
    
    sys_state["reporterPath"] = os.path.join(sys_state["projectRoot"],sys_state["version"]+"_report")

def main(config):
    ignoreKey = [
        "threads","ACCModel","logRootPath",
        "projectRoot","projectSummary","projectCheckpoints",
        "projectSamples","projectScripts","reporterPath",
        "specifiedTestImages","dataset_path","dataset_path_attr"
    ]
    sys_state = {}

    sys_state["threads"] = -1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)
    # For fast training

    # read system environment path
    env_config = read_config('env/config.json')
    env_config = env_config["path"]
    
    # Train mode
    if config.mode == "train":
        
        sys_state["version"]                = config.version
        sys_state["experimentDescription"]  = config.experimentDescription
        sys_state["mode"]                   = config.mode
        # read training configurations
        ymal_config = getConfigYaml(os.path.join(env_config["trainConfigPath"], config.trainYaml))
        for item in ymal_config.items():
            sys_state[item[0]] = item[1]

        # create dirs
        sys_state["logRootPath"]        = env_config["trainLogRoot"]
        create_dirs(sys_state)
        
        # create reporter file
        reporter = Reporter(sys_state["reporterPath"])

        # save the config json
        config_json = os.path.join(sys_state["projectRoot"], env_config["configJsonName"])
        write_config(config_json, sys_state)

        # get the dataset path
        sys_state["dataset_path"]        = env_config["datasetPath"][sys_state["dataset"].lower()]
        sys_state["dataset_path_attr"]   = env_config["datasetAttrPath"][sys_state["dataset"].lower()]

        # save the scripts
        # copy the scripts to the project dir 
        file1       = os.path.join(env_config["trainScriptsPath"], "trainer_%s.py"%sys_state["trainScriptName"])
        tgtfile1    = os.path.join(sys_state["projectScripts"], "trainer_%s.py"%sys_state["trainScriptName"])
        shutil.copyfile(file1,tgtfile1)

        file2       = os.path.join("./components", "%s.py"%sys_state["modelScriptName"])
        tgtfile2    = os.path.join(sys_state["projectScripts"], "%s.py"%sys_state["modelScriptName"])
        shutil.copyfile(file2,tgtfile2)

        file3       = os.path.join("./components/STU", "%s.py"%sys_state["stuScriptName"])
        tgtfile3    = os.path.join(sys_state["projectScripts"], "%s.py"%sys_state["stuScriptName"])
        shutil.copyfile(file3,tgtfile3)

        # display the training information
        moduleName  = "train_scripts.trainer_" + sys_state["trainScriptName"]
        print("Start to run training script: {}".format(moduleName))
        print("Traning version: %s"%sys_state["version"])
        print("Training Script Name: %s"%sys_state["trainScriptName"])
        print("Model Script Name: %s"%sys_state["modelScriptName"])
        print("STU Script Name: %s"%sys_state["stuScriptName"])
        print("Image Size: %d"%sys_state["imsize"])
        print("Image Crop Size: %d"%sys_state["imCropSize"])
        print("ThresInt: %d"%sys_state["thresInt"])
        print("D : G = %d : %d"%(sys_state["dStep"],sys_state["gStep"]))

        

        # Load the training script and start to train
        reporter.writeConfig(sys_state)
        package     = __import__(moduleName, fromlist=True)
        trainerClass= getattr(package, 'Trainer')
        trainer     = trainerClass(sys_state,reporter)
        trainer.train()

    elif config.mode == "finetune":
        sys_state["logRootPath"]    = env_config["trainLogRoot"]
        sys_state["version"]        = config.version
        sys_state["projectRoot"]    = os.path.join(sys_state["logRootPath"], sys_state["version"])

        config_json                 = os.path.join(sys_state["projectRoot"], env_config["configJsonName"])
        train_config                = read_config(config_json)
        for item in train_config.items():
            if item[0] in ignoreKey:
                pass
            else:
                sys_state[item[0]] = item[1]
        
        sys_state["mode"]           = config.mode
        create_dirs(sys_state)
        reporter = Reporter(sys_state["reporterPath"])
        # get the dataset path
        sys_state["dataset_path"]        = env_config["datasetPath"][sys_state["dataset"].lower()]
        sys_state["dataset_path_attr"]   = env_config["datasetAttrPath"][sys_state["dataset"].lower()]

        # display the training information
        moduleName  = "train_scripts.trainer_" + sys_state["trainScriptName"]
        print("Start to run training script: {}".format(moduleName))
        print("Traning version: %s"%sys_state["version"])
        print("Training Script Name: %s"%sys_state["trainScriptName"])
        print("Model Script Name: %s"%sys_state["modelScriptName"])
        print("STU Script Name: %s"%sys_state["stuScriptName"])
        print("Image Size: %d"%sys_state["imsize"])
        print("Image Crop Size: %d"%sys_state["imCropSize"])
        print("ThresInt: %d"%sys_state["thresInt"])
        print("D : G = %d : %d"%(sys_state["dStep"],sys_state["gStep"]))
        reporter.writeConfig(sys_state)
        package     = __import__(moduleName, fromlist=True)
        trainerClass= getattr(package, 'Trainer')
        trainer     = trainerClass(sys_state,reporter)
        trainer.train()

        
        
    elif config.mode == "test":
        sys_state["version"]        = config.testVersion
        sys_state["logRootPath"]    = env_config["testLogRoot"]
        sys_state["nodeName"]       = config.nodeName
        sys_state["testBatchSize"]  = config.testBatchSize
        sys_state["totalImg"]       = config.totalImg
        sys_state["saveTestImg"]    = config.saveTestImg
        sys_state["useSpecifiedImage"]= config.useSpecifiedImage
        if config.useSpecifiedImage:  
            sys_state["specifiedTestImages"]   = config.specifiedTestImages       
        # Create dirs
        create_dirs(sys_state)
        config_json = os.path.join(sys_state["projectRoot"], env_config["configJsonName"])
        
        # Read model_config.json from remote machine
        if sys_state["nodeName"]!="localhost":
            print("ready to fetch the %s from the server!"%config_json)
            nodeinf     = read_config(env_config["remoteNodeInfo"])
            nodeinf     = nodeinf[sys_state["nodeName"]]
            uploader    = fileUploaderClass(nodeinf["ip"],nodeinf["user"],nodeinf["passwd"])
            remotebase  = os.path.join(nodeinf['basePath'],"train_logs",sys_state["version"]).replace('\\','/')
            # Get the config.json
            print("ready to get the config.json...")
            remoteFile  = os.path.join(remotebase, env_config["configJsonName"]).replace('\\','/')
            localFile   = config_json
            uploader.sshScpGet(remoteFile,localFile)
            print("success get the config file from server %s"%nodeinf['ip'])

        # Read model_config.json
        json_obj    = read_config(config_json)
        for item in json_obj.items():
            # sys_state[item[0]] = item[1]
            if item[0] in ignoreKey:
                pass
            else:
                sys_state[item[0]] = item[1]
        
        # get the dataset path
        sys_state["dataset_path"]        = env_config["datasetPath"][sys_state["dataset"].lower()]
        sys_state["dataset_path_attr"]   = env_config["datasetAttrPath"][sys_state["dataset"].lower()]

        # Get checkpoint
        n_d         = sys_state["dStep"]
        data_len    = 182000// (sys_state["batchSize"] * (n_d + 1))
        # ckpt_prefix = "Epoch_(%d)_(%dof%d).ckpt"%(config.testCheckpointEpoch, data_len, data_len)
        ckpt_prefix = "Epoch_(%d).ckpt"%(config.testCheckpointEpoch)
        sys_state["ckpt_prefix"] = ckpt_prefix
            
        # Read scripts from remote machine
        if sys_state["nodeName"]!="localhost":
            # Get scripts
            remoteFile  = os.path.join(remotebase, "scripts", sys_state["modelScriptName"]+".py").replace('\\','/')
            localFile   = os.path.join(sys_state["projectScripts"], sys_state["modelScriptName"]+".py") 
            uploader.sshScpGet(remoteFile, localFile)
            print("Get the scripts:%s.py successfully"%sys_state["modelScriptName"])
            remoteFile  = os.path.join(remotebase, "scripts", sys_state["stuScriptName"]+".py").replace('\\','/')
            localFile   = os.path.join(sys_state["projectScripts"], sys_state["stuScriptName"]+".py") 
            uploader.sshScpGet(remoteFile, localFile)
            print("Get the scripts:%s.py successfully"%sys_state["stuScriptName"])
            # Get data-00000-of-00001
            localFile   = os.path.join(sys_state["projectCheckpoints"], ckpt_prefix + ".data-00000-of-00001")
            if not os.path.exists(localFile):
                remoteFile  = os.path.join(remotebase, "checkpoints", ckpt_prefix + ".data-00000-of-00001").replace('\\','/')
                uploader.sshScpGet(remoteFile, localFile, True)
                print("Get the %s file successfully"%(ckpt_prefix + ".data-00000-of-00001"))
            else:
                print("%s file exists"%(ckpt_prefix + ".data-00000-of-00001"))
            # Get index
            localFile   = os.path.join(sys_state["projectCheckpoints"], ckpt_prefix + ".index")
            if not os.path.exists(localFile):
                remoteFile  = os.path.join(remotebase, "checkpoints", ckpt_prefix + ".index").replace('\\','/')
                uploader.sshScpGet(remoteFile, localFile, True)
                print("Get the %s file successfully"%(ckpt_prefix + ".index"))
            else:
                print("%s file exists"%(ckpt_prefix + ".index"))
            # Get meta
            localFile   = os.path.join(sys_state["projectCheckpoints"], ckpt_prefix + ".meta")
            if not os.path.exists(localFile):
                remoteFile  = os.path.join(remotebase, "checkpoints", ckpt_prefix + ".meta").replace('\\','/')
                uploader.sshScpGet(remoteFile, localFile, True)
                print("Get the %s file successfully"%(ckpt_prefix + ".meta"))
            else:
                print("%s file exists"%(ckpt_prefix + ".meta"))
            
        # Get the test configurations
        sys_state["testScriptsName"]      = config.testScriptsName
        sys_state["batchSize"]            = config.testBatchSize
        sys_state["totalImg"]             = config.totalImg
        sys_state["saveTestImg"]          = config.saveTestImg
        sys_state["enableThresIntSetting"]= config.enableThresIntSetting
        sys_state["com_base"]             = "test_logs.%s.scripts."%sys_state["version"]

        # If thresInt setting enable, program will discard the thresInt in model_config.json
        if sys_state["enableThresIntSetting"]:
            sys_state["sampleThresInt"]   = config.testThresInt
        # else:
        #     sys_state["testThresInt"]   = sys_state["sampleThresInt"]
        # Create the reporter file
        reporter = Reporter(sys_state["reporterPath"])
        
        # Display the test information
        moduleName  = "test_scripts.tester_" + sys_state["testScriptsName"]
        print("Start to run test script: {}".format(moduleName))
        print("Test version: %s"%sys_state["version"])
        print("Test Script Name: %s"%sys_state["testScriptsName"])
        print("Model Script Name: %s"%sys_state["modelScriptName"])
        print("STU Script Name: %s"%sys_state["stuScriptName"])
        print("Image Size: %d"%sys_state["imsize"])
        print("Image Crop Size: %d"%sys_state["imCropSize"])
        print("testThresInt: %d"%sys_state["thresInt"])
        package     = __import__(moduleName, fromlist=True)
        testerClass = getattr(package, 'Tester')
        tester      = testerClass(sys_state,reporter)
        tester.test()

if __name__ == '__main__':
    config = getParameters()
    main(config)