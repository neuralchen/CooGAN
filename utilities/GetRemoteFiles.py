import json
import os
from nodesInfo.sshupload import fileUploaderClass

def getModel(self):
    server = self.model_node.lower()
    with open('nodesInfo/nodes.json','r') as cf:
        nodestr = cf.read()
        nodeinf = json.loads(nodestr)
    if server == "localhost":
        return
    else:
        nodeinf = nodeinf[server]
        # makeFolder(self.logRootPath, self.version)
        currentProjectPath  = os.path.join(self.logRootPath, self.version)
        # makeFolder(currentProjectPath, "checkpoint")
        
        remoteFile          = nodeinf['basePath']+ self.version + "/checkpoint/" + "%d_LocalG.pth"%self.chechpoint_step
        localFile           = os.path.join(currentProjectPath,"checkpoint","%d_LocalG.pth"%self.chechpoint_step)
        if os.path.exists(localFile):
            print("checkpoint already exists")
        else:
            uploader        = fileUploaderClass(nodeinf["ip"],nodeinf["user"],nodeinf["passwd"])
            uploader.sshScpGet(remoteFile,localFile)
            print("success get the model from server %s"%nodeinf['ip'])