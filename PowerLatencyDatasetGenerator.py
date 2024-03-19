from PowerLatencySampler import getLayerwisePowerLatency
from operations import database_spawn
from checker import get_model_list
import os
import time

'''
To change the number of data samples collected per image, change the parameters defined in `PowerLatencySampler.py` script.
'''
#### Parameters #### 

# Define the mode to run the experiment
# Make sure the mode specified is alredy set
MODE = 0
ModeS = f"Mode{MODE}"

# Define the device for the evaluation : {"cpu", "cuda:0"}
Device = "cpu"
if "cuda" in Device:
    DeviceS = "gpu"
elif Device == "cpu":
    DeviceS = "cpu"

# Define sample paths 
sample_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']

def modeInfoExtractor():
    modeValid = False
    Configs = []
    line_idx = 0
    command = "cp /etc/nvpmodel.conf ."
    os.system(command)
    confFile = open("nvpmodel.conf", "r")
    content = confFile.readlines()
    confFile.close()
    for i,line in enumerate(content):
        if not line[0] == "#":
            if "POWER_MODEL ID" in line:
                lineMode = int(line.split("=")[1].split(" ")[0])
                if lineMode == MODE:
                    modeValid = True
                    line_idx = i
                    while (content[line_idx].strip() != ''):
                        Configs.append(content[line_idx])
                        line_idx +=1
                    break
    if modeValid:

        print(f"Mode{lineMode} found")
        if not os.path.exists(f"Dataset/{ModeS}/{DeviceS}"):
            os.makedirs(f"Dataset/{ModeS}/{DeviceS}")
        ConfFilePath = f"Dataset/{ModeS}/{ModeS}.conf"
        confFile = open(ConfFilePath,"w")
        confFile.writelines(Configs)
        confFile.close()

        op_executor = database_spawn(preprocess=True)

        model_list = get_model_list()

        for model_name in model_list:

            # Run power and latency monitoring and create datasets for the given image set
            getLayerwisePowerLatency(op_executor,model_name,Device,sample_paths,ModeS)

    else:
        print("Defined mode not found")

if __name__ == "__main__":
    try:
        t1 = time.time()
        modeInfoExtractor()
        t2 = time.time()
        print(f"Time taken for Mode{MODE}: {t2-t1} seconds")
    except Exception as e:
        print("Error found: ",e)
