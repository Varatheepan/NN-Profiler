from PowerLatencySampler import getLayerwisePowerLatency
from layer_stats.utils.operations import database_spawn
from layer_stats.utils.checker import get_model_list
import os
import time

'''
To change the number of data samples collected per image, change the parameters defined in `PowerLatencySampler.py` script.
'''
#### User defined parameters #### 

# Define the mode to run the experiment
# Make sure the mode specified is alredy set
MODE = 0
ModeS = f"Mode{MODE}"

# Define the device for the evaluation : {"cpu", "cuda:0"}
Device = "cpu"

EvaluateLatency = True
EvaluatePower = True        # Or other tegrastats parameters

# Define a custom set of parameters to be extracted from tegerastats. Defaults to current device power if left empty. 
OverWriteParams = []

# Define sample paths: Assumed to be in project_root/data/imagenet
sample_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']

######## Other Parameters ######### 

# Available Parameters to be sampled from tegrastats. Listed in the order params appear in the command output.
AvailParams = ["RAM", "SWAP","CPU","EMC_FREQ","MCPU","GPU","BCPU","VDD_SYS_GPU","VDD_SYS_SOC","VDD_SYS_CPU","VDD_SYS_DDR",]

''' Parameters which might be usful to capture for this project are listed. Check Tegrastats to find other parameters
"VDD_SYS_CPU"   : CPU Power usage      
"VDD_SYS_GPU"   : GPU Power usage 
"VDD_SYS_SOC"   : SOC Power usage 
"VDD_SYS_DDR"   : RAM Power usage 
"CPU"           : CPU utilization percentage and frequency 
"EMC_FREQ"      : RAM utilization percentage and frequency
"GPU"           : GPU Temperature
"BCPU"          : BCPU Temperature
"MCPU"          : MCPU Temperature
'''

# Assigin device name for data storing
if "cuda" in Device:
    DeviceS = "gpu"
    Parameters = ["VDD_SYS_GPU"]
elif Device == "cpu":
    DeviceS = "cpu"
    Parameters = ["VDD_SYS_CPU",]

# Overwrite the tegarstasts parameters
if len(OverWriteParams):
    Paramtemp = []
    for param in AvailParams:
        if param in OverWriteParams:
            Paramtemp.append(param)
    
    InvalidParams = [param for param in OverWriteParams if param not in Paramtemp]
    if len(InvalidParams): print(f"Parameters `{InvalidParams}` not identified in Available Parameters") 
    Parameters = Paramtemp
print(f"Parameters sampled: {Parameters}")

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
            getLayerwisePowerLatency(op_executor,model_name,Device,sample_paths,ModeS, Parameters, EvaluatePower, EvaluateLatency)

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
