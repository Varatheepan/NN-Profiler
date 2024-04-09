from PowerLatencySampler import getLayerwisePowerLatency
from layer_stats.utils.operations import database_spawn
from layer_stats.utils.checker import get_model_list
import os
import time
import argparse
from pathlib import Path
import sys

'''
To change the number of data samples collected per image, change the parameters defined in `PowerLatencySampler.py` script.
'''

# Project path
project_path = Path(__file__).resolve().parents[2]

# List of default images
sample_set = ['img1.jpg', 'img2.jpg', 'img3.jpg']


def arguments_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, type=int, help="The mode that is being experimented. Make sure to switch the mode in Tx2.")
    parser.add_argument("--device", default="cpu", choices=["cpu","cuda"], help="The device to run the experiment on.")
    parser.add_argument("--imgs", default=sample_set, nargs='+',help="A comma seperated list of images to run the experimets. \
                        Images should be stored in data/imagenet.")
    
    # Tegratstats related parameters
    parser.add_argument("--tgr_params", default=["RAM", "SWAP","CPU","EMC_FREQ","GR3D_FREQ","MCPU","GPU","BCPU","VDD_SYS_GPU","VDD_SYS_SOC","VDD_SYS_CPU","VDD_SYS_DDR"],\
                        nargs='+' ,help="A space seperatedlist of parameters from the defaults list of parameters to extract from tegratstats.")
    parser.add_argument("--eval_tgr", action='store_true' ,help="Whether to evaluate tegrastats parameters. Intended for power evaluation.")
    parser.add_argument("--tgr_interval", default=20,type=int, help="The tegrastats data sampling interval. Data sampled at every `tgr_interval` mS time.")
    parser.add_argument("--tgr_smpl_duration", default=5, type=int, help="The time duration to run sample tegrastats data for a layer.")
    parser.add_argument("--tgr_smpl_boundry", default=1, type=int, help="Time window allowed for the tegratstats to stablize for the layer being experimented")

    # Latency related parameters
    parser.add_argument("--eval_latency", action='store_true', help="Whether to evaluate latency.")
    parser.add_argument("--lt_smpl_count", default=100, type=int, help="The number of times to run a layer for the latency experiment.")

    return parser.parse_args()

def InitializeParams(args):

    if args.eval_tgr:
        print(f"Tegrastats configuration: \nSAMPLING INTERVAL: {args.tgr_interval}\nSAMPLING DURATION: {args.tgr_smpl_duration}\
            \nSAMPLING BOUNDRY: {args.tgr_smpl_boundry}")
    if args.eval_latency:
        print(f"SAMPLING COUNT: {args.tgr_smpl_boundry}")

    # Define the mode to run the experiment
    # Make sure the mode specified is alredy set in Tx2
    MODE = args.mode
    ModeS = f"Mode{MODE}"

    ######## Other Parameters ######### 
    Parameters = []

    if args.eval_tgr:

        # Available Parameters to be sampled from tegrastats. Listed in the order params appear in the command output.
        AvailParams = ["RAM", "SWAP","CPU","EMC_FREQ","GR3D_FREQ","MCPU","GPU","BCPU","VDD_SYS_GPU","VDD_SYS_SOC","VDD_SYS_CPU","VDD_SYS_DDR"]

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

        # Overwrite the tegarstasts parameters
        # Define a custom set of parameters to be extracted from tegerastats. Defaults to current device power if left empty. 
        OverWriteParams = args.tgr_params
        if len(OverWriteParams):
            Paramtemp = []
            for param in AvailParams:
                if param in OverWriteParams:
                    Paramtemp.append(param)
            
            InvalidParams = [param for param in OverWriteParams if param not in Paramtemp]
            if len(InvalidParams):
                print(f"Parameters `{InvalidParams}` not identified in Available Parameters")
                return ModeS, Parameters
            Parameters = Paramtemp
        print(f"Parameters sampled: {Parameters}")

    return ModeS, Parameters

def modeInfoExtractor(args, ModeS, Parameters):

    # Assigin device name for data storing
    if "cuda" in args.device:
        DeviceS = "gpu"
        if len(Parameters) == 0: Parameters = ["VDD_SYS_GPU"]
    elif args.device == "cpu":
        DeviceS = "cpu"
        if len(Parameters) == 0: Parameters = ["VDD_SYS_CPU",]

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
                if lineMode == args.mode:
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
            getLayerwisePowerLatency(args,op_executor,model_name,ModeS, Parameters)

    else:
        print("Defined mode not found")

if __name__ == "__main__":
    try:
        t1 = time.time()
        args = arguments_parser()
        if not (args.eval_tgr or args.eval_latency):
            print("Evaluation parameters not enabled")
        else:
            ModeS, Parameters = InitializeParams(args)
            if len(Parameters)>0:
                modeInfoExtractor(args,ModeS, Parameters)
        t2 = time.time()
        print(f"Time taken for Mode{args.mode}: {t2-t1} seconds")
    except Exception as e:
        print("Error: ", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
