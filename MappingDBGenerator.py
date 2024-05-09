import argparse
import json
import sys
import os
import time
from torchvision import transforms
import torch
import threading
from PIL import Image
import queue
from copy import deepcopy
from pathlib import Path
import subprocess
import signal
import random
import numpy as np
import gc

from multi_network_stats.Stage import Stage
from multi_network_stats.NetworkMappingGenerator import MappingGenerator
from layer_stats.utils.checker import get_model, get_model_list
from layer_stats.utils.operations import CustomOpExecutor, database_spawn
from MappingDBSampler import MappingDataExtractor

from concurrent.futures import ThreadPoolExecutor

from tegrestats.tegrastats_utils import analyze_power_stats

# Project path
project_path = Path(__file__).resolve().parents[0]

# List of default images
sample_set = ['img1.jpg']#, 'img2.jpg', 'img3.jpg']

custom_model_list = ['alexnet','mobilenet_v2','mobilenet_v3_large']
randSeed = 34

Parameters = ["RAM", "SWAP","CPU","EMC_FREQ","GR3D_FREQ","MCPU","GPU","BCPU","VDD_SYS_GPU","VDD_SYS_SOC","VDD_SYS_CPU","VDD_SYS_DDR", "VDD_IN"]
model_list = get_model_list()

default_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)])

def arguments_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, type=int, help="The mode that is being experimented. Make sure to switch the mode in Tx2.")
    parser.add_argument("--num_samples", default=20, type=int, help="the number of samples to be generated for thie selected mode.")
    parser.add_argument("--device_list", default=["cpu","cuda"], help="The devices to run the experiment on.")
    parser.add_argument("--device_priorities", default=None, help="The proirities of using the devices for mapping. Deafault will give equal priority each device.")
    parser.add_argument("--model_list", default=get_model_list(),help="The list of models to be used to generate the dataset.")
    parser.add_argument("--imgs", default=sample_set, nargs='+',help="A comma seperated list of images to run the experimets. \
                        Images should be stored in data/imagenet.")
    parser.add_argument("--smpl_duration", default=30, type=int, help="The time interval to to run a workload for the measurements.")
    
    # Tegratstats related parameters
    parser.add_argument("--tgr_params", default=["RAM", "SWAP","CPU","EMC_FREQ","GR3D_FREQ","MCPU","GPU","BCPU","VDD_SYS_GPU","VDD_SYS_SOC","VDD_SYS_CPU","VDD_SYS_DDR", "VDD_IN"],\
                        nargs='+' ,help="A space seperatedlist of parameters from the defaults list of parameters to extract from tegratstats.")
    parser.add_argument("--eval_tgr", action='store_true' ,help="Whether to evaluate tegrastats parameters. Intended for power evaluation.")
    parser.add_argument("--tgr_interval", default=50,type=int, help="The tegrastats data sampling interval. Data sampled at every `tgr_interval` mS time.")
    # parser.add_argument("--smpl_duration", default=30, type=int, help="The time duration to run sample tegrastats data for a layer.")
    parser.add_argument("--tgr_smpl_boundry", default=2, type=int, help="Time window allowed for the tegratstats to stablize for the layer being experimented")

    # throughput related parameters
    parser.add_argument("--eval_thr", action='store_true', help="Whether to evaluate throughput.")
    # parser.add_argument("--thr_interval", default=30, type=int, help="The time interval to to run a workload for the throughput measurement.")

    return parser.parse_args()

def InitializeParams(args):

    print(f"SAMPLING DURATION: {args.smpl_duration}")
    if args.eval_tgr:
        # print(f"Tegrastats configuration: \nSAMPLING INTERVAL: {args.tgr_interval}\nSAMPLING DURATION: {args.smpl_duration}\
        #     \nSAMPLING BOUNDRY: {args.tgr_smpl_boundry}")
        print(f"Tegrastats configuration: \nSAMPLING INTERVAL: {args.tgr_interval}\nSAMPLING BOUNDRY: {args.tgr_smpl_boundry}")
    # if args.eval_thr:
    #     print(f"SAMPLING COUNT: {args.tgr_smpl_boundry}")

    # Define the mode to run the experiment
    # Make sure the mode specified is alredy set in Tx2
    MODE = args.mode
    ModeS = f"Mode{MODE}"

    ######## Other Parameters ######### 
    Parameters = []

    if args.eval_tgr:

        # Available Parameters to be sampled from tegrastats. Listed in the order params appear in the command output.
        AvailParams = ["RAM", "SWAP","CPU","EMC_FREQ","GR3D_FREQ","MCPU","GPU","BCPU","VDD_SYS_GPU","VDD_SYS_SOC","VDD_SYS_CPU","VDD_SYS_DDR", "VDD_IN"]

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

def WorkloadDataGenerator(args, ModeS, Parameters):
    try:

        trailRun = True

        sampling_freq = 1000.0/args.tgr_interval
        tgr_NS = int((1000/args.tgr_interval)*args.smpl_duration) 

        model_list = args.model_list
        random.shuffle(model_list)

        NumModelsCases = range(5,11)

        NumCases = len(NumModelsCases)

        ModelsPerCases = [int(args.num_samples/NumCases)]*NumCases

        for i in range(args.num_samples%NumCases):
            ModelsPerCases[-(i+1)] += 1
        print(f"ModelsPerCase: {ModelsPerCases}")

        GeneratedModelsPerCases = [0]*NumCases

        caseIdx = 0

        image = Image.open("data/imagenet/img1.jpg")
        image = default_preprocess(image)
        image = image.unsqueeze(0)

        numProcessedMappings = 0

        # while GeneratedModelsPerCases[caseIdx] < ModelsPerCases[caseIdx]:
        for caseIdx in range(NumCases):

            print(f"running case {caseIdx}")

            # Number of MappingeGenerator objects created for the current case
            NumObjs = 0

            while GeneratedModelsPerCases[caseIdx] < ModelsPerCases[caseIdx]:

                # Number of mappings to generate from each object
                SmplPerObj = 20 #int(min(20,ModelsPerCases[caseIdx], ModelsPerCases[caseIdx] - 20*NumObjs))

                # Randomly choose n number of  models accroding to the case
                Model_set = np.random.choice(model_list,NumModelsCases[caseIdx], replace = False)

                if trailRun:
                    MapperObj = MappingGenerator(args.device_list,Model_set,NumSamples=SmplPerObj,device_prioriy = args.device_priorities, seed=randSeed)
                    ObjStages,mapping = MapperObj.iter()
                    InferenceSummary, power_stats = MappingDataExtractor(args,ModeS,Parameters, ObjStages, mapping,image,tgr_NS)
                    trailRun = False

                # TODO: Determine the number of parallel network cases (eg: range(4,11)). Randomly sample n number of network. --> Done
                MapperObj = MappingGenerator(args.device_list,Model_set,NumSamples=SmplPerObj,device_prioriy = args.device_priorities, seed=randSeed)

                ObjStages,mapping = MapperObj.iter()

                while ObjStages != False:
                    # ObjStages = []
                    # mapping = []
                    # ObjStages,mapping = MapperObj.iter()

                    print(f"Mapping {numProcessedMappings}")

                    if args.eval_tgr:
                        InferenceSummary, power_stats = MappingDataExtractor(args,ModeS,Parameters, ObjStages, mapping,image,tgr_NS)
                    else:
                        InferenceSummary, power_stats = MappingDataExtractor(args,ModeS,Parameters, ObjStages, mapping,image)
                    
                    # if not trailRun:
                    numProcessedMappings += 1

                    # Path to store the mapping data samples
                    if not os.path.exists(os.path.join(project_path, 'Dataset/Multinet', ModeS)):
                        os.makedirs(os.path.join(project_path, 'Dataset/Multinet', ModeS))

                    # File name to store the mapping data samples
                    # stats_file_name = f"{os.path.join(project_path, 'Dataset/Multinet', ModeS)}/Map{numProcessedMappings}_sf_{sampling_freq}_sb_{args.tgr_smpl_boundry}.json"
                    stats_file_name = f"{os.path.join(project_path, 'Dataset/Multinet', ModeS)}/Map{numProcessedMappings}_nm_{NumModelsCases[caseIdx]}_sf_{sampling_freq}.json"
                    
                    # Convert the mapping object keys to strings 
                    mappingS = {}
                    for net, mapN in mapping.items():
                        mappingS[net] = {str(ID):dev for ID,dev in mapping[net].items()}
                    outJson = {"mapping":mappingS,"stageSummary": InferenceSummary, "power":power_stats,"samplingDuration":args.smpl_duration}
                    
                    # if not trailRun:
                    # Write data to a json file
                    with open(stats_file_name, "w") as stats_file:
                        json.dump(outJson,stats_file)

                    # Inceremenet the number of samples counter of the current case 
                    GeneratedModelsPerCases[caseIdx] += 1

                    ObjStages = []
                    mapping = []

                    # Remove the object stages from  memory
                    del ObjStages
                    gc.collect()

                    ObjStages,mapping = MapperObj.iter()

                    print("\n")
                
                # NumObjs += 1

            # # Change the case index if defined number of sameple are generated for the currenet case
            # if int(GeneratedModelsPerCases[caseIdx]) == int(ModelsPerCases[caseIdx]):
            #     if caseIdx == NumCases - 1:
            #         print("Dataset generation completed!")
            #         break 
            #     else:
            #         caseIdx += 1
        print("Dataset generation completed!")

    except Exception as e1:
        print("Error: ", e1)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == "__main__":
    try:
        t1 = time.time()
        args = arguments_parser()
        if not (args.eval_tgr or args.eval_thr):
            print("Evaluation parameters not enabled")
        else:
            ModeS, Parameters = InitializeParams(args)
            if len(Parameters)>0:
                WorkloadDataGenerator(args,ModeS, Parameters)
        t2 = time.time()
        print(f"Time taken for Mode{args.mode}: {t2-t1} seconds")
    except Exception as e:
        print("Error: ", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

