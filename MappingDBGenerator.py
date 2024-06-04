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
# randSeed = 34

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
    parser.add_argument("--jetsonDevice", default="Tx2",choices=["Orin","Tx2","Xavier"], help="The Jetson device to run the experiment on.")
    parser.add_argument("--mode", required=True, type=int, help="The mode that is being experimented. Make sure to switch the mode in Tx2.")
    parser.add_argument("--num_samples", default=20, type=int, help="the number of samples to be generated for thie selected mode.")
    parser.add_argument("--device_list", default=["cpu","cuda"], help="The devices to run the experiment on.")
    parser.add_argument("--device_priorities", default=None, help="The proirities of using the devices for mapping. Default will give equal priority each device.")
    parser.add_argument("--model_list", default=get_model_list(),help="The list of models to be used to generate the dataset.")
    parser.add_argument("--single_nets", action='store_true', help="Whether to generate single network mappings.")
    parser.add_argument("--gpu_only_maps", action='store_true', help="Whether to generate mappings with only GPU devices.")
    parser.add_argument("--numNetsRange", default=[5,10], nargs=2, type=int, help="The range of number of networks to be used in the mapping.")
    parser.add_argument("--samples_per_set", default=20, type=int, help="The number of samples to be generated for each set of networks.")
    parser.add_argument("--imgs", default=sample_set, nargs='+',help="A comma seperated list of images to run the experimets. \
                        Images should be stored in data/imagenet.")
    parser.add_argument("--smpl_duration", default=30, type=int, help="The time interval to to run a workload for the measurements.")
    
    # Tegratstats related parameters
    parser.add_argument("--tgr_params", default=["RAM", "SWAP","CPU","EMC_FREQ","GR3D_FREQ","APE","PLL","MCPU","PMIC","Tboard","GPU","BCPU","thermal","Tdiode","VDD_SYS_GPU","VDD_SYS_SOC","VDD_4V0_WIFI","VDD_IN","VDD_SYS_CPU","VDD_SYS_DDR"],\
                        nargs='+' ,help="A space seperatedlist of parameters from the defaults list of parameters to extract from tegratstats.")
    parser.add_argument("--eval_tgr", action='store_true' ,help="Whether to evaluate tegrastats parameters. Intended for power evaluation.")
    parser.add_argument("--tgr_interval", default=50,type=int, help="The tegrastats data sampling interval. Data sampled at every `tgr_interval` mS time.")
    # parser.add_argument("--smpl_duration", default=30, type=int, help="The time duration to run sample tegrastats data for a layer.")
    parser.add_argument("--tgr_smpl_boundry", default=2, type=int, help="Time window allowed for the tegratstats to stablize for the layer being experimented")

    # throughput related parameters
    parser.add_argument("--eval_thr", action='store_true', help="Whether to evaluate throughput.")
    # parser.add_argument("--thr_interval", default=30, type=int, help="The time interval to to run a workload for the throughput measurement.")

    parser.add_argument("--max_split_nets", default=-1, type=int, help="The maximum number of networks to be split in a mapping.")
    parser.add_argument("--seed", default=0, type=int, help="The seed to be used for random number generation.")

    return parser.parse_args()

def InitializeParams(args):
    print("The follwoing parameters are initialized for the experiment:")
    print(f"MODE: {args.mode}")
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
        AvailParams = ["RAM", "SWAP","CPU","EMC_FREQ","GR3D_FREQ","APE","PLL","MCPU","PMIC","Tboard","GPU","BCPU","thermal","Tdiode","VDD_SYS_GPU","VDD_SYS_SOC","VDD_4V0_WIFI","VDD_IN","VDD_SYS_CPU","VDD_SYS_DDR"]

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
        print(f"Parameters sampled: {Parameters}\n")

    return ModeS, Parameters

def WorkloadDataGenerator(args, ModeS, Parameters):
    try:
        # Set the seed for random number generation
        if args.seed > 0:
            randSeed = args.seed
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
        else:
            randSeed = args.mode
            random.seed(args.mode)
            np.random.seed(args.mode)
            torch.manual_seed(args.mode)

        trailRun = True

        if args.device_priorities is not None:
            args.device_priorities = [float(dev) for dev in args.device_priorities.split(",")]

        sampling_freq = 1000.0/args.tgr_interval
        tgr_NS = int((1000/args.tgr_interval)*args.smpl_duration) 
        print(f"Tegrastats Sampling Frequency: {sampling_freq} Hz")
        print(f"Number of tegrastat samples to be captured for each mapping: {tgr_NS}")

        if type(args.model_list) == str:
            model_list = args.model_list.split(",")
        elif type(args.model_list) == list:
            model_list = args.model_list
        else:
            print("Invalid model list")
            return
        random.shuffle(model_list)

        NumModelsCases = list(range(args.numNetsRange[0],args.numNetsRange[1]+1))

        NumCases = len(NumModelsCases)

        MappingsperCases = [int(args.num_samples/NumCases)]*NumCases

        for i in range(args.num_samples%NumCases):
            MappingsperCases[-(i+1)] += 1

        print("The following two lists provide the number of models and mappings to be generated for each case.")
        print(f"Number of Models for each case: {NumModelsCases}")
        print(f"Number of Mappings for each case: {MappingsperCases}\n")

        GeneratedModelsPerCases = [0]*NumCases

        caseIdx = 0

        image = Image.open("data/imagenet/img1.jpg")
        image = default_preprocess(image)
        image = image.unsqueeze(0)

        numProcessedMappings = 0

        # Generate single network mappings
        if args.single_nets:

            if trailRun:
                # Randomly choose n number of  models accroding to the case
                Model_set = np.random.choice(model_list,min(len(model_list),5), replace = False)
                MapperObj = MappingGenerator(args.jetsonDevice,args.device_list,model_list,NumSamples=1,device_prioriy = args.device_priorities, seed=randSeed)
                ObjStages,mapping = MapperObj.iter()
                InferenceSummary, power_stats = MappingDataExtractor(args,ModeS,Parameters, ObjStages, mapping,image,tgr_NS)
                trailRun = False

            for model_name in model_list:
                
                # Number of sets of networks created for the current case
                NumObjs = 1

                for device in args.device_list:

                    # Mappier object with one network and one device(GPU)
                    MapperObj = MappingGenerator(args.jetsonDevice,[device],[model_name],NumSamples=1,device_prioriy = [1], seed=randSeed)

                    ObjStages,mapping = MapperObj.iter()

                    if args.eval_tgr:
                        InferenceSummary, power_stats = MappingDataExtractor(args,ModeS,Parameters, ObjStages, mapping,image,tgr_NS)
                    else:
                        InferenceSummary, power_stats = MappingDataExtractor(args,ModeS,Parameters, ObjStages, mapping,image)
                    numProcessedMappings += 1

                    # Path to store the mapping data samples
                    if not os.path.exists(os.path.join(project_path, 'Dataset/Multinet', ModeS)):
                        os.makedirs(os.path.join(project_path, 'Dataset/Multinet', ModeS))

                    # File name to store the mapping data samples
                    stats_file_name = f"{os.path.join(project_path, 'Dataset/Multinet', ModeS)}/SingleNet_Map{numProcessedMappings}_nm_{1}_set_{NumObjs}_classA.json"

                    # Convert the mapping object keys to strings 
                    mappingS = {}
                    for net, mapN in mapping.items():
                        mappingS[net] = {str(ID):dev for ID,dev in mapping[net].items()}
                    outJson = {"mapping":mappingS,"stageSummary": InferenceSummary, "power":power_stats,"samplingDuration":args.smpl_duration}

                    # Write data to a json file
                    with open(stats_file_name, "w") as stats_file:
                        json.dump(outJson,stats_file)
                    
                    ObjStages = []
                    mapping = []

                    # Remove the object stages from  memory
                    del ObjStages
                    gc.collect()
                NumObjs += 1

            numProcessedMappings = 0

        # while GeneratedModelsPerCases[caseIdx] < MappingsperCases[caseIdx]:
        for caseIdx in range(NumCases):

            print(f"New case with number of models in each mapping = {NumModelsCases[caseIdx]}\n")

            # Number of sets of networks created for the current case
            NumObjs = 1

            while GeneratedModelsPerCases[caseIdx] < MappingsperCases[caseIdx]:

                # Number of mappings to generate from each object
                SmplPerObj = args.samples_per_set #int(min(20,MappingsperCases[caseIdx], MappingsperCases[caseIdx] - 20*NumObjs))

                # Indicate if the split mapping has been started
                splitMapperStarted = False

                # Randomly choose n number of  models accroding to the case
                Model_set = np.random.choice(model_list,NumModelsCases[caseIdx], replace = False)

                # Set of models to be used for the mapper object
                print("\nModel set: ", Model_set)

                # Warm up the system with a trail run 
                if trailRun:
                    MapperObj = MappingGenerator(args.jetsonDevice,args.device_list,Model_set,NumSamples=SmplPerObj,device_prioriy = args.device_priorities, seed=randSeed)
                    ObjStages,mapping = MapperObj.iter()
                    InferenceSummary, power_stats = MappingDataExtractor(args,ModeS,Parameters, ObjStages, mapping,image,tgr_NS)
                    trailRun = False

                CaseList = None
                if args.max_split_nets >= 0:
                    CaseList = [0]*(NumModelsCases[caseIdx]+1)
                    if args.gpu_only_maps:
                        CaseList[NumModelsCases[caseIdx] - args.max_split_nets:] = [int(np.floor(((SmplPerObj-1)/(args.max_split_nets+1))))]*(args.max_split_nets+1)
                        # Assign the remaining samples to the last case
                        CaseList[-1] += (SmplPerObj-1)%(args.max_split_nets+1)
                    else:
                        CaseList[NumModelsCases[caseIdx] - args.max_split_nets:] = [int(SmplPerObj/(args.max_split_nets+1))]*(args.max_split_nets+1)
                        # Assign the remaining samples to the last case
                        CaseList[-1] += SmplPerObj%(args.max_split_nets+1)

                    # # Assign the remaining samples to the last few cases
                    # for i in range(SmplPerObj%(args.max_split_nets+1)):
                    #     CaseList[-(i+1)] += 1

                    print(f"CaseList: {CaseList}")

                # Map all networks in the model set to GPU
                if args.gpu_only_maps:
                    MapperObj = MappingGenerator(args.jetsonDevice,["cuda"],Model_set,NumSamples=[1],CaseSamples=None,device_prioriy = args.device_priorities, seed=randSeed)
                else:
                    MapperObj = MappingGenerator(args.jetsonDevice,args.device_list,Model_set,NumSamples=SmplPerObj,CaseSamples=CaseList,device_prioriy = args.device_priorities, seed=randSeed)
                    splitMapperStarted = True
                
                # Incremented upon mappings fail due to cuda out of memory error
                retryCount = 0

                print("\n")

                # Store the current case mapping counts
                OldMapperObjCaseCounts = deepcopy(MapperObj.mapCaseCounts)

                # enumerate mapper among the defined number of samples
                ObjStages,mapping = MapperObj.iter()

                while ObjStages != False:

                    while ObjStages == True and mapping == False:
                        retryCount += 1
                        ObjStages,mapping = MapperObj.iter()
                        if retryCount > 10:
                            print("Retries exceeded! Failed to generate mapping due to cuda out of memory.")
                            break
                    
                    if ObjStages == False:
                        break
                    
                    # reset the retry count
                    retryCount = 0

                    print(f"Mapping ID {numProcessedMappings}")

                    if args.eval_tgr:
                        InferenceSummary, power_stats = MappingDataExtractor(args,ModeS,Parameters, ObjStages, mapping,image,tgr_NS)
                    else:
                        InferenceSummary, power_stats = MappingDataExtractor(args,ModeS,Parameters, ObjStages, mapping,image)
                    
                    numProcessedMappings += 1

                    # Path to store the mapping data samples
                    if not os.path.exists(os.path.join(project_path, 'Dataset/Multinet', ModeS)):
                        os.makedirs(os.path.join(project_path, 'Dataset/Multinet', ModeS))
                    
                    # Check the difference in the number of mappings generated for the current case
                    Diff = [i-j for i,j in zip(MapperObj.mapCaseCounts,OldMapperObjCaseCounts)]

                    print(f"Diff: {Diff}")

                    # File name to store the mapping data samples
                    if splitMapperStarted == False:
                        stats_file_name = f"{os.path.join(project_path, 'Dataset/Multinet', ModeS)}/MultiNet_Map{numProcessedMappings}_nm_{NumModelsCases[caseIdx]}_set_{NumObjs}_classB.json"
                    elif 1 in Diff and Diff.index(1) == len(OldMapperObjCaseCounts)-1:
                        stats_file_name = f"{os.path.join(project_path, 'Dataset/Multinet', ModeS)}/MultiNet_Map{numProcessedMappings}_nm_{NumModelsCases[caseIdx]}_set_{NumObjs}_classC.json"
                    else:
                        stats_file_name = f"{os.path.join(project_path, 'Dataset/Multinet', ModeS)}/MultiNet_Map{numProcessedMappings}_nm_{NumModelsCases[caseIdx]}_set_{NumObjs}_classD.json"
                    
                    # Convert the mapping object keys to strings 
                    mappingS = {}
                    for net, mapN in mapping.items():
                        mappingS[net] = {str(ID):dev for ID,dev in mapping[net].items()}
                    outJson = {"mapping":mappingS,"stageSummary": InferenceSummary, "power":power_stats,"samplingDuration":args.smpl_duration}
                    
                    # Write data to a json file
                    with open(stats_file_name, "w") as stats_file:
                        json.dump(outJson,stats_file)

                    # Inceremenet the number of samples counter of the current case 
                    GeneratedModelsPerCases[caseIdx] += 1

                    ObjStages = []
                    mapping = []

                    # Store the current case mapping counts
                    OldMapperObjCaseCounts = deepcopy(MapperObj.mapCaseCounts)

                    # Remove the object stages from  memory
                    del ObjStages
                    gc.collect()

                    # when gpu_only_maps is enabled, generate one such mapping for each model set and run remaining mappings with all devices
                    if not splitMapperStarted:
                        MapperObj = MappingGenerator(args.jetsonDevice,args.device_list,Model_set,NumSamples=SmplPerObj-1,CaseSamples=CaseList,device_prioriy = args.device_priorities, seed=randSeed)
                        splitMapperStarted = True
                    
                    print("\n")
                    ObjStages,mapping = MapperObj.iter()

                NumObjs += 1

        print("\nDataset generation completed!")

    except Exception as e1:
        print("\nError: ", e1)
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
        print(f"\nTime taken for Mode{args.mode}: {t2-t1} seconds")
    except Exception as e:
        print("\nError: ", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

