
import argparse
import copy
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

from multi_network_stats.Stage import Stage
from multi_network_stats.NetworkMappingGenerator import MappingGenerator
from layer_stats.rawDataGeneration.PowerLatencySampler import flatten_layers
from layer_stats.utils.checker import get_model
from layer_stats.utils.operations import CustomOpExecutor, database_spawn

from tegrestats.tegrastats_utils import analyze_power_stats

# Project path
project_path = Path(__file__).resolve().parents[0]

Parameters = ["RAM", "SWAP","CPU","EMC_FREQ","GR3D_FREQ","MCPU","GPU","BCPU","VDD_SYS_GPU","VDD_SYS_SOC","VDD_SYS_CPU","VDD_SYS_DDR"]

default_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)])

# Running_active = True
# For pipelined implementaion
def RunStagesSequential(stageList:list, imgQueue):
    global Running_active
    global project_path
    global tgr_interval
    global tgr_NS
    global sampling_freq
    global sampling_window
    
    infCount = 0
    # while not imgQueue.empty():
    while Running_active: 
        x = imgQueue#.get()
        for stageN in stageList:
            x = stageN.forwardSeq(x)
        infCount += 1  
    # print("Number of inferences: ", infCount)

def MappingDataExtractor(args, ModeS, Parameters, ObjStages, mapping, image,tgr_NS:int = None):
    try:
        global Running_active

        InferenceSummary = None
        power_stats = None

        threadDict= {}

        model_list = list(mapping.keys())


        print("Mapping: ", mapping)
        # print("ObjStages: ", ObjStages)

        if args.eval_thr:
            ##########################################################
                        # Throughtput capturing 
            ########################################################## 
            # threadDict= {}

            print("Throughput evaluation ...")

            for model_name in model_list:            
                threadDict[model_name] = threading.Thread(target = RunStagesSequential, args = (ObjStages[model_name],image))

            # print("threadDict: ", threadDict)
            # print("Number of stages: ", len(threadDict))

            InferenceSummary = {model_name:{} for model_name in model_list}
            
            Running_active = True
            t1 = time.time()
            for thread_name in threadDict.keys():
                threadDict[thread_name].start()

            while time.time() - t1 < args.smpl_duration:
                time.sleep(1)
            
            Running_active = False

            t2 = time.time()
            # InferenceSummary = {model_name:{} for model_name in model_list}
            for model_name in ObjStages.keys():
                for stage in ObjStages[model_name]:
                    if stage.device.type not in InferenceSummary[model_name]:
                        InferenceSummary[model_name][stage.device.type] = {}
                    InferenceSummary[model_name][stage.device.type]["infCount"] = stage.infCount
                    InferenceSummary[model_name][stage.device.type]["infDurations"] = copy.deepcopy(stage.inferDurations)
                    InferenceSummary[model_name][stage.device.type]["transferDurations"] = copy.deepcopy(stage.tranferDurations)


            for thread_name in threadDict.keys():
                threadDict[thread_name].join()
            
            t3 = time.time()

            print("Time taken to stop threads: " , t3-t2)

        if args.eval_tgr:
            #########################################################
                        # Power stats capturing 
            #########################################################

            print("Power evaluation ...")

            thread_names = list(threadDict.keys())
            for thread_name in thread_names:
                threadDict.pop(thread_name)

            # TODO: adda reset part in stage to set all the value to defaults

            for model_name in model_list:            
                threadDict[model_name] = threading.Thread(target = RunStagesSequential, args = (ObjStages[model_name],image))

            tempPath = os.path.join(project_path,"tegrastatsDataTemp.txt")

            if os.path.exists(tempPath):
                os.remove(tempPath)
            
            command = f'tegrastats --interval {args.tgr_interval} | head -n {tgr_NS} > {tempPath}'

            # # A subprocess to run the tegrastats command
            stats_proc = subprocess.Popen(command,shell=True)
            
            Running_active = True

            for thread_name in threadDict.keys():
                threadDict[thread_name].start()
            
            # let the workload begin
            time.sleep(args.tgr_smpl_boundry)

            t1 = time.time()

            while (stats_proc.poll()==None):
            # while (time.time() - t1 < args.smpl_duration):
                time.sleep(1)
            
            t2 = time.time()

            print("Execution time: " , t2-t1)
            
            Running_active = False

            for thread_name in threadDict.keys():
                threadDict[thread_name].join()
            
            power_stats = analyze_power_stats(tempPath,' ', Parameters)

            # # Count the number of samples for each power stat
            # powerStatCount = {key:len(value) for key,value in power_stats.items()}
            # print(f"Power stats data sample count: {powerStatCount}\n")

            thread_names = list(threadDict.keys())
            for thread_name in thread_names:
                threadDict.pop(thread_name)

            for model_name in model_list:
                for stage in ObjStages[model_name]:
                    stage.removeStage()

        return InferenceSummary, power_stats
    except Exception as e1:
        print("Error: ", e1)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)