
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

# List of default images
# sample_set = ['img1.jpg']#, 'img2.jpg', 'img3.jpg']

# Mode = 7
# ModeS = f"Mode{str(Mode)}"
# numSamples= 3
# model_list = ['alexnet','mobilenet_v2','mobilenet_v3_large']
# model_list = ['alexnet','mobilenet_v2','mobilenet_v3_large',"regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_x_3_2gf", "regnet_x_800mf", "regnet_x_400mf"]

# standardRunDevice = 'cpu'
# deviceList = ['cpu','cpu']
# devicePriorities = [0.5,0.5]

# randSeed = 34
# customMapping = None
# sampling_window = 2

# ## Tegrastats parameters
# tgr_interval = 20                                           # tegrastats sampling interval in milliseconds
# tgr_NS = int((1000/tgr_interval)*sampling_window)           # Number of sample to extract from tegrastats
# sampling_freq = int(1/tgr_interval)                         # number of sample per second

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


            for thread_name in threadDict.keys():
                threadDict[thread_name].join()
            
            t3 = time.time()

            print("time taken to stop: " , t3-t2)

            # for model_name in ObjStages.keys():
            #     print(f"Inf count of `{model_name}`: {ObjStages[model_name][-1].infCount}")
            #     print(f"Throughput of `{model_name}`: {ObjStages[model_name][-1].infCount/(t2-t1)}")
            
            print("Inference counts: ", InferenceSummary)


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
            # startCommand = f'exec tegrastats --interval {tgr_interval} > {tempPath}'
            # startCommand = f'tegrastats --interval {tgr_interval} --logfile {tempPath} --start'
            # stopCommand = f'tegrastats --stop'
            # stopCommand = "kill -9 $(ps -ax | grep Mapping | fgrep -v grep | awk '{ print $1 }'"

            # # A subprocess to run the tegrastats command
            stats_proc = subprocess.Popen(command,shell=True)
            
            Running_active = True

            for thread_name in threadDict.keys():
                threadDict[thread_name].start()
            
            # let the workload begin
            time.sleep(args.tgr_smpl_boundry)

            # A subprocess to run the tegrastats command
            # stats_proc_start = subprocess.Popen(startCommand,shell=False)#,preexec_fn=os.setsid)
            # os.system(startCommand)
            # os.wait()

            # time.sleep(2)

            t1 = time.time()

            while (stats_proc.poll()==None):
            # while (time.time() - t1 < args.smpl_duration):
                time.sleep(1)
            
            t2 = time.time()

            print("excution time: " , t2-t1)

            # Stop running tegrastats process
            # stats_proc_end = subprocess.Popen(stopCommand)
            # stats_proc_start.kill()
            # os.killpg(os.getpgid(stats_proc_start), signal.SIGTERM)
            # os.system(stopCommand)
            # os.wait()
            
            Running_active = False

            for thread_name in threadDict.keys():
                threadDict[thread_name].join()
            
            power_stats = analyze_power_stats(tempPath,' ', Parameters)

            powerStatCount = {key:len(value) for key,value in power_stats.items()}
            print("power stats count: ",powerStatCount)

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