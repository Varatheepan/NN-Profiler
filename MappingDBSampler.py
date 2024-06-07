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
import logging
try:
    from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
    DeepSPeed = True
except Exception as e:
    DeepSPeed = False
    pass
from thop import profile
from thop import clever_format

from multi_network_stats.Stage import Stage
from multi_network_stats.NetworkMappingGenerator import MappingGenerator
from layer_stats.rawDataGeneration.PowerLatencySampler import flatten_layers
from layer_stats.utils.checker import get_model, get_num_layers
from layer_stats.utils.operations import CustomOpExecutor, database_spawn

from tegrestats.tegrastats_utils import analyze_power_stats

# Project path
project_path = Path(__file__).resolve().parents[0]

default_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)])

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

def MappingDataExtractor(args, ModeS, Parameters, ObjStages, mapping, image,tgr_NS:int = None):
    try:
        global Running_active

        # get logger
        logger = logging.getLogger(__name__)

        if DeepSPeed == False:
            logger.warning("DeepSpeed cannot be loaded.")

        InferenceSummary = None
        power_stats = None

        threadDict= {}

        model_list = list(mapping.keys())


        logger.info(f"Mapping: , {mapping}")

        if args.eval_thr:
            ##########################################################
                        # Throughtput capturing 
            ########################################################## 

            logger.info("Throughput evaluation ...")

            for model_name in model_list:            
                threadDict[model_name] = threading.Thread(target = RunStagesSequential, args = (ObjStages[model_name],image))

            InferenceSummary = {model_name:{} for model_name in model_list}
            
            Running_active = True
            t1 = time.time()
            for thread_name in threadDict.keys():
                threadDict[thread_name].start()

            while time.time() - t1 < args.smpl_duration:
                time.sleep(1)
            
            Running_active = False

            t2 = time.time()
            
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

            logger.info(f"Time taken to stop threads: {t3-t2}")

        if args.eval_tgr:
            #########################################################
                        # Power stats capturing 
            #########################################################

            logger.info("Power evaluation ...")

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

            logger.info(f"Execution time: {t2-t1}")
            
            Running_active = False

            for thread_name in threadDict.keys():
                threadDict[thread_name].join()
            
            power_stats = analyze_power_stats(tempPath,' ', Parameters, args.jetsonDevice)

            thread_names = list(threadDict.keys())
            for thread_name in thread_names:
                threadDict.pop(thread_name)
        
        #########################################################
        # Profile the stage to get the flops, macs, and params
        #########################################################
        if InferenceSummary is None:
            InferenceSummary = {model_name:{} for model_name in model_list}


        for model_name in model_list:
            tempStage = None
            for stageID,stage in enumerate(ObjStages[model_name]):
                if stage.device.type not in InferenceSummary[model_name]:
                        InferenceSummary[model_name][stage.device.type] = {}

                flops, macs, params = None, None, None
                if stage.device.type == "cuda":
                    tempStage = stage.layerSet.to(torch.device("cpu"))
                else: 
                    tempStage = stage.layerSet
                # logger.info(f"tempStage assigned to {next(tempStage.parameters()).device}")
                
                if DeepSPeed:
                    try:
                        # DeepSpeed
                        prof = FlopsProfiler(tempStage)
                        prof.start_profile()
                        x = torch.randn(tuple(stage.inputSize))
                        # tempStage.to("cpu")
                        tempStage(x)
                        flops = prof.get_total_flops(as_string=True)
                        macs = prof.get_total_macs(as_string=True)
                        params = prof.get_total_params(as_string=True)
                        prof.end_profile()
                    except Exception as e:
                        logger.error(f"Profiling error in DeepSpeed: {e}")
                        flops, macs, params = None, None, None

                if params is None:
                    try:
                        # use thops
                        macs, params = profile(tempStage, inputs=(torch.randn(tuple(stage.inputSize)), ), verbose=False)
                        macs, params = clever_format([macs, params], "%.3f")
                        # logger.info(f"Mac count: {macs}, Param count: {params}", model_name, stage.device.type)
                    except Exception as e:
                        logger.error(f"Profiling error in Thops: {e}")
                        macs, params = None, None

                if DeepSPeed and (flops is None) and (not macs is None):
                    logger.warning("DeepSpeed profiling failed. Using thop for profiling.")
                              
                InferenceSummary[model_name][stage.device.type]["macs"] = macs
                InferenceSummary[model_name][stage.device.type]["params"] = params 
                InferenceSummary[model_name][stage.device.type]["flops"] = flops
                if args.funcLayerCount:
                    InferenceSummary[model_name][stage.device.type]["functionalLayerCount"] = get_num_layers(tempStage)
                    
                # Remove the stage layerSet to avoid memory issues
                ret = stage.removeStage()
                if not ret:
                    logger.warning(f"Error while removing {stageID} from {model_name}")
        

        return InferenceSummary, power_stats
    except Exception as e1:
        logger.error(f"Error: {e1}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logger.error(f"{exc_type}, {fname}, {exc_tb.tb_lineno}")
        return {},{}