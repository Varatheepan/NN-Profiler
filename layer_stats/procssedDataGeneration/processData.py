import argparse
import os
from pathlib import Path
import csv
import sys
import numpy as np
import json


# TODO: store standard deviation with all the avearage parameters

# Project path
project_path = Path(__file__).resolve().parents[2]

AvailModes = os.listdir(os.path.join(project_path,"Dataset"))
AvailModes = [int(mode.split("Mode")[1]) for mode in AvailModes]

def arg_parser():
    parser = argparse.ArgumentParser(
        description= "This takes `MODE` as the argument to process")
    parser.add_argument("--modes", default=AvailModes,type=list, help="list of mode numbers to process")
    parser.add_argument("--params", default=[])
    return parser.parse_args()

def processModeStats(args):

    try:
        # Object to store all processed data for each mode
        ModeWiseStats = {}

        # Networks available across each mode
        ModeWiseNetworks = {}

        # get the modes to post process
        modes = args.modes

        # Falg to overwrite tegrastats parameters to process
        isCustomStatList = bool(len(args.params)) 

        # Process raw data for each mode
        for modeID in modes:
            
            # Object to store all processed current mode data
            ModeStats = {}

            ModeWiseNetworks[modeID] = []

            # PAth of the selected mode
            ModeRoot = os.path.join(project_path,"Dataset","Mode"+str(modeID))
            
            # Devices available in the mode
            Devices = os.listdir(ModeRoot)
            Devices = [element for element in Devices if os.path.isdir(os.path.join(ModeRoot,element))]
            
            
            for Device in Devices:
                
                # Object to store processed devise-wise data for the mode
                DeviceStats = {}

                # Paths of devices available in the mode
                DevicePath = os.path.join(ModeRoot,Device)
                
                # Available Parameter: Power & Latency
                Parameters = os.listdir(DevicePath)
                
                for Parameter in Parameters:

                    ParameterStats = {}
                    # Path of the parameter
                    ParamPath = os.path.join(DevicePath,Parameter)
                
                    # Available networks
                    Nnets = os.listdir(ParamPath)

                    for net in Nnets:
                        
                        # Networks available in the mode
                        if net not in ModeWiseNetworks[modeID]:
                            ModeWiseNetworks[modeID].append(net)

                        # Object to store average values for the mode-device-network
                        NetStats = {}

                        # Network path
                        NetPath = os.path.join(ParamPath,net)

                        # Available image samples
                        imgSamples = os.listdir(NetPath)

                        # Object to load values from all images for the mode-device-network
                        Stats = {}

                        if Parameter == "latency":

                            # Combine layer-wise data for all images
                            for imgSample in imgSamples:
                                StatsFile = open(os.path.join(NetPath,imgSample), "r")

                                # Read the file line by line
                                while True:
                                    # A line represents a layer data
                                    line = StatsFile.readline().strip()

                                    if not line:
                                        break
                                    line = json.loads(line)
                                    for layerIdx, stat in line.items():
                                        if layerIdx not in Stats:
                                            Stats[layerIdx] = []
                                        Stats[layerIdx].extend(stat)

                            # Calculate average values for all the layers
                            for layerIdx, stat in Stats.items():
                                NetStats[layerIdx] = float(sum(stat))/len(stat)
                        
                        # TODO: Implementation for other tegratstats parameters
                                
                        # Process power parameters
                        elif Parameter == "power":
                            # Combine layer-wise data for all images
                            for imgSample in imgSamples:
                                StatsFile = open(os.path.join(NetPath,imgSample), "r")

                                # Read the file line by line
                                while True:
                                    # A line represents a layer data
                                    line = StatsFile.readline().strip()

                                    if not line:
                                        break
                                    line = json.loads(line)
                                    for layerIdx, statDict in line.items():
                                        for statName, stat in statDict.items():
                                            
                                            # TODO: process a cutom stats list
                                            # if isCustomStatList:
                                            # else: 
                                            #If parameter list is not defined process corresponeding device power data

                                            # process CPU power data
                                            if Device == "cpu":
                                                if statName == "VDD_SYS_CPU":
                                                    if layerIdx not in Stats:
                                                        Stats[layerIdx] = {}
                                                    if "VDD_SYS_CPU" not in Stats[layerIdx]:
                                                        Stats[layerIdx]["VDD_SYS_CPU"] = []
                                                    Stats[layerIdx]["VDD_SYS_CPU"].extend(stat)
                                            elif Device == "gpu":
                                                if statName == "VDD_SYS_GPU":
                                                    if layerIdx not in Stats:
                                                        Stats[layerIdx] = {}
                                                    if "VDD_SYS_GPU" not in Stats[layerIdx]:
                                                        Stats[layerIdx]["VDD_SYS_GPU"] = []
                                                    Stats[layerIdx]["VDD_SYS_GPU"].extend(stat)

                            # Calculate average values for all the layers
                            for layerIdx, statDict in Stats.items():
                                NetStats[layerIdx] = {}
                                for statName, stat in statDict.items():
                                    NetStats[layerIdx][statName] = float(sum(stat))/len(stat)

                        # Store processed values for each Network    
                        ParameterStats[net] = NetStats
                    
                    # Store processed values for each Parameter    
                    DeviceStats[Parameter] = ParameterStats
                
                # Store processed values for each Device    
                ModeStats[Device] = DeviceStats

            # Store processed values for each Mode    
            ModeWiseStats[modeID] = ModeStats

        return ModeWiseStats, ModeWiseNetworks

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == "__main__":
    try:
        args = arg_parser()
        ModeWiseStats , ModeWiseNetworks = processModeStats(args)
        print("Processed Networks for each mode: ",ModeWiseNetworks)
        print("Processed Mode stats: ")
        for Parameter, statDict in ModeWiseStats[0]["cpu"].items():
            print("\nParameter: ", Parameter)
            for net, stat in statDict.items():
                print(f"Network: {net}")
                print(stat)

    except Exception as e:
        print("Error found: ",e)
