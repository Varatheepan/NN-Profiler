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
AvailModes = [int(mode.split("Mode")[1]) for mode in AvailModes if (os.path.isdir(os.path.join(project_path,"Dataset",mode)) and "Mode" in mode)]

def arg_parser():
    parser = argparse.ArgumentParser(
        description= "This takes `MODE` as the argument to process")
    parser.add_argument("--modes", default=AvailModes,type=list, help="list of mode numbers to process")
    parser.add_argument("--params", default=[])
    parser.add_argument("--output_file_root", default=os.path.join(project_path,"Dataset/Processed"), help="The root directory to store all the mode stats files")
    return parser.parse_args()

def processModeStats(args):

    try:
        # Object to store all processed data for each mode in the order Mode-Device-Parameter-Network
        ModeWiseStats = {}

        # Networks available across each mode
        ModeWiseNetworks = {}

        # get the modes to post process
        modes = args.modes

        # Flag to overwrite tegrastats parameters to process
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

                                # Extract the tegrastats sampling related  arguments
                                nameSplit = imgSample[:-5].split("_")
                                sampligFreq = int(nameSplit[2])
                                samplingBoundry = int(nameSplit[4])

                                elimNumSamples = sampligFreq*samplingBoundry

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
                                                    # Values inside the intended stablized window
                                                    statWindow = stat[elimNumSamples:-elimNumSamples]
                                                    if layerIdx not in Stats:
                                                        Stats[layerIdx] = {}
                                                    if "VDD_SYS_CPU" not in Stats[layerIdx]:
                                                        Stats[layerIdx]["VDD_SYS_CPU"] = []
                                                    Stats[layerIdx]["VDD_SYS_CPU"].extend(statWindow)
                                            elif Device == "gpu":
                                                if statName == "VDD_SYS_GPU":
                                                    # Values inside the intended stablized window
                                                    statWindow = stat[elimNumSamples:-elimNumSamples]
                                                    if layerIdx not in Stats:
                                                        Stats[layerIdx] = {}
                                                    if "VDD_SYS_GPU" not in Stats[layerIdx]:
                                                        Stats[layerIdx]["VDD_SYS_GPU"] = []
                                                    Stats[layerIdx]["VDD_SYS_GPU"].extend(statWindow)

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

# Format processed data. For each mode stats of each network
def formatProcessedData(args,ModeWiseStats,ModeWiseNetworks):

    try:
    
        # Object to store all formatted data for each mode in the order Mode-Network-Device-Parameter
        ModeWiseStatsF = {}

        for modeID in ModeWiseNetworks.keys():
            ModeWiseStatsF[modeID] = {}

            for net in ModeWiseNetworks[modeID]:
                ModeWiseStatsF[modeID][net] = {}

                for Device in ModeWiseStats[modeID].keys():
                    ModeWiseStatsF[modeID][net][Device] = {}

                    for Parameter in ModeWiseStats[modeID][Device].keys():
                        if Device in ModeWiseStats[modeID]:
                            if Parameter in ModeWiseStats[modeID][Device]:
                                if net in ModeWiseStats[modeID][Device][Parameter]:
                                    ModeWiseStatsF[modeID][net][Device][Parameter] = ModeWiseStats[modeID][Device][Parameter][net]
        
        return ModeWiseStatsF 
    
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def storeDataset(args,ModeWiseStatsF):
    FileRoot= args.output_file_root

    if not os.path.exists(FileRoot):
        os.makedirs(FileRoot)
    for modeID in ModeWiseStatsF.keys():
        filePath = os.path.join(FileRoot,"Mode"+str(modeID)+".json")
        with open(filePath,"w") as jsonFile:
            json.dump(ModeWiseStatsF[modeID],jsonFile)

if __name__ == "__main__":
    try:
        args = arg_parser()

        # Process the raw data an get the averages
        ModeWiseStats , ModeWiseNetworks = processModeStats(args)

        # Fprmat the order to Mode-Network-Device-Parameter
        ModeWiseStatsF = formatProcessedData(args,ModeWiseStats,ModeWiseNetworks)

        storeDataset(args,ModeWiseStatsF)

        """
        # Print statement for manual verification
        print("Processed Networks for each mode: ",ModeWiseNetworks)
        print("Processed Mode stats: \n")
        
        for net, statDict1 in ModeWiseStatsF[1].items():
            print("\nNetwork: ", net)
            for Device, statDict2 in statDict1.items():
                print(f"\tDevice: {Device}")
                for Parameter, stat in statDict2.items():
                    print(f"\t\tParameter: {Parameter}")
                    print(f"\t\t\t{stat}")
            print("#########################################################################################")"""

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
