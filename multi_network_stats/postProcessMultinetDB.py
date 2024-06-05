import argparse
from copy import deepcopy
import os
from pathlib import Path
import csv
import sys
import numpy as np
import json


# TODO: store standard deviation with all the avearage parameters

# Project path
project_path = Path(__file__).resolve().parents[1]

# AvailModes = os.listdir(os.path.join(project_path,"Dataset/Multinet"))
# AvailModes = [int(mode.split("Mode")[1]) for mode in AvailModes if (os.path.isdir(os.path.join(project_path,"Dataset/Multinet",mode)) and "Mode" in mode)]

def arguments_parser():
    parser = argparse.ArgumentParser(
        description= "This takes `Device MODEs` as the arguments to process")
    parser.add_argument("--tx2Modes", type=int, nargs='+', help="The modes to process for the TX2")
    parser.add_argument("--xavierModes", type=int, nargs='+', help="The modes to process for the Xavier")
    parser.add_argument("--orinModes", type=int, nargs='+', help="The modes to process for the Orin")
        
    parser.add_argument("--output_file_root", default=os.path.join(project_path,"Dataset/Multinet/Processed"), help="The root directory to store all the mode stats files")
    parser.add_argument("--input_file_root", default=os.path.join(project_path,"Dataset/Multinet"), help="The root directory to read all the mode stats files")
    return parser.parse_args()

def getParams(args):

    Params = {}

    if args.tx2Modes:
        Params["Tx2"] = ["RAM", "SWAP","CPU","EMC_FREQ","GR3D_FREQ","APE","PLL","MCPU","PMIC","Tboard","GPU","BCPU","Thermal","Tdiode","VDD_SYS_GPU","VDD_SYS_SOC","VDD_4V0_WIFI","VDD_IN","VDD_SYS_CPU","VDD_SYS_DDR"]
    if args.xavierModes:
        Params["Xavier"] = ['RAM', 'SWAP', 'CPU', 'EMC_FREQ', 'GR3D_FREQ', 'VIC', 'APE', 'Aux Temperature', 'CPU Temperature', 'Thermal sensor', 'Tboard Temperature', 'AO Temperature', 'GPU Temperature', 'Tdiode Temperature', 'PMIC Temperature', 'GPU Power', 'CPU Power', 'SOC Power', 'CV Power', 'VDDRQ Power', 'SYS5V Power']
    if args.orinModes:
        Params["Orin"] = ['RAM', 'SWAP', 'CPU', 'EMC_FREQ', 'GR3D_FREQ', 'NVDEC', 'NVJPG', 'NVJPG1', 'VIC', 'OFA', 'APE', 'CPU Temperature', 'SOC 2 Temperature', 'SOC 0 Temperature', 'GPU Temperature', 'Tjunction Temperature', 'SOC 1 Temperature', 'IN Power', 'GPU CPU CV Power', 'SOC Power']
    
    return Params
    

def processMode(args):
    try:
        DevParams = getParams(args)

        if len(DevParams) == 0:
            print("No modes provided to process!")
            return

        # Process each available device
        for device, params in DevParams.items():

            # Get the modes to process
            if device == "Tx2":
                modes = args.tx2Modes
            elif device == "Xavier":
                modes = args.xavierModes
            elif device == "Orin":
                modes = args.orinModes
            
            # Process each mode
            for mode in modes:
                
                NumFilesCompleted = 0

                ModeS = f"Mode{mode}"

                ModeRoot = os.path.join(project_path,args.input_file_root, ModeS)

                print(f"Mode root: {ModeRoot}")

                OutFileRoot= os.path.join(args.output_file_root,ModeS)

                if not os.path.exists(OutFileRoot):
                    os.makedirs(OutFileRoot)

                fileList = os.listdir(ModeRoot)

                fileList = sorted(fileList, key=lambda a: int(a.split("Map")[1].split("_")[0]))
                # print(fileList)

                for fileName in fileList:

                    Stats = {}

                    StatsFile = open(os.path.join(ModeRoot, fileName), "r")
                    line = StatsFile.readline().strip()
                    line = json.loads(line)

                    Stats = deepcopy(line)
                    Stats["Mode"] = mode
                    Stats["Device"] = device
                    del Stats["power"]
                    Stats["Tegrastats"] = {}
                    Stats["ModelInfo"] = {}

                    for param in params:
                        # print(f"Processing param: {param}")
                        paramStat = line["power"][param]

                        if type(paramStat) == list:
                            if len(paramStat) != 0 and type(paramStat[0]) == str:
                                Stats["Tegrastats"][param] = paramStat[0]
                            else:  
                                Stats["Tegrastats"][param] = np.round(float(sum(paramStat))/len(paramStat),4)
                        
                        elif type(paramStat) == dict:
                            Stats["Tegrastats"][param] = {}

                            if param == "CPU":
                                for key, val in paramStat.items():
                                    Stats["Tegrastats"][param][key] = {"min":None, "max":None, "avg":None}
                                    tempVal = np.array(val)
                                    Stats["Tegrastats"][param][key]["avg"] = np.round(np.mean(tempVal,axis=0),2).tolist()
                                    Stats["Tegrastats"][param][key]["min"] = np.round(np.min(tempVal,axis=0),2).tolist()
                                    Stats["Tegrastats"][param][key]["max"] = np.round(np.max(tempVal,axis=0),2).tolist()
                            else:
                                for key, val in paramStat.items():
                                    Stats["Tegrastats"][param][key] = {"min":None, "max":None, "avg":None}
                                    Stats["Tegrastats"][param][key]["avg"] = round(float(sum(val))/len(val),2)
                                    Stats["Tegrastats"][param][key]["min"] = round(min(val),2)
                                    Stats["Tegrastats"][param][key]["max"] = round(max(val),2)

                    for net, netStats in line["stageSummary"].items():

                        # Add model parameters, macs and flops
                        Stats["ModelInfo"][net] = {}

                        for compComponent, DevStats in netStats.items():

                            del Stats["stageSummary"][net][compComponent]["infDurations"]
                            if len(DevStats["infDurations"]) == 0:
                                Stats["stageSummary"][net][compComponent]["avgInfTime"] = -1
                            else:
                                Stats["stageSummary"][net][compComponent]["avgInfTime"] = sum(DevStats["infDurations"])/len(DevStats["infDurations"])
                            
                            del Stats["stageSummary"][net][compComponent]["transferDurations"]
                            if len(DevStats["transferDurations"]) == 0:
                                Stats["stageSummary"][net][compComponent]["avgTransferTime"] = -1
                            else:
                                Stats["stageSummary"][net][compComponent]["avgTransferTime"] = sum(DevStats["transferDurations"])/len(DevStats["transferDurations"])
                            
                            del Stats["stageSummary"][net][compComponent]["macs"]
                            del Stats["stageSummary"][net][compComponent]["params"]
                            del Stats["stageSummary"][net][compComponent]["flops"]

                            Stats["ModelInfo"][net][compComponent] = {"params":DevStats["params"], "macs":DevStats["macs"].split("MACs")[0], "flops":DevStats["flops"], "functionalLayerCount":DevStats["functionalLayerCount"]}
                    

                    # for statsKey, statsVal in Stats["Tegrastats"].items():
                    #     print(f"{statsKey}: {statsVal}")

                    newFileName = f"Mode{mode}_"+"_".join(fileName.split("_")[1:])
                    filePath = os.path.join(OutFileRoot,newFileName)
                    with open(filePath,"w") as jsonFile:
                        json.dump(Stats,jsonFile)

                    NumFilesCompleted += 1

                    if NumFilesCompleted % 50 == 0:
                        print(f"Num file completed: {NumFilesCompleted}")
                
                print(f"Mode: {mode}, Num file completed: {NumFilesCompleted}")
            
    except Exception as e1:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
                    
if __name__ == "__main__":
    try:
        args = arguments_parser()

        processMode(args)

        print("Post process complete!")

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
