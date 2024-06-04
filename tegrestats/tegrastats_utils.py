import os, sys
import re

# This function will extract the power (or other parameters needed) from tegrastats output

def analyze_power_stats(txt_file: str, device,parameters: list, jetsonDevice: str = "Tx2"):
    try:
        stats = open(txt_file,"r")
        content = stats.readlines()

        ReadingOutput = {}

        tx2_extractor = {
        "RAM": r"RAM (\d+/\d+MB)",
        "SWAP": r"SWAP (\d+/\d+MB)",
        "CPU": r"CPU \[([^\]]+)\]",
        "EMC_FREQ": r"EMC_FREQ (\d+%@\d+)",
        "GR3D_FREQ": r"GR3D_FREQ (\d+%@\d+)",
        "APE": r"APE (\d+)",
        "PLL": r"PLL@(\d+(\.\d+)?C)",
        "MCPU": r"MCPU@(\d+(\.\d+)?C)",
        "BCPU": r"BCPU@(\d+(\.\d+)?C)",
        "Thermal": r"thermal@(\d+(\.\d+)?C)",
        "Tboard": r"Tboard@(\d+(\.\d+)?C)",
        "GPU": r"GPU@(\d+(\.\d+)?C)",
        "Tdiode": r"Tdiode@(\d+(\.\d+)?C)",
        "PMIC": r"PMIC@(\d+(\.\d+)?C)",        
        "VDD_SYS_GPU": r"VDD_SYS_GPU (\d+/\d+)",
        "VDD_SYS_CPU": r"VDD_SYS_CPU (\d+/\d+)",
        "VDD_SYS_SOC": r"VDD_SYS_SOC (\d+/\d+)",
        "VDD_IN": r"VDD_IN (\d+/\d+)",
        "VDD_SYS_DDR": r"VDD_SYS_DDR (\d+/\d+)",
        "VDD_4V0_WIFI": r"VDD_4V0_WIFI (\d+/\d+)",
        }
        
        orin_extractor = {
        "Timestamp": r"(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})",
        "RAM": r"RAM (\d+/\d+MB)",
        "SWAP": r"SWAP (\d+/\d+MB)",
        "CPU": r"CPU \[([^\]]+)\]",
        "EMC_FREQ": r"EMC_FREQ (\d+%@\d+)",
        "GR3D_FREQ": r"GR3D_FREQ (\d+%@\[\d+\])",
        "NVDEC": r"NVDEC (on|off)",
        "NVJPG": r"NVJPG (on|off)",
        "NVJPG1": r"NVJPG1 (on|off)",
        "VIC": r"VIC (on|off)",
        "OFA": r"OFA (on|off)",
        "APE": r"APE (\d+)",
        "CPU Temperature": r"cpu@(\d+(\.\d+)?C)",
        "SOC 2 Temperature": r"soc2@(\d+(\.\d+)?C)",
        "SOC 0 Temperature": r"soc0@(\d+(\.\d+)?C)",
        "GPU Temperature": r"gpu@(\d+(\.\d+)?C)",
        "Tjunction Temperature": r"tj@(\d+(\.\d+)?C)",
        "SOC 1 Temperature": r"soc1@(\d+(\.\d+)?C)",
        "IN Power": r"VDD_IN (\d+mW/\d+mW)",
        "GPU CPU CV Power": r"VDD_CPU_GPU_CV (\d+mW/\d+mW)",
        "SOC Power": r"VDD_SOC (\d+mW/\d+mW)"
        }
        
        agx_extractor = {
        "Timestamp": r"(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})",
        "RAM": r"RAM (\d+/\d+MB)",
        "SWAP": r"SWAP (\d+/\d+MB)",
        "CPU": r"CPU \[([^\]]+)\]",
        "EMC_FREQ": r"EMC_FREQ (\d+%@\d+)",
        "GR3D_FREQ": r"GR3D_FREQ (\d+%@\[\d+\])",
        "VIC": r"VIC_FREQ (\d+)",
        "APE": r"APE (\d+)",
        "Aux Temperature": r"AUX@(\d+(\.\d+)?C)",
        "CPU Temperature": r"CPU@(\d+(\.\d+)?C)",
        "Thermal sensor": r"thermal@(\d+(\.\d+)?C)",
        "Tboard Temperature": r"Tboard@(\d+(\.\d+)?C)",
        "AO Temperature": r"AO@(\d+(\.\d+)?C)",
        "GPU Temperature": r"GPU@(\d+(\.\d+)?C)",
        "Tdiode Temperature": r"Tdiode@(\d+(\.\d+)?C)",
        "PMIC Temperature": r"PMIC@(\d+(\.\d+)?C)",        
        "GPU Power": r"GPU (\d+mW/\d+mW)",
        "CPU Power": r"CPU (\d+mW/\d+mW)",
        "SOC Power": r"SOC (\d+mW/\d+mW)",
        "CV Power": r"CV (\d+mW/\d+mW)",
        "VDDRQ Power": r"VDDRQ (\d+mW/\d+mW)",
        "SYS5V Power": r"SYS5V (\d+mW/\d+mW)",
        }
        
        if jetsonDevice == "Tx2":
            extractor = tx2_extractor
        elif jetsonDevice == "Orin":
            extractor = orin_extractor
        elif jetsonDevice == "Xavier":
            extractor = agx_extractor
        
        for param, pattern in extractor.items():
            ReadingOutput[param] = []

        for line in content:
            row = {}
            for metric, pattern in extractor.items():
                match = re.search(pattern, line)
                if match:
                    row[metric] = match.group(1)
                    # Readings[metric].append(match.group(1))

            # print(f"Row: {row}")

            # Ram and Swap parameters
            ReadingOutput["RAM"].append(int(row["RAM"].split("/")[0]))
            ReadingOutput["SWAP"].append(int(row["SWAP"].split("/")[0]))
    
            # CPU utilization and running frequencies
            if len(ReadingOutput["CPU"]) == 0:
                ReadingOutput["CPU"] = {"Utilization":[],"Frequencies":[]}
            tempvals = row["CPU"].split(",")
            tempUtil = []
            tempFreq = []
            for tempval in tempvals:
                if tempval == "off":
                    tempUtil.append(-1)
                    tempFreq.append(-1)
                else:  
                    vals = tempval.split("%@")
                    tempUtil.append(int(vals[0]))
                    tempFreq.append(int(vals[1]))
            ReadingOutput["CPU"]["Utilization"].append(tempUtil)
            ReadingOutput["CPU"]["Frequencies"].append(tempFreq)

            #EMC utilization and running frequencies
            if len(ReadingOutput["EMC_FREQ"]) == 0:
                ReadingOutput["EMC_FREQ"] = {"Utilization":[],"Frequencies":[]}
            ReadingOutput["EMC_FREQ"]["Utilization"].append(int(row["EMC_FREQ"].split("%@")[0]))
            ReadingOutput["EMC_FREQ"]["Frequencies"].append(int(row["EMC_FREQ"].split("%@")[1]))

            #APE utilization
            ReadingOutput["APE"].append(int(row["APE"]))

            if jetsonDevice == "Tx2":
                #GR3D utilization and running frequencies
                if len(ReadingOutput["GR3D_FREQ"]) == 0:
                    ReadingOutput["GR3D_FREQ"] = {"Utilization":[],"Frequencies":[]}
                ReadingOutput["GR3D_FREQ"]["Utilization"].append(int(row["GR3D_FREQ"].split("%@")[0]))
                ReadingOutput["GR3D_FREQ"]["Frequencies"].append(int(row["GR3D_FREQ"].split("%@")[1]))
            
                #Temperature parameters
                for temp in ["PLL","MCPU","BCPU","Thermal","Tboard","GPU","Tdiode","PMIC"]:
                    ReadingOutput[temp].append(float(row[temp].split("C")[0]))

                #Power parameters
                for temp in ["VDD_SYS_GPU","VDD_SYS_CPU","VDD_SYS_SOC","VDD_4V0_WIFI","VDD_IN","VDD_SYS_DDR"]:
                    ReadingOutput[temp].append(int(row[temp].split("/")[0]))
                    if temp+"_Avg" not in ReadingOutput:
                        ReadingOutput[temp+"_Avg"] = []
                    ReadingOutput[temp+"_Avg"].append(int(row[temp].split("/")[1]))
            
            elif jetsonDevice == "Orin":
                #GR3D utilization and running frequencies
                if len(ReadingOutput["GR3D_FREQ"]) == 0:
                    ReadingOutput["GR3D_FREQ"] = {"Utilization":[],"Frequencies":[]}
                ReadingOutput["GR3D_FREQ"]["Utilization"].append(int(row["GR3D_FREQ"].split("%@")[0]))
                ReadingOutput["GR3D_FREQ"]["Frequencies"].append(int(row["GR3D_FREQ"].split("%@")[1][1:-1]))

                #Temperature parameters
                for temp in ["CPU Temperature","SOC 2 Temperature","SOC 0 Temperature","GPU Temperature","Tjunction Temperature","SOC 1 Temperature"]:
                    ReadingOutput[temp].append(float(row[temp].split("C")[0]))

                #Power parameters
                for temp in ["IN Power","GPU CPU CV Power","SOC Power"]:
                    ReadingOutput[temp].append(int(row[temp].split("/")[0][:-2]))
                    if temp+"_Avg" not in ReadingOutput:
                        ReadingOutput[temp+"_Avg"] = []
                    ReadingOutput[temp+"_Avg"].append(int(row[temp].split("/")[1][:-2]))
                
                for temp in ["NVDEC","NVJPG","NVJPG1","VIC","OFA"]:
                    ReadingOutput[temp].append(row[temp])
            
            elif jetsonDevice == "Xavier":
                #GR3D utilization and running frequencies
                if len(ReadingOutput["GR3D_FREQ"]) == 0:
                    ReadingOutput["GR3D_FREQ"] = {"Utilization":[],"Frequencies":[]}
                ReadingOutput["GR3D_FREQ"]["Utilization"].append(int(row["GR3D_FREQ"].split("%@")[0]))
                ReadingOutput["GR3D_FREQ"]["Frequencies"].append(int(row["GR3D_FREQ"].split("%@")[1][1:-1]))

                #Temperature parameters
                for temp in ["Aux Temperature","CPU Temperature","Thermal sensor","Tboard Temperature","AO Temperature","GPU Temperature","Tdiode Temperature","PMIC Temperature"]:
                    ReadingOutput[temp].append(float(row[temp].split("C")[0]))

                #Power parameters
                for temp in ["GPU Power","CPU Power","SOC Power","CV Power","VDDRQ Power","SYS5V Power"]:
                    ReadingOutput[temp].append(int(row[temp].split("/")[0][:-2]))
                    if temp+"_Avg" not in ReadingOutput:
                        ReadingOutput[temp+"_Avg"] = []
                    ReadingOutput[temp+"_Avg"].append(int(row[temp].split("/")[1][:-2]))
                
                ReadingOutput["VIC"].append(int(row["VIC"]))

        return ReadingOutput
    except Exception as e:
        print(f"Error: {e}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return exc_type
    

def get_params(jetsonDevice: str = None):
    if jetsonDevice is None:
        return None
    
    if jetsonDevice == "Tx2":
        return ['RAM', 'SWAP', 'CPU', 'EMC_FREQ', 'GR3D_FREQ', 'APE', 'PLL', 'MCPU', 'BCPU', 'Thermal', 'Tboard', 'GPU', 'Tdiode', 'PMIC', 'VDD_SYS_GPU', 'VDD_SYS_CPU', 'VDD_SYS_SOC', 'VDD_IN', 'VDD_SYS_DDR', 'VDD_4V0_WIFI']    
    elif jetsonDevice == "Xavier":
        return ['RAM', 'SWAP', 'CPU', 'EMC_FREQ', 'GR3D_FREQ', 'VIC', 'APE', 'Aux Temperature', 'CPU Temperature', 'Thermal sensor', 'Tboard Temperature', 'AO Temperature', 'GPU Temperature', 'Tdiode Temperature', 'PMIC Temperature', 'GPU Power', 'CPU Power', 'SOC Power', 'CV Power', 'VDDRQ Power', 'SYS5V Power']
    elif jetsonDevice == "Orin":
        return ['RAM', 'SWAP', 'CPU', 'EMC_FREQ', 'GR3D_FREQ', 'NVDEC', 'NVJPG', 'NVJPG1', 'VIC', 'OFA', 'APE', 'CPU Temperature', 'SOC 2 Temperature', 'SOC 0 Temperature', 'GPU Temperature', 'Tjunction Temperature', 'SOC 1 Temperature', 'IN Power', 'GPU CPU CV Power', 'SOC Power']
    else:
        return None