import os, sys

# This function will extract the power (or other parameters needed) from tegrastats output

def analyze_power_stats(txt_file: str, device,parameters: list):
    try:
        stats = open(txt_file,"r")
        content = stats.readlines()

        if len(parameters) == 0:
            parameters = ["VDD_SYS_CPU","VDD_SYS_GPU"]
            print("Empty parameter list given. Parameter list Defaults to [\"VDD_SYS_CPU\",\"VDD_SYS_GPU\"]")

        paramIndex = 0
        Readings = {param:[] for param in parameters}

        for j,line in enumerate(content):
            lineSplit = line.split(" ")
            lineSplit = [lineS.strip() for lineS in lineSplit]
            for i,txt in enumerate(lineSplit):
                if txt == parameters[paramIndex]:

                    # Power parameters, RAM and SWAP parameters
                    if (txt != "CPU") and (txt != "EMC_FREQ") and (txt != "GR3D_FREQ"): 
                        tempval = int(lineSplit[i+1].split("/")[0])
                        Readings[txt].append(tempval)
                        paramIndex += 1

                    # Entries for CPU utilization and running frequencies
                    elif txt == "CPU":
                        if len(Readings["CPU"]) == 0:
                            Readings["CPU"] = {"Utilization":[],"Frequencies":[]}
                        tempvals = lineSplit[i+1][1:-1].split(",")
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
                        Readings["CPU"]["Utilization"].append(tempUtil)
                        Readings["CPU"]["Frequencies"].append(tempFreq) 
                        paramIndex += 1
                    
                    # Entries for RAM frequency
                    elif txt == "EMC_FREQ":
                        if len(Readings["EMC_FREQ"]) == 0:
                            Readings["EMC_FREQ"] = {"Utilization":[],"Frequencies":[]}
                        tempvals = lineSplit[i+1].split("%@")
                        # Readings["EMC_FREQ"] = {"Utilization":int(tempvals[0]),"Frequencies":int(tempvals[1])}
                        Readings["EMC_FREQ"]["Utilization"].append(int(tempvals[0]))
                        Readings["EMC_FREQ"]["Frequencies"].append(int(tempvals[1])) 
                        paramIndex +=  1

                    # Entries for RAM frequency
                    elif txt == "GR3D_FREQ":
                        if len(Readings["GR3D_FREQ"]) == 0:
                            Readings["GR3D_FREQ"] = {"Utilization":[],"Frequencies":[]}
                        tempvals = lineSplit[i+1].split("%@")
                        # Readings["GR3D_FREQ"] = {"Utilization":int(tempvals[0]),"Frequencies":int(tempvals[1])}
                        Readings["GR3D_FREQ"]["Utilization"].append(int(tempvals[0]))
                        Readings["GR3D_FREQ"]["Frequencies"].append(int(tempvals[1])) 
                        paramIndex +=  1

                
                # Temperature parameters
                elif txt.split("@")[0] == parameters[paramIndex]:
                    tempval = float(txt.split("@")[1].split("C")[0])
                    Readings[txt.split("@")[0]].append(tempval)
                    paramIndex += 1

                # Reset parameter index for the next line 
                if paramIndex == len(parameters):
                    paramIndex = 0
                    break
        return Readings
    except Exception as e:
        # print(f"txt: `{txt}` next txt: {lineSplit[i+1]} split: {lineSplit[i+1].split('/')} Param: {parameters[paramIndex]}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return exc_type