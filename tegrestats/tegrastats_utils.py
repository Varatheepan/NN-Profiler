import os, sys

# This function will extract the power (or other parameters needed) from tegrastats output
# TODO: define CPU utilization & frequency, EMC frequency parameters
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
            for i,txt in enumerate(lineSplit):
                if txt == parameters[paramIndex]:
                    if txt != "CPU": 
                        tempval = int(lineSplit[i+1].split("/")[0])
                        Readings[txt].append(tempval)
                        paramIndex += 1
                    # else:     # TODO: Define for CPU utilization parameters 
                elif txt.split("@")[0] == parameters[paramIndex]:
                    tempval = float(txt.split("@")[1].split("C")[0])
                    Readings[txt.split("@")[0]].append(tempval)
                    paramIndex += 1
                if paramIndex == len(parameters):
                    paramIndex = 0
                    break
        return Readings
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return exc_type