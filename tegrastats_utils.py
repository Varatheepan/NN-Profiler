# This function will extract the power (or other parameters needed) from tegrastats output
def analyze_power_stats(txt_file: str, device,sampling_boundry, layerSamplingTime,sampling_freq):
    stats = open(txt_file,"r")
    content = stats.readlines()

    powerStats = []
    freq = 0
    last_val = 0
    target_data = "VDD_SYS_CPU"         #default value

    if device == "cpu":
        target_data = "VDD_SYS_CPU"
    elif device == "gpu":
        target_data = "VDD_SYS_GPU"
    else:
        print(f"Devices \"{device}\" is not identified. Device defaults to CPU.")

    for j,line in enumerate(content):
        lineSplit = line.split(" ")
        for i,txt in enumerate(lineSplit):
            if txt == target_data:
                tempval = int(lineSplit[i+1].split("/")[0])
                powerStats.append(tempval)
    # sampling_set = powerStats[int(sampling_boundry*sampling_freq):int(sampling_boundry*sampling_freq)+int(layerSamplingTime*sampling_freq)]
    # print(f"avg power:  {sum(sampling_set)/len(sampling_set)} over {len(sampling_set)} middle samples")

    return powerStats