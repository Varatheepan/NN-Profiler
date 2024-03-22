<!-- TODO: add requirements -->
# Power and Latency stats 

This repository can be used to generate data of layer-wise power and latency in Jetson Tx2. The set of models supported can be found in `AvailableModel.txt`. 
The Power and other internal Tx2 related parameters are extracted using [Tegrastats](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).

# Usage

## Generating layer wise evaluation raw data
To generate layer wise data for the list of available networks, run the following command.
Note that the ```mode``` parameter given will not change the mode of the Tx2. The mode should be set before hand.
If the device is not specified, it will default to ```CPU```.
```powershell
python3 layer_stats/rawDataGeneration/PowerLatencyDatasetGenerator.py --mode 1 --device cpu
```

The above command will generate raw data files in the folder structure ```Mode-Device-Parameter-Network```.

The following are the arguments to customize the analysis.
```
- mode: The mode that is being experimented. Make sure to switch the mode in Tx2. Required parameter.
- device: he device to run the experiment on (default = "cpu")  choices=["cpu","cuda"]
- imgs: A comma separated list of images to run the experiments. Images should be stored in data/imagenet directory.

Tegrastats related parameters

- tgr_params: A space separated list of parameters from the defaults list of parameters to extract from tegrastats.
    default=[RAM, SWAP,CPU,EMC_FREQ,MCPU,GPU,BCPU,VDD_SYS_GPU,VDD_SYS_SOC,VDD_SYS_CPU,VDD_SYS_DDR]
- eval_tgr: Whether to evaluate tegrastats parameters. Intended for power evaluation(store_true).
- tgr_interval: The tegrastats data sampling interval. Data sampled at every 'tgr_interval' mS time (default=20).
- tgr_smpl_duration:The time duration to run sample tegrastats data for a layer. default=5.
- tgr_smpl_boundary: Time window allowed for the tegrastats to stabilize for the layer being experimented (default=1).

Latency related parameters

- eval_latency: Whether to evaluate latency. Intended for power evaluation(store_true). 
- lt_smpl_count: The number of times to run a layer for the latency experiment (default=100).

```


## Post process data
To post process the data and get averaged layer-wise values for each parameter, run
```powershell
python3 layer_stats/processedDataGeneration/processData.py
```
This will create a json file for each mode present (or specified). Each json file file will have network wise stats in the ```Network-Device-Parameter``` order.</br>
An example mode data will look like following.
```python
{
    "efficientnet_b2": {
        "gpu": {
            "power": {
                "0": {
                    "VDD_SYS_GPU": 234.62
                },
                "1": {
                    "VDD_SYS_GPU": 1228.79
                }
                .
                .
            },
            "latency": {
                "0": 0.014187506437301635,
                "1": 0.006302738189697265,
                "2": 0.01274215817451477,
                .
                .
            }
        },
        "cpu": {
            "power": {
                "0": {
                    "VDD_SYS_CPU": 1151.52
                },
                .
                .
            },
            "latency": {
                "0": 0.052880855798721316,
                "1": 0.20610104918479918,
                .
                .
            }
        }
    },
    "alexnet": {
        .
        .
    }
}
``` 

