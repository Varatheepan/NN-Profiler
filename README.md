<!-- TODO: add requirements -->
# DNN stats 

This repository can be used to generate data of layer-wise and Multi-DNN workload power, latency `(and several other parameters in tegrastats)` on Jetson Tx2, Orin and AGx Xavier devices. The set of models supported can be found in `AvailableModel.txt`. 
The Power and other internal Tegra-SOC related parameters are extracted using [Tegrastats](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html).

# Usage

## Layer-wise data generation

### Raw data generation
To generate layer wise data for the list of available networks, run the following command.
Note that the ```mode``` parameter given will not change the mode of the device. The mode should be set before hand.

To change the mode to mode0, run
```powershell
sudo nvpmodel -m 0 
```

If the device is not specified, it will default to ```CPU```.
```powershell
python3 layer_stats/rawDataGeneration/PowerLatencyDatasetGenerator.py --mode 1 --device cpu
```

The above command will generate raw data files in the folder structure ```Mode-Device-Parameter-Network```.

The following are the arguments to customize the analysis.
```
- mode: The mode that is being experimented. Make sure to switch the mode in device. Required parameter.
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

### Post processing
To post process the data and get averaged layer-wise values for each parameter, run
```powershell
python3 layer_stats/processedDataGeneration/processData.py
```
This will create a json file for each mode present (or specified). Each json file file will have network wise stats in the ```Network-Device-Parameter``` order.

<details>
<summary>An example mode data will look like following.</summary>

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
</details>

## Multi-DNN workload data generation

### Raw data generation

The following command command will generate Multi-DNN workload evaluation data.
Note that the ```mode``` parameter given will not change the mode of the device. The mode should be set before hand.

```powershell
sudo ../torch210/bin/python3 MappingDBGenerator.py --jetsonDevice Orin --mode 1 --num_samples 2000 --eval_tgr --eval_thr --max_split_nets 5 --device_priorities 0.4,0.6 --numNetsRange 2 6 --single_nets --gpu_only_maps --samples_per_set 20 --seed 5 --funcLayerCount --log_level INFO --smpl_duration 30 --warmup &
```

The list of arguments is provided below.
```
- jetsonDevice {Orin,Tx2,Xavier}: The Jetson device to run the experiment on.
- mode MODE: The mode that is being experimented. Make sure to switch the mode in device.
- num_samples NUM_SAMPLES: the number of samples to be generated for the selected mode (default=20).
- device_list DEVICE_LIST: The devices to run the experiment on (default=["cpu","cuda"]).
- device_priorities DEVICE_PRIORITIES: The proirities of using the devices for mapping. Default will give equal priority each device.
- model_list MODEL_LIST: The list of models to be used to generate the dataset. Deafaults to the entire list of model which can be mplemented in the device of interest.
- single_nets: Whether to generate single network mappings.
- gpu_only_maps: Whether to generate mappings with only GPU devices.
- numNetsRange: The range of number of networks to be used in the mapping(default=2 6).
- samples_per_set SAMPLES_PER_SET: The number of samples to be generated for each set of networks (default=20).
- imgs: A comma seperated list of images to run the experimets. Images should be stored in data/imagenet.
- smpl_duration: The time interval to to run a workload for the measurements (default=30).
- funcLayerCount: Whether to count the number of functional layers in the network.
- tgr_params: A space seperatedlist of parameters to extract from tegratstats. Defaults to the list parameters that can be sampled from the device of interest.
- eval_tgr: Whether to evaluate tegrastats parameters. Intended for power evaluation.
- tgr_interval: The tegrastats data sampling interval. Data sampled at every `tgr_interval` mS time (default=50).
- tgr_smpl_boundry: Time window allowed for the tegratstats to stablize for the layer being experimented in seconds(default=2).
- eval_thr: Whether to evaluate throughput.
- max_split_nets: The maximum number of networks to be split in a mapping(default=-1 : no limit on the number of split networks. All the networks in the worload can be split).
- seed: The seed to be used for random number generation(default=0). By deafault the mode number will be set as the seed for that mode. Settig a value other that 0 will allow the same mappings to be generated across modes.
- batch_size: The batch size to be used for the inference (default=1).
- remove_weights: Whether to remove the weights of the model after each mapping. This is enabled to handle storage limitation of the device considered.
- log_file: The log file path to store the logs of the experiment.
- log_level: The log level to be used for the experiment(default=INFO). By default, logs will be stored in the log file. Passing DEBUG will enable logging in console and the file. 
- warmup: Whether to warm up the system before the experiment.
- start_idx: The index to start the mapping generation. In case of interuption during the generation, data generation can be applied from the recently completed number of networks category by changing the --numNetsRange and this parameter to the start of the current number of networks category. This eleminates the need to re-run the entire mode and re-runs from only the last procssed number of networks category. TODO: resume from the last sample.
```
### Post processing
To post process the data and get averaged values for each parameter, run
```powershell
python3 multi_network_stats/postProcessMultinetDB.py --orinModes 0 1
```

The following arguments can be passed to post process selective modes of corresponding devices.

```
- xavierModes: Post process the space seperated list of modes of Agx Xavier.
- orinModes: Post process the space seperated list of modes of Orin.
- tx2Modes: Post process the space seperated list of modes of Tx2.
```

This will create json files for each sample present in the mode from the mode list.

<details>
<summary>AN example workload of 4 DNNs.</summary>

```python
{
    "mapping": {
        "resnet50": {
            "22": "cuda"
        },
        "mnasnet1_0": {
            "19": "cuda"
        },
        "alexnet": {
            "21": "cuda"
        },
        "mobilenet_v2": {
            "21": "cuda"
        }
    },
    "stageSummary": {
        "resnet50": {
            "cuda": {
                "infCount": 436,
                "avgInfTime": 0.0651025526020505,
                "avgTransferTime": 0.003606883210873385
            }
        },
        "mnasnet1_0": {
            "cuda": {
                "infCount": 504,
                "avgInfTime": 0.05598432393301101,
                "avgTransferTime": 0.003510780277706328
            }
        },
        "alexnet": {
            "cuda": {
                "infCount": 2324,
                "avgInfTime": 0.008181304303791355,
                "avgTransferTime": 0.0046622749039720544
            }
        },
        "mobilenet_v2": {
            "cuda": {
                "infCount": 501,
                "avgInfTime": 0.05676399590726384,
                "avgTransferTime": 0.003014081014606529
            }
        }
    },
    "samplingDuration": 30,
    "Mode": 0,
    "Device": "Orin",
    "Tegrastats": {
        "RAM": 2459.9683,
        "SWAP": 0.0,
        "CPU": {
            "Utilization": {
                "min": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0
                ],
                "max": [
                    100,
                    85,
                    100,
                    85,
                    100,
                    87
                ],
                "avg": [
                    49.8,
                    65.19,
                    44.41,
                    46.23,
                    17.11,
                    35.59
                ]
            },
            "Frequencies": {
                "min": [
                    729,
                    729,
                    729,
                    729,
                    729,
                    729
                ],
                "max": [
                    1510,
                    1510,
                    1510,
                    1510,
                    1510,
                    1511
                ],
                "avg": [
                    907.36,
                    888.22,
                    874.72,
                    862.77,
                    832.14,
                    814.97
                ]
            }
        },
        "EMC_FREQ": {
            "Utilization": {
                "min": 49,
                "max": 51,
                "avg": 50.7
            },
            "Frequencies": {
                "min": 2133,
                "max": 2133,
                "avg": 2133.0
            }
        },
        "GR3D_FREQ": {
            "Utilization": {
                "min": 90,
                "max": 99,
                "avg": 97.78
            },
            "Frequencies": {
                "min": 604,
                "max": 624,
                "avg": 607.86
            }
        },
        "NVDEC": "off",
        "NVJPG": "off",
        "NVJPG1": "off",
        "VIC": "off",
        "OFA": "off",
        "APE": 200.0,
        "CPU Temperature": 59.8312,
        "SOC 2 Temperature": 58.5429,
        "SOC 0 Temperature": 56.6085,
        "GPU Temperature": 60.1915,
        "Tjunction Temperature": 60.1926,
        "SOC 1 Temperature": 58.2723,
        "IN Power": 11532.9867,
        "GPU CPU CV Power": 4039.5717,
        "SOC Power": 2599.0067
    },
    "ModelInfo": {
        "resnet50": {
            "cuda": {
                "params": "25.56 M",
                "macs": "4.09 G",
                "flops": "8.21 G",
                "functionalLayerCount": 127
            }
        },
        "mnasnet1_0": {
            "cuda": {
                "params": "4.38 M",
                "macs": "314.42 M",
                "flops": "644.54 M",
                "functionalLayerCount": 142
            }
        },
        "alexnet": {
            "cuda": {
                "params": "61.1 M",
                "macs": "714.19 M",
                "flops": "1.43 G",
                "functionalLayerCount": 22
            }
        },
        "mobilenet_v2": {
            "cuda": {
                "params": "3.5 M",
                "macs": "300.77 M",
                "flops": "614.97 M",
                "functionalLayerCount": 142
            }
        }
    }
}
```
</details>
