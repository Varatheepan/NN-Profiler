import os, sys
import torch
import torch.nn as nn
import queue
import numpy as np
from itertools import permutations
from copy import deepcopy

from layer_stats.rawDataGeneration.PowerLatencySampler import flatten_layers
from layer_stats.utils.checker import get_model
from multi_network_stats.Stage import Stage

class MappingGenerator:

    # class variable
    Instances = []

    def __init__(self, device_list:list = ['cpu', 'cuda'], network_list = ['alexnet','mobilenet_v2'], NumSamples:int = 1000, CaseSamples:list = None, device_selecion_bias:list = None) -> None:
        """
            Parameters
            ----------
            compute_capability: The weightage given to the devices in the device_list according while craeting mapping samples
            Eg: [1,2] will allocate the stage with more layers to `cuda` often
        """
        # The list of devices
        self.device_list = device_list

        # The number of devices
        self.num_devices = len(device_list)

        # The list of networks in the workload
        self.network_list = network_list

        # Number of smaples to generate in the dataset
        self.NumSamples = NumSamples

        # Mapping object ~ The mapped layers in the order of device_list
        self.stageMap = [None]*self.num_devices

        # NOTE: eliminated
        # # The possible set of mapping combinations
        # self.mapPatterns = []
        # self.generate_mapping_combinations()

        # An object to keep the count of mapping cases considered
        self.mapCaseCounts = []

        # Number of samples to be generated for each mapping case
        self.mapCases = []
        self.generate_mapCases(self.NumSamples,CaseSamples)

        # A dictionary of number of layers per network
        self.numLayerDict = {'alexnet':21,'mobilenet_v2':35}

        # The total number of samples generated
        self.iterIdx = 0

        # Index of the current case
        self.caseIdx = 0

        # Probablity of selecting each device for randomization
        if device_selecion_bias == None:
            device_selecion_bias = [1]*self.num_devices
        self.deviceProb = list(np.array(device_selecion_bias)/float(sum(device_selecion_bias)))



        """
        # NOTE: eliminated
        # # Number of samples generated for each pattern
        # self.patternCount = [0]*len(self.mapPatterns)

        # # The order of the stages in the pipeline
        # self.pipelineOrder = [None]*self.num_devices
        """

        # Add the instance to the list of instances
        MappingGenerator.addInstance(self)

        # Random seeding
        np.random.seed(45)

    @classmethod
    def addInstance(cls,self):
        # Add the instance to the list of instances
        cls.Instances.append(self)

    @classmethod
    def getInstances(self):
        # get the instances of the class
        return MappingGenerator.Instances
    
    # NOTE: eliminated
    # def generate_mapping_combinations(self):
    #     # TODO: generalize for different set of devices -> implemented
    #     """
    #     Generate the possible set of permutations for each combinations a neural network can be mapped across the given devices
    #     """       
    #     # # The possile pattern of mapping pairs ~ Num_of_devices_used : Pattern 
    #     # self.mapPatterns = {1:[[0],[1]], 2:[[0,1],[1,0]]}

    #     # Generate a set of permutations for each possible combination of devices
    #     self.mapPatterns = [i for i in range(self.num_devices)]
    #     for i in range(2,self.num_devices+1):
    #         perms = list(permutations(range(self.num_devices),i))
    #         for j in perms:
    #             self.mapPatterns.append(j)

    #     # self.mapPatterns = [[0],[1],[0,1],[1,0]]

    def get_layers(self,model_name):
        """
        Get the model weights and extract the layers
        """
        
        # Instantiate the model
        if model_name in ["regnet_y_128gf", "regnet_y_16gf", "regnet_y_32gf", "vit_b_16", "vit_h_14", "vit_l_16"]:
            model = get_model(model_name)
            print(f'Found model {model_name} with pretrained weights')
        else:
            try:
                model = get_model(model_name)
                print(f'Found model {model_name} with pretrained weights')
            except Exception as e:
                model = get_model(model_name)
                print(f'Trying to load model {model_name} with weights DEFAULT')

        # Set model to eval mode
        model.eval()

        # Flatten the layers of the model
        layers = flatten_layers(model,
                                is_vit=model_name in [
                                    "vit_b_16", "vit_b_32", "vit_h_14", "vit_l_16", "vit_l_32"],
                                is_inception=model_name in ["inception_v3"],
                                is_squeezenet=model_name in [
                                    "squeezenet1_0", "squeezenet1_1"],
                                is_google_net=model_name in ["googlenet"])

        return layers
        

    def generate_mapping(self, case):
        """
        TODO: Not completed yet
        Split the layers into stages(num stages <= num_devices) and assign a device for each stage.
        """     
        
        # Number of stand alone network stages
        NumSts = case

        # Number of splitted network stages
        NumSpl = len(self.network_list) - NumSts

        model_list = deepcopy(self.network_list)

        # Object for network mappings
        mapping = {}

        # Satndalone state models
        Sts_models = np.random.choice(model_list,NumSts,replace=False)

        for model_name in Sts_models:
            mapping[model_name] = {self.numLayerDict[model_name]:np.random.choice(self.device_list,1,replace=True,p = self.deviceProb)[0]}
        
        Spl_models = [model_name for model_name in model_list if model_name not in Sts_models]

        for model_name in Spl_models:
            mapping[model_name] = {}
            devices = deepcopy(self.device_list)
            numSplits = self.num_devices-1
            if numSplits != 1:
                numSplits = np.random.randint(1,numSplits+1)
            splitPoints = np.random.choice(range(self.numLayerDict[model_name]-1),numSplits,replace=False)
            sorted(splitPoints)
            stageDevices = np.random.choice(devices, numSplits+1, replace=False,p = self.deviceProb)
            # print("splitPoints: ", splitPoints, "stageDevices: ", stageDevices)
            modelMapping = {}
            for idx,selDevice in enumerate(stageDevices):
                if idx > 0 and idx < len(splitPoints):
                    mapping[model_name][splitPoints[idx]] = selDevice
                elif idx == 0:
                    mapping[model_name][splitPoints[idx]] = selDevice
                else:
                    mapping[model_name][self.numLayerDict[model_name]] = selDevice
        return mapping
    

    def create_network_stages(self,model_name,mapping:dict):
        """
        Create stages from the mapping given.
        """
        layers = self.get_layers(model_name)

        # Number of layers in the given neural network
        num_layers = len(layers)
        
        # List of stages in the pipeline order 
        stages = []

        if len(mapping) == 1:
            layerBlock = nn.Sequential()
            for layer in layers:
                layerBlock.append(layer)
            stage = Stage(list(mapping.values())[0],layerBlock,stagePosition=3)
            stages.append(stage)

        else:
            indexes = mapping.keys()
            orderedIdx = sorted(list(indexes))
            Devices = [mapping[idx] for idx in orderedIdx]

            numMaps = len(mapping)

            for idx,layerIDX in enumerate(orderedIdx):
                if idx > 0 and idx < numMaps - 1:
                    layerBlock = nn.Sequential()
                    for layer in layers[orderedIdx[idx-1]+1:layerIDX+1]:
                        layerBlock.append(layer)
                    stage = Stage(Devices[idx],layerBlock,stagePosition=1)
                    
                elif idx == 0:
                    layerBlock = nn.Sequential()
                    for layer in layers[:layerIDX+1]:
                        layerBlock.append(layer)
                    stage = Stage(Devices[idx],layerBlock,stagePosition=1)
                else:
                    layerBlock = nn.Sequential()
                    for layer in layers[orderedIdx[idx-1]+1:]:
                        layerBlock.append(layer)
                    stage = Stage(Devices[idx],layerBlock,stagePosition=2)
                
                stages.append(stage)               

        return stages

    def generate_mapCases(self, NumSamples, CaseSamples:list = None):
        #  TODO: If num of devices more the the nummber of networks in the workload
        # Number cases to be considered
        num_cases = self.num_devices + 1

        if not CaseSamples or len(CaseSamples) < num_cases:

            W = np.array([np.exp(-0.5*val) for val in range(num_cases)])

            normalizedW = W/W.sum()

            CaseSamples = [int(val*NumSamples) for val in normalizedW]

            CaseSamples[0] = CaseSamples[0] + (NumSamples - sum(CaseSamples))

            if len(CaseSamples) < num_cases: print(F"A list of {num_cases} numbers are expected. Defaulting to a linear function.")
            
        # The array of cases with keys denoting the number of stand alone network stages and the values are the total number of generated examples.
        # The rest of the networks will be splitted among devices randomly.
        self.mapCases = [val for key,val in enumerate(CaseSamples)]
        self.mapCaseCounts = [0]*num_cases

        print("self.mapCases: ",self.mapCases)


    def iter(self):
        """
        Create a new mapping sample
        """

        if self.iterIdx == self.NumSamples:
            return False
        
        # TODO:Generate a random image and preporcess according to the op executers

        if self.mapCaseCounts[self.caseIdx] != self.mapCases[self.caseIdx]:
            Mappings = self.generate_mapping(self.caseIdx)
        else:
            self.caseIdx +=1
            Mappings = self.generate_mapping(self.caseIdx)

        # TODO: implement a memory overflow management
        # MappingSucess = False
        # while not MappingSucess:
            

        # for model_name,mapping in Mappings.items():
        #     # try:
        #     stages = self.create_network_stages(model_name,mapping)
        
        self.mapCaseCounts[self.caseIdx] += 1
        self.iterIdx += 1
        print("self.iterIdx: ",self.iterIdx,"  Mappings: ", Mappings)
        print(f"mapCaseCounts: {self.mapCaseCounts}")

        # return stages


            


        