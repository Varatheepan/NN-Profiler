import os, sys
import torch
import torch.nn as nn
import queue
import numpy as np
from itertools import permutations

from layer_stats.rawDataGeneration.PowerLatencySampler import flatten_layers
from layer_stats.utils.checker import get_model
from multi_network_stats.Stage import Stage

class MappingGenerator:

    # class variable
    Instances = []

    def __init__(self,model_name:str, device_list:list = ['cpu', 'cuda'], compute_capability_bias:list = None) -> None:
        """
            Parameters
            ----------
            compute_capability: The weightage given to the devices in the device_list according while craeting mapping samples
            Eg: [1,2] will allocate the stage with more layers to `cuda` often
        """
        self.model_name = model_name

        # The list of devices
        self.device_list = device_list

        # The number of devices
        self.num_devices = len(device_list)

        # Mapping object ~ The mapped layers in the order of device_list
        self.stageMap = [None]*self.num_devices

        # The possible set of mapping combinations
        self.mapPatterns = []
        self.generate_mapping_combinations()

        # An object to keep the count of mapping cases considered
        self.mapCases = {}
        self.generate_mapCases()

        # Number of samples generated for each pattern
        self.patternCount = [0]*len(self.mapPatterns)

        # The order of the stages in the pipeline
        self.pipelineOrder = [None]*self.num_devices

        # initiatlize an empty list of layers
        self.layers = []

        # Get the layers for the network
        self.get_layers()

        # Add the instance to the list of instances
        MappingGenerator.addInstance(self)

    @classmethod
    def addInstance(cls,self):
        # Add the instance to the list of instances
        cls.Instances.append(self)

    @classmethod
    def getInstances(self):
        # get the instances of the class
        return MappingGenerator.Instances
    
    def generate_mapping_combinations(self):
        # TODO: generalize for different set of devices -> implemented
        """
        Generate the possible set of permutations for each combinations a neural network can be mapped across the given devices
        """       
        # # The possile pattern of mapping pairs ~ Num_of_devices_used : Pattern 
        # self.mapPatterns = {1:[[0],[1]], 2:[[0,1],[1,0]]}

        # Generate a set of permutations for each possible combination of devices
        self.mapPatterns = [i for i in range(self.num_devices)]
        for i in range(2,self.num_devices+1):
            perms = list(permutations(range(self.num_devices),i))
            for j in perms:
                self.mapPatterns.append(j)

        # self.mapPatterns = [[0],[1],[0,1],[1,0]]



    def get_layers(self):
        """
        Get the model weights and extract the layers
        """
        
        # Instantiate the model
        if self.model_name in ["regnet_y_128gf", "regnet_y_16gf", "regnet_y_32gf", "vit_b_16", "vit_h_14", "vit_l_16"]:
            model = get_model(self.model_name)
            print(f'Found model {self.model_name} with pretrained weights')
        else:
            try:
                model = get_model(self.model_name)
                print(f'Found model {self.model_name} with pretrained weights')
            except Exception as e:
                model = get_model(self.model_name)
                print(f'Trying to load model {self.model_name} with weights DEFAULT')

        # Set model to eval mode
        model.eval()

        # Flatten the layers of the model
        self.layers = flatten_layers(model,
                                is_vit=self.model_name in [
                                    "vit_b_16", "vit_b_32", "vit_h_14", "vit_l_16", "vit_l_32"],
                                is_inception=self.model_name in ["inception_v3"],
                                is_squeezenet=self.model_name in [
                                    "squeezenet1_0", "squeezenet1_1"],
                                is_google_net=self.model_name in ["googlenet"])
        

    def generate_mapping(self, force_SAS:bool = False):
        """
        TODO: Not completed yet
        Split the layers into stages(num stages <= num_devices) and assign a device for each stage.
        """
        # Number of layers in the given neural network
        num_layers = len(self.layers)

        
        
        # Generate layer splits
        splitPositions = np.random.choice(range(num_layers), replace=False) 
    

    def create_network_stages(self,mapping:dict):
        """
        Create stages from the mapping given.
        """
        
        # Number of layers in the given neural network
        num_layers = len(self.layers)
        
        # List of stages in the pipeline order 
        stages = []

        if len(mapping) == 1:
            layerBlock = nn.Sequential(self.layers)
            stage = Stage(mapping.values()[0],layerBlock,stagePosition=3)
            stages.append(stage)

        else:
            indexes = mapping.keys()
            orderedIdx = sorted(list(indexes))
            Devices = [mapping[idx] for idx in orderedIdx]

            numMaps = len(mapping)

            for idx,layerIDX in enumerate(orderedIdx):
                if idx > 0 and idx < numMaps - 1:
                    # layerIDX > 0 and layerIDX < num_layers:
                    layerBlock = nn.Sequential()
                    for layer in self.layers[orderedIdx[idx-1]+1:layerIDX+1]:
                        layerBlock.append(layer)
                    stage = Stage(Devices[idx],layerBlock,stagePosition=1)
                    
                elif idx == 0:
                    layerBlock = nn.Sequential()
                    for layer in self.layers[:layerIDX+1]:
                        layerBlock.append(layer)
                    # layerBlock = nn.Sequential(self.layers[:layerIDX+1])
                    stage = Stage(Devices[idx],layerBlock,stagePosition=1)
                else:
                    layerBlock = nn.Sequential()
                    for layer in self.layers[orderedIdx[idx-1]+1:]:
                        layerBlock.append(layer)
                    # layerBlock = nn.Sequential(self.layers[layerIDX:])
                    stage = Stage(Devices[idx],layerBlock,stagePosition=2)
                
                stages.append(stage)
            
            # for layer in self.layers:
            #     print(layer)
            # print('\n')
            # for stage in stages:
            #     print("Mapping: ")
            #     print(stage.device)
            #     print(stage.layerSet,'\n')
                

        return stages

    def generate_mapCases(self):
        """
        Generate an object to keep track of the cases considered for the mapping. 
        """

        num_cases = self.num_devices + 1

        # The array of cases with keys denoting the number of stand alone network stages. The rest of the networks will be splitted among devices randomly.
        self.mapCases = {key:0 for key in range(num_cases)}




            


        