import json
import os, sys
import torch
import torch.nn as nn
import queue
import numpy as np
from itertools import permutations
from copy import deepcopy
from pathlib import Path
import collections

from layer_stats.rawDataGeneration.PowerLatencySampler import flatten_layers, count_layers
from layer_stats.utils.checker import get_model
from multi_network_stats.Stage import Stage
from multi_network_stats.utils.operations import CustomOpExecutor, database_spawn

projectPath = Path(__file__).resolve().parents[0]
print(projectPath)

class MappingGenerator:

    # class variable
    Instances = []

    def __init__(self, jetsonDevice: str = "TX2", compComponentList:list = ['cpu', 'cuda'], network_list = ['alexnet','mobilenet_v2'], NumSamples:int = 1000, CaseSamples:list = None, device_prioriy:list = None, seed: int = 45) -> None:
        """
            Parameters
            ----------
            device_prioriy: The weightage given to the devices in the compComponentList according while craeting mapping samples
            Eg: [1,2] will allocate the stage with more layers to `cuda` often
        """
        # The list of devices
        self.compComponentList = compComponentList

        # The number of devices
        self.num_devices = len(compComponentList)

        # The list of networks in the workload
        self.network_list = network_list

        # Number of smaples to generate in the dataset
        self.NumSamples = NumSamples

        # Mapping object ~ The mapped layers in the order of compComponentList
        self.stageMap = [None]*self.num_devices

        self.jetsonDevice = jetsonDevice

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
        self.numLayerDict = {}

        # Download and store the model weights and update numLayerDict
        self.download_model_weights()

        # The total number of samples generated
        self.iterIdx = 0

        # Index of the current case
        self.caseIdx = 0

        # op_excuter for the intermediate (pre-)processes
        self.op_executor = database_spawn(preprocess=True,jetson_device=jetsonDevice)

        # Probablity of selecting each device for randomization
        if device_prioriy == None:
            device_prioriy = [1]*self.num_devices
        self.deviceProb = list(np.array(device_prioriy)/float(sum(device_prioriy)))



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
        np.random.seed(seed)

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
        
        # # Instantiate the model
        # if model_name in ["regnet_y_128gf", "regnet_y_16gf", "regnet_y_32gf", "vit_b_16", "vit_h_14", "vit_l_16"]:
        #     model = get_model(model_name)
        #     print(f'Found model {model_name} with pretrained weights')
        # else:
        #     try:
        #         model = get_model(model_name)
        #         print(f'Found model {model_name} with pretrained weights')
        #     except Exception as e:
        #         model = get_model(model_name)
        #         print(f'Trying to load model {model_name} with weights DEFAULT')

        modelCkptPath = os.path.join(projectPath,"ModelCheckPoints",model_name,model_name+".ckpt")

        model = torch.load(modelCkptPath)

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
        TODO: Handle if only one device is present.  --implemented
        Split the layers into stages(num stages <= num_devices) and assign a device for each stage.
        """     
        
        # Number of stand alone network stages
        NumSts = case

        # Number of splitted network stages
        NumSpl = len(self.network_list) - NumSts

        model_list = deepcopy(self.network_list)

        # Object for network mappings
        mapping = {}

        if self.num_devices == 0:
            print("Device list is empty!")
        elif self.num_devices == 1:
            for model_name in model_list:
                mapping[model_name] = {self.numLayerDict[model_name]:self.compComponentList[0]}

        else:
            # Models to run in a stand-alone stage
            Sts_models = np.random.choice(model_list,NumSts,replace=False)

            # Random selection of devices for each stand-alone stage
            for model_name in Sts_models:
                mapping[model_name] = {self.numLayerDict[model_name]:np.random.choice(self.compComponentList,1,replace=True,p = self.deviceProb)[0]}
            
            # List of models to be splitted among some devices
            Spl_models = [model_name for model_name in model_list if model_name not in Sts_models]

            for model_name in Spl_models:
                mapping[model_name] = {}
                devices = deepcopy(self.compComponentList)
                
                # Maximum possible number of model breaking points
                numSplits = self.num_devices-1

                # Random selection of number of breaks
                if numSplits != 1:
                    numSplits = np.random.randint(1,numSplits+1)
                
                # Random selection of layer layer indexes to split
                splitPoints = np.random.choice(range(self.numLayerDict[model_name]-1),numSplits,replace=False)

                # Sorting the selected layer indexes
                splitPoints = sorted(splitPoints)

                # Random seletcion of devices for each split set of layers
                stageDevices = np.random.choice(devices, numSplits+1, replace=False,p = self.deviceProb)
                # print("splitPoints: ", splitPoints, "stageDevices: ", stageDevices)

                # Formatting the mapping 
                for idx,selDevice in enumerate(stageDevices):

                    # Pair of layer number and a device. 
                    # A stage associated with a pair will consist of of the layers after the prevois set upto the current layer index(inclusive).
                    if idx >= 0 and idx < len(splitPoints):
                        mapping[model_name][splitPoints[idx]] = selDevice
                    # elif idx == 0:
                    #     mapping[model_name][splitPoints[idx]] = selDevice
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

        # Network with a stand-alone stage
        if len(mapping) == 1:

            # # Adding all the layers to a sequencial block
            # layerBlock = collections.OrderedDict()#nn.ModuleDict()
            # for layer in layers:
            #     layerBlock['0'] = layer

            modelCkptPath = os.path.join(projectPath,"ModelCheckPoints",model_name,model_name+".ckpt")
            model = torch.load(modelCkptPath)
            
            # Creating a stage 
            # stage = Stage(torch.device(list(mapping.values())[0]),nn.Sequential(model),stagePosition=3)
            layer_ID = 0
            layerBlock = collections.OrderedDict()
            for layer in layers:
                op_name = model_name+"_"+str(layer_ID)
                if op_name in self.op_executor.operations and layer_ID != 0:
                    layerBlock[str(layer_ID)+"_op"]= self.op_executor.operations[op_name][0]
                layer_ID += 1
                layerBlock[str(layer_ID)] = layer
            stage = Stage(torch.device(list(mapping.values())[0]),nn.Sequential(layerBlock),stagePosition=3)
            stages.append(stage)

        # For networks split among devices
        else:
            # The layer indexes
            indexes = mapping.keys()

            # Sorting the layer indexes
            orderedIdx = sorted(list(indexes))

            # Sorting the devices in the same order as layers
            Devices = [mapping[idx] for idx in orderedIdx]

            numMaps = len(mapping)

            layer_ID = 0

            # Creating stages from each set of layers
            for idx,layerIDX in enumerate(orderedIdx):
                
                # An intermediate stage: for more than two stages
                if idx > 0 and idx < numMaps - 1:
                    layerBlock = collections.OrderedDict()#nn.ModuleDict()
                    for layer in layers[orderedIdx[idx-1]+1:layerIDX+1]:
                        op_name = model_name+"_"+str(layer_ID)
                        if op_name in self.op_executor.operations:
                            layerBlock[str(layer_ID)+"_op"]= self.op_executor.operations[op_name][0]
                        layer_ID += 1
                        layerBlock[str(layer_ID)] = layer
                    stage = Stage(torch.device(Devices[idx]),nn.Sequential(layerBlock),stagePosition=1)

                # First stage of a network pipeline 
                elif idx == 0:
                    layerBlock = collections.OrderedDict()#nn.ModuleDict()
                    for layer in layers[:layerIDX+1]:
                        op_name = model_name+"_"+str(layer_ID)
                        if op_name in self.op_executor.operations and layer_ID != 0:
                            layerBlock[str(layer_ID)+"_op"]= self.op_executor.operations[op_name][0]
                        layer_ID += 1
                        layerBlock[str(layer_ID)] = layer
                    stage = Stage(torch.device(Devices[idx]),nn.Sequential(layerBlock),stagePosition=0)
                
                # Last stage of network pipeline
                else:
                    layerBlock = collections.OrderedDict()#nn.ModuleDict()
                    for layer in layers[orderedIdx[idx-1]+1:]:
                        op_name = model_name+"_"+str(layer_ID)
                        if op_name in self.op_executor.operations:
                            layerBlock[str(layer_ID)+"_op"]= self.op_executor.operations[op_name][0]
                        layer_ID += 1
                        layerBlock[str(layer_ID)] = layer
                    stage = Stage(torch.device(Devices[idx]),nn.Sequential(layerBlock),stagePosition=2)
                
                stages.append(stage)               

        return stages

    def generate_mapCases(self, NumSamples, CaseSamples:list = None):

        # Number cases to be considered
        num_cases = len(self.network_list) + 1

        if self.num_devices == 1:
            self.mapCases = [NumSamples]
            self.mapCaseCounts = [0]

        else:
            if not CaseSamples or len(CaseSamples) < num_cases:
            
                # A function to determine the number of sample to generate per case
                # W = np.array([np.exp(-0.5*val) for val in range(num_cases)])
                W = np.array([1.0/num_cases for val in range(num_cases)])

                normalizedW = W/W.sum()

                # Mapping samples to generate per case
                CaseSamples = [int(val*NumSamples) for val in normalizedW]

                # # Assign remaining sample to case 0 to meet the total num of samples
                # CaseSamples[0] = CaseSamples[0] + (NumSamples - sum(CaseSamples))
                
                # Assign remaining samples across the cases
                for i1 in range(NumSamples % num_cases):
                    CaseSamples[i1] += 1
                
                if len(CaseSamples) < num_cases: print(F"A list of {num_cases} numbers are expected. Defaulting to a linear function.")
                
            # The array of cases with keys denoting the number of stand alone network stages and the values are the total number of generated examples.
            # The rest of the networks will be splitted among devices randomly.
            self.mapCases = [val for key,val in enumerate(CaseSamples)]

            self.mapCaseCounts = [0]*num_cases

        print("self.mapCases: ",self.mapCases)

    def download_model_weights(self):

        """
        Download the model checkpoints if not present alraedy, and update the number of layers
        """

        checkpointsPath = os.path.join(projectPath,"ModelCheckPoints")

        NumOfLayersJsonFile = os.path.join(projectPath,"numLayers.json")

        NumOfLayersJson = {}

        if not os.path.exists(checkpointsPath):
                os.mkdir(checkpointsPath)
            
        for model_name in self.network_list:
            
            modelCktPath = os.path.join(checkpointsPath,model_name)

            if not os.path.exists(modelCktPath):
                os.mkdir(modelCktPath)

                model = get_model(model_name,jetson_device=self.jetsonDevice)

                torch.save(model,os.path.join(modelCktPath,model_name+".ckpt"))

                numLayers = count_layers(model)

                NumOfLayersJson[model_name] = numLayers

        JsonObject = {}

        if os.path.exists(NumOfLayersJsonFile):
            with open(NumOfLayersJsonFile,"r")  as F:
                line = F.readline()
                JsonObject = json.loads(line)
        
        for key, value in NumOfLayersJson.items():
            if key not in JsonObject:
                JsonObject[key] = value
        
        with open(NumOfLayersJsonFile,"w") as F:
            json.dump(JsonObject,F)

        self.numLayerDict = JsonObject

            

    def iter(self, Mappings:dict = None):
        """
        Create a new mapping sample
        """

        if self.iterIdx == self.NumSamples:
            return False, False
        
        # TODO:Generate a random image and preporcess according to the op executers
        
        MapValid = False

        if Mappings != None:
            MapValid = self.validateMapping(Mappings)

        if MapValid == False:
            if self.mapCaseCounts[self.caseIdx] != self.mapCases[self.caseIdx]:
                Mappings = self.generate_mapping(self.caseIdx)
            else:
                while self.mapCaseCounts[self.caseIdx] == self.mapCases[self.caseIdx]:
                    self.caseIdx +=1
                Mappings = self.generate_mapping(self.caseIdx)

        # TODO: implement a memory overflow management
        # MappingSucess = False
        # while not MappingSucess:
            
        stageDict = {}
        for model_name,mapping in Mappings.items():
            # try:
            stages = self.create_network_stages(model_name,mapping)
            stageDict[model_name] = stages

        self.mapCaseCounts[self.caseIdx] += 1
        self.iterIdx += 1
        # print("self.iterIdx: ",self.iterIdx,"  Mappings: ", Mappings)
        print(f"MapCaseCounts: {self.mapCaseCounts}")

        return stageDict,Mappings

    def validateMapping(self,Mappings):
        numMappings = len(Mappings)

        if numMappings != len(self.network_list):
            print("Not all the networks are mapped. Invalid mapping!")
            return False
            
        for model_name in Mappings.keys():
            if model_name not in self.network_list:
                print(f"IModel `{model_name}` not in network list. Invalid mapping")
                return False
            
            if len(Mappings[model_name]) == 0:
                print(f"Mapping for `{model_name}`is Invalid!")
                return False
            
            for device in Mappings[model_name].values():
                if device not in self.compComponentList:
                    print(f"Device `{device}` not in device list. `{model_name}` mapping is Invalid!")
                    return False
            
            for layerIdx in Mappings[model_name].keys():
                if layerIdx> self.numLayerDict[model_name]:
                    print(f"Layer index exceeded. Mapping for `{model_name}`is Invalid!")
                    return False
        return True




            
        