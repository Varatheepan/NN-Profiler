# from __future__ import annotations
import os, sys
import torch
import torch.nn as nn
import queue
from copy import deepcopy
import time
import gc

# This class will handle a subset of layers of a network to run on a device
class Stage:
    # def __init__(self,Device:torch.DeviceObjType,Layers:nn.Module, OutPutQueue: queue.Queue, InOutBuf:int = -1) -> None:
    def __init__(self,Device:torch.DeviceObjType,Layers:nn.Module, InOutBuf:int = -1, stagePosition:int = 0) -> None:
        
        # A subset of layers from a network 
        self.layerSet = Layers      

        # The device to run the stage
        self.device = Device        

        # Assign the layerStage to the device
        self.status = self.assignToDevice()
        
        # A queue to buffer the inputs to the stage
        self.InputQueue = None

        # # A queue to buffer the inputs to the stage
        # self.OutputQueue = OutPutQueue    

        # If the stage is at the start(0)/middle(1)/end(2) of the network pipleine OR is a stand alone(3) 
        self.stagePos = stagePosition       
        
        if stagePosition == 2 or stagePosition == 3:
            self.OutputQueue = queue.Queue(-1)

        # Number of inferences performed by the stage
        self.infCount = 0

        # Whether to to keep the stage active
        self.stageActive = False

        # Whether the stage is running active inference
        self.stageRunning = False

        # Time taken for inferences in the stage
        self.inferDurations = []

        # Time taken for data transfer to the compute component
        self.tranferDurations = []

        # Inuput size of the stage
        self.inputSize = None

    def forward(self, x, NextStage= None):  #NextStage: Stage = None):

        """Forward the next input present in the input queue and store the output in the output queue.
        
        Parameters
        ----------
        NextStage:
            Next stage in the pipeline of the NN
        """

        # # set the status of the stage to running
        # self.stageRunning = True

        # if self.stagePos == 0 or self.stagePos == 
        
        # # Retrieve input from the queue
        # x = deepcopy(self.InputQueue)

        # self.InputQueue = None

        # set input size
        if self.inputSize == None:
            self.inputSize = x.shape

        # For the first stage set the device
        if (self.stagePos == 0 or self.stagePos == 3) and x.device.type != self.device.type:
            x = x.to(self.device)
        # Run inference 

        t1 = time.time()

        with torch.no_grad():
            x = self.layerSet(x)

        t2 = time.time()

        self.inferDurations.append(t2-t1)

        # # Store the output to the output queue
        # self.OutputQueue.put(x)

        # If there are stages after this, store the output to the input queue of the next stage
        if self.stagePos == 0 or self.stagePos == 1:
            if NextStage == None:
                print("Next stage should be passed if the stage is at the start or middile of the pipeline")
                return ValueError

            # Set the output to the next stage's device
            if self.device.type != NextStage.device.type:
                x = x.to(NextStage.device)
            NextStage.putToQueue(x)
        
        # Put the data to cpu if this stage is at the end
        else:
            if self.device.type != 'cpu':
                x = x.to('cpu')
            self.OutputQueue.put(x)
        
        return True

    def forwardSeq(self,x):  #NextStage: Stage = None):

        """Forward the next input present in the input queue and store the output in the output queue.
        
        Parameters
        ----------
        NextStage:
            Next stage in the pipeline of the NN
        """
        # set input size
        if self.inputSize == None:
            self.inputSize = x.shape

        t3 = time.time()

        # Set the device
        if x.device.type != self.device.type:
            x = x.to(self.device)

        t1 = time.time()
        
        # Run inference 
        with torch.no_grad():
            x = self.layerSet(x)

        t2 = time.time()

        self.inferDurations.append(t2-t1)
        self.tranferDurations.append(t1-t3)
        self.infCount += 1

        return x

    def assignToDevice(self):
        """ Assign the stage to the device."""
        try:
            self.layerSet.to(self.device)
            return True
        except RuntimeError:
            print("Error in assigning the stage to the device")
            return False


    def putToQueue(self,x): 
        # Push x to the FIFO queue
        self.InputQueue = x

    # TODO: add the next stage as a class parameter
    # def run(self,PrevStage = None,NextStage= None):  #: Stage = None):
    def run(self,NextStage= None, image = None):  #: Stage = None):
        self.stageActive = True
        
        # set the status of the stage as Running
        self.stageRunning = True

        # Run the forward until all the images in the input queue are processed

        # If a prevoius stage exist, wait until it finishes
        # if PrevStage != None:
        if self.stagePos == 3:
            while (self.stageActive):
                # Input image for first stage
                # self.InputQueue = image 
                result = self.forward(image)
                self.infCount += 1
        elif self.stagePos == 0:
            while (self.stageActive):
                if NextStage.InputQueue == None:
                    # self.InputQueue = image
                    result = self.forward(image,NextStage)
                    self.infCount += 1
                else:
                    time.sleep(0.005)
        elif self.stagePos == 1:
            while (self.stageActive):
                if self.InputQueue != None and NextStage.InputQueue == None:
                    # Retrieve input from the queue
                    x = self.InputQueue.detach().clone()
                    self.InputQueue = None
                    result = self.forward(x,NextStage)
                    self.infCount += 1
                else:
                    time.sleep(0.005)

        else:
            while (self.stageActive):
                if self.InputQueue != None:
                    # Retrieve input from the queue
                    x = self.InputQueue.detach().clone()
                    self.InputQueue = None
                    result = self.forward(x)
                    self.infCount += 1
                else:
                    time.sleep(0.005)
            # print(f"self.infCount: {self.infCount}, PrevStage.infCount: {PrevStage.infCount}")
        # else:
        #     while not self.InputQueue.empty():
        #         self.forward(NextStage)
        #         self.infCount += 1
        
        # set the status of the stage as finished
        self.stageRunning = False

        # print("Stage: ", self.stagePos,", number of inference: ", self.infCount)
    
    def activateStage(self):
        self.stageActive = True 

    def deactivateStage(self):
        self.stageActive = False
    
    def isStageQueueEmpty(self):
        return self.InputQueue.empty()
    
    def removeStage(self):
        self.layerSet.cpu()
        del self.layerSet
        gc.collect()


