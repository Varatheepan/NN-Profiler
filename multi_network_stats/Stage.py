# from __future__ import annotations
import os, sys
import torch
import torch.nn as nn
import queue

# This class will handle a subset of layers of a network to run on a device
class Stage:
    # def __init__(self,Device:torch.DeviceObjType,Layers:nn.Module, OutPutQueue: queue.Queue, InOutBuf:int = -1) -> None:
    def __init__(self,Device:torch.DeviceObjType,Layers:nn.Module, InOutBuf:int = -1, stagePosition:int = 0) -> None:
        
        # A subset of layers from a network 
        self.layerSet = Layers      

        # The device to run the stage
        self.device = Device        

        # Assign the layerStage to the device
        self.assignToDevice()
        
        # A queue to buffer the inputs to the stage
        self.InputQueue = queue.Queue(InOutBuf)  

        # # A queue to buffer the inputs to the stage
        # self.OutputQueue = OutPutQueue    

        # If the stage is at the start(0)/middle(1)/end(2) of the network pipleine OR is a stand alone(3) 
        self.stagePos = stagePosition       
        
        if stagePosition == 2 or stagePosition == 3:
            self.OutputQueue = queue.Queue()

        # Number of inferences performed by the stage
        self.infCount = 0

        # WHether to to keep the stage active
        self.stageActive = False

    def forward(self, NextStage= None):  #NextStage: Stage = None):

        """Forward the next input present in the input queue and store the output in the output queue.
        
        Parameters
        ----------
        NextStage:
            Next stage in the pipeline of the NN
        """
        
        # Retrieve input from the queue
        x = self.InputQueue.get()

        # For the first stage set the device
        if (self.stagePos == 0 or self.stagePos == 3) and x.device.type != self.device.type:
            x = x.to(self.device)
        # Run inference 
        x = self.layerSet(x)

        # # Store the output to the output queue
        # self.OutputQueue.put(x)

        # If there are stages after this, store the output to the input queue of the next stage
        if self.stagePos == 0 or self.stagePos == 1:
            if NextStage == None:
                print("Next stage is should be passed if the stage is at the start or middile of the pipeline")
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


    def assignToDevice(self):
        """ Assign the stage to the device."""
        self.layerSet.to(self.device)


    def putToQueue(self,x): 
        # Push x to the FIFO queue
        self.InputQueue.put(x)

    # TODO: add the next stage as a class parameter
    def run(self,NextStage= None):  #: Stage = None):
        # self.stageActive = True
        while not self.InputQueue.empty():
            self.forward(NextStage)
            self.infCount += 1
        # print("number of inference: ", self.infCount)
    
    def activateStage(self):
        self.stageActive = True 

    def deactivateStage(self):
        self.stageActive = False
    
    def isStageQueueEmpty(self):
        return self.InputQueue.empty()