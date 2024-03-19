import sys
sys.path.append("./")

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from torchvision.transforms import InterpolationMode
from pprint import pprint
from pathlib import Path
from PIL import Image
from torchvision import transforms
from copy import deepcopy

from layer_stats.utils.operations import CustomOpExecutor
from layer_stats.utils.modules import VitConvOp, VitPosOp

from layer_stats.utils.checker import get_model, get_model_list
import time
import numpy as np
import csv

# Parameters
sampling_count = 100       # Number of latency data to collect per layer 

def flatten_layers(dnn, is_vit=False, is_inception=False, is_squeezenet=False, is_google_net=False):
    layers = []
    params = []  # TODO: Make this a dictionary so that we can access the parameters by name and NOT by index
    if is_vit:
        for name, param in dnn.named_parameters():
            if name == 'encoder.pos_embedding':
                params.append(param)
            if name == "class_token":
                params.append(param)
        for child in dnn.children():
            if isinstance(child, models.vision_transformer.Encoder):
                for grandchild in child.children():
                    if isinstance(grandchild, nn.Sequential):
                        for greatgrandchild in grandchild.children():
                            layers.append(greatgrandchild)
                    else:
                        layers.append(grandchild)
            else:
                layers.append(child)
        # Insert the positional embedding layer
        layers.insert(1, VitPosOp(params[1]))
        # Edit the first layer
        layers[0] = VitConvOp(layers[0], params[0], dnn.patch_size,
                              dnn.image_size, dnn.hidden_dim)
    elif is_inception or is_google_net:
        for child in dnn.children():
            if isinstance(child, nn.Sequential):
                for grandchild in child.children():
                    layers.append(grandchild)
            elif isinstance(child, nn.ModuleList):
                intermediate = nn.Sequential(*child)
                for grandchild in intermediate.children():
                    layers.append(grandchild)
            else:
                layers.append(child)

            if layers[-1] == dnn.AuxLogits:
                # print("Found InceptionAux")
                layers.pop()
    elif is_squeezenet:
        for child in dnn.children():
            if isinstance(child, nn.Sequential):
                for grandchild in child.children():
                    layers.append(grandchild)
            elif isinstance(child, nn.ModuleList):
                intermediate = nn.Sequential(*child)
                for grandchild in intermediate.children():
                    layers.append(grandchild)
            else:
                layers.append(child)
        layers.append(nn.Flatten())
    else:
        for child in dnn.children():
            if isinstance(child, nn.Sequential):
                for grandchild in child.children():
                    layers.append(grandchild)
            elif isinstance(child, nn.ModuleList):
                intermediate = nn.Sequential(*child)
                for grandchild in intermediate.children():
                    layers.append(grandchild)
            else:
                layers.append(child)
    return layers


def count_layers(dnn, is_vit=False, is_inception=False, is_squeezenet=False, is_google_net=False):
    count = 0
    if is_vit:
        count += 1
        for child in dnn.children():
            if isinstance(child, models.vision_transformer.Encoder):
                for grandchild in child.children():
                    if isinstance(grandchild, nn.Sequential):
                        for greatgrandchild in grandchild.children():
                            count += 1
                    else:
                        count += 1
            else:
                count += 1
    elif is_inception or is_google_net:
        for child in dnn.children():
            if isinstance(child, nn.Sequential):
                for grandchild in child.children():
                    count += 1
            elif isinstance(child, nn.ModuleList):
                intermediate = nn.Sequential(*child)
                for grandchild in intermediate.children():
                    count += 1
            else:
                grandchild = child
                count += 1

            if grandchild == dnn.AuxLogits:
                # print("Found InceptionAux")
                count -= 1
    elif is_squeezenet:
        for child in dnn.children():
            if isinstance(child, nn.Sequential):
                for grandchild in child.children():
                    count += 1
            elif isinstance(child, nn.ModuleList):
                intermediate = nn.Sequential(*child)
                for grandchild in intermediate.children():
                    count += 1
            else:
                count += 1
        count += 1
    else:
        for child in dnn.children():
            if isinstance(child, nn.Sequential):
                for grandchild in child.children():
                    count += 1
            elif isinstance(child, nn.ModuleList):
                intermediate = nn.Sequential(*child)
                for grandchild in intermediate.children():
                    count += 1
            else:
                count += 1
    return count


def load_imagenet_classes(txt_file: str):
    with open(txt_file) as f:
        labels = [line.strip() for line in f.readlines()]
    idx_to_label = {idx: label for idx, label in enumerate(labels)}
    return idx_to_label

'''# Define the dataset transformations
default_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)])

swag_preprocess = transforms.Compose([
    transforms.Resize(384, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

vit_h_preprocess = transforms.Compose([
    transforms.Resize(518, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

vit_l_preprocess = transforms.Compose([
    transforms.Resize(512, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])

inception_preprocess = transforms.Compose([
    transforms.Resize(342),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]),
])'''

def getLayerwiselatency(op_executor:CustomOpExecutor,model_name:str, Device:str, sample_paths:list):

    # Declare a device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(Device)

    # Instantiate the model
    if model_name in ["regnet_y_128gf", "regnet_y_16gf", "regnet_y_32gf", "vit_b_16", "vit_h_14", "vit_l_16"]:
        model = get_model(model_name).to(device)#, weights="IMAGENET1K_SWAG_E2E_V1").to(device)
        print(f'Found model {model_name} with weights IMAGENET1K_SWAG_E2E_V1')
    else:
        try:
            model = get_model(model_name).to(device)#, weights="IMAGENET1K_V1").to(device)
            print(f'Found model {model_name} with weights IMAGENET1K_V1')
        except Exception as e:
            model = get_model(model_name).to(device)#, weights="DEFAULT").to(device)
            print(f'Trying to load model {model_name} with weights DEFAULT')

    # Set model to eval mode
    model.eval()

    # Count the layers of the model
    total_layers = count_layers(model,
                                is_vit=model_name in [
                                    "vit_b_16", "vit_b_32", "vit_h_14", "vit_l_16", "vit_l_32"],
                                is_inception=model_name in ["inception_v3"],
                                is_squeezenet=model_name in [
                                    "squeezenet1_0", "squeezenet1_1"],
                                is_google_net=model_name in ["googlenet"])

    # Flatten the layers of the model
    layers = flatten_layers(model,
                            is_vit=model_name in [
                                "vit_b_16", "vit_b_32", "vit_h_14", "vit_l_16", "vit_l_32"],
                            is_inception=model_name in ["inception_v3"],
                            is_squeezenet=model_name in [
                                "squeezenet1_0", "squeezenet1_1"],
                            is_google_net=model_name in ["googlenet"])

    '''# Print the layers of the model
    for idx, layer in enumerate(layers):
        print(f'Layer {idx}: {layer}')
    # exit(0)'''

    # Project path
    project_path = Path(__file__).resolve().parents[0]
    # print("file path: ", list(Path(__file__).resolve().parents))

    # Path to store the latency dataset
    if not os.path.exists(os.path.join(project_path, 'data','latency',model_name)):
        os.makedirs(os.path.join(project_path, 'data','latency',model_name))

    # Loop over the sample paths
    for img_path in sample_paths:
        # File name to store the collected data samples
        stats_file_name = f"{os.path.join(project_path, 'data','latency',model_name)}/{img_path.split('.')[0]}_sc_{sampling_count}.csv"
        
        # Load the image
        x = Image.open(os.path.join(project_path, 'data', 'imagenet', img_path))

        # Open a csv file and a csvwriter object to store Power stats
        stats_file = open(stats_file_name, "w", newline='')
        csvwriter = csv.writer(stats_file,delimiter='\t')
        
        # Pass the image through the layers 100 times and get an average latency 
        for idx, layer in enumerate(layers):
            try:
                if idx == 0:
                    x_sample = deepcopy(x)
                else:
                    x_sample = x.detach().clone()
                local_latency_array = []
                for j in range(sampling_count):
                    t1 = time.time()
                    x = op_executor.execute(model_name, idx, x_sample, device)
                    x = layer(x)
                    t2 = time.time()
                    local_latency_array.append(t2-t1)

                local_latency_array.insert(0,idx)
                csvwriter.writerow(local_latency_array)
            except Exception as e:
                print(f'Layer {idx} failed: {e}')
                break
    stats_file.close()
