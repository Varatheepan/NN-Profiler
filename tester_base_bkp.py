
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

from operations import CustomOpExecutor
from modules import VitConvOp, VitPosOp


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

# Define the dataset transformations
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
])

# Get the list of available models
model_list = models.list_models(module=models)

# # Log the list of available models
# pprint(model_list)

# Set a model for testing
model_family = 'alexnet'
model_name = 'alexnet'

# Declare a device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Instantiate the model
if model_name in ["regnet_y_128gf", "regnet_y_16gf", "regnet_y_32gf", "vit_b_16", "vit_h_14", "vit_l_16"]:
    model = models.get_model(model_name, weights="IMAGENET1K_SWAG_E2E_V1").to(device)
    print(f'Found model {model_name} with weights IMAGENET1K_SWAG_E2E_V1')
else:
    try:
        model = models.get_model(model_name, weights="IMAGENET1K_V1").to(device)
        print(f'Found model {model_name} with weights IMAGENET1K_V1')
    except Exception as e:
        model = models.get_model(model_name, weights="DEFAULT").to(device)
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

# Print the layers of the model
for idx, layer in enumerate(layers):
    print(f'Layer {idx}: {layer}')
# exit(0)

# Project path
project_path = Path(__file__).resolve().parents[0]

# load the imagenet classes
idx_to_label = load_imagenet_classes(txt_file=os.path.join(project_path, 'data', 'imagenet', 'imagenet.txt'))

# Define sample paths 
sample_paths = ['n01531178_goldfinch.jpg', 'n01534433_junco.jpg', 'n01667114_mud_turtle.jpg']

# Loop over the sample paths
for img_path in sample_paths:

    # Load the image
    x = Image.open(os.path.join(project_path, 'data', 'imagenet', img_path))
    # Copy the image
    x_hat = deepcopy(x)

    # Register a custom operation executor
    op_executor = CustomOpExecutor(preprocess=True, verbose=True)

    # Register DNN families
    op_executor.register_family_operation("alexnet", CustomOpExecutor._alexnet_op)
    op_executor.register_family_operation("mnasnet", CustomOpExecutor._mnasnet_op)

    # Registering preprocessing step
    model_list = models.list_models(module=models)
    for m_n in model_list:
        if m_n in ["inception_v3"]:
            op_executor.register_preprocess(m_n, inception_preprocess)
        elif m_n in ["regnet_y_128gf", "regnet_y_16gf", "regnet_y_32gf", "vit_b_16"]:
            op_executor.register_preprocess(m_n, swag_preprocess)
        elif m_n in ["vit_h_14"]:
            op_executor.register_preprocess(m_n, vit_h_preprocess)
        elif m_n in ["vit_l_16"]:
            op_executor.register_preprocess(m_n, vit_l_preprocess)
        else:
            op_executor.register_preprocess(m_n, default_preprocess)

    # Registering model operations based on family
    op_executor.register_operation("alexnet", "alexnet", 14)
    op_executor.register_operation("mnasnet0_75", "mnasnet", 17)

    # Finalize registration
    op_executor.finalize_registration()

    # Pass the image through the layers
    for idx, layer in enumerate(layers):
        try:
            x = op_executor.execute(model_name, idx, x, device)
            x = layer(x)
        except Exception as e:
            print(f'Layer {idx} failed: {e}')
            break
    print("x.shape: ", x.shape)

    # Prepare the image for the model
    if model_name in ["regnet_y_128gf", "regnet_y_16gf", "regnet_y_32gf", "vit_b_16"]:
        x_hat = swag_preprocess(x_hat)
    elif model_name in ["vit_h_14"]:
        x_hat = vit_h_preprocess(x_hat)
    elif model_name in ["vit_l_16"]:
        x_hat = vit_l_preprocess(x_hat)
    elif model_name in ["inception_v3"]:
        x_hat = inception_preprocess(x_hat)
    else:
        x_hat = default_preprocess(x_hat)
    # Add batch dimension and move to device
    x_hat = x_hat.unsqueeze(0).to(device)
    # Pass the image through the model
    x_hat = model(x_hat)
    
    # Post-process the output
    x, x_hat = x.detach().cpu().squeeze(0), x_hat.detach().cpu().squeeze(0)

    # Apply softmax to the output to get probabilities
    x, x_hat = F.softmax(x, dim=0), F.softmax(x_hat, dim=0)

    # get top 5 predicted class indices
    top5_idx, top5_idx_hat = torch.topk(x, k=5).indices.tolist(), torch.topk(x_hat, k=5).indices.tolist()

    # get top 5 predicted class labels
    top5_label, top5_label_hat = [idx_to_label[idx] for idx in top5_idx], [idx_to_label[idx] for idx in top5_idx_hat]

    # print the sample path
    print(f"Sample: {img_path}")

    # print the top 5 predicted class labels
    for rank, (idx, label, idx_hat, label_hat) in enumerate(zip(top5_idx, top5_label, top5_idx_hat, top5_label_hat), start=1):
        print(f"\tTop {rank}: {label:20s} {x[idx]:.5f} {label_hat:20s} {x_hat[idx_hat]:.5f}")

        assert label == label_hat, f"Top {rank} labels do not match: {label} != {label_hat}"
        assert torch.isclose(x[idx], x_hat[idx_hat], atol=1e-4), f"Top {rank} probabilities do not match: {x[idx]} != {x_hat[idx_hat]}"

    # Leave a blank line
    print()
