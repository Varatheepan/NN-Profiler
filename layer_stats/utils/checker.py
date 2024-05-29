
import torch
import torchvision.models as models

def banned_mods():
    """https://pytorch.org/vision/stable/models.html
    """
    bugged = ["inception_v3", "googlenet", "maxvit_t", 
              "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", 
              "shufflenet_v2_x1_5", "shufflenet_v2_x2_0", 
              "swin_b", "swin_s", "swin_t", "swin_v2_b", 
              "swin_v2_s", "swin_v2_t", "vit_b_16", "vit_b_32", 
              "vit_h_14", "vit_l_16", "vit_l_32"]
    timely = ["convnext_base", "convnext_large", "convnext_small", 
              "convnext_tiny", "densenet121, densenet161", "densenet201", 
              "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
              "efficientnet_v2_l", "efficientnet_v2_m", "mobilenet_v3_large", 
              "mobilenet_v3_small", "regnet_y_128gf", "regnet_y_16gf", "regnet_y_32gf",
              "vgg16", "vgg16_bn"]
    return bugged + timely

def get_model_list(jetson_device: str = "Tx2"):

    if jetson_device in ["Xavier", "Orin"]:
        Available_Models = models.list_models(module=models)

        banned_models = banned_mods()

        return [model for model in Available_Models if model not in banned_models]

    else:# jetson_device == "Tx2":
        return [
            "alexnet",
            "densenet121",
            "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
            "mnasnet0_5", "mnasnet1_0",
            "mobilenet_v2",
            "mobilenet_v3_large", "mobilenet_v3_small",
            "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_x_3_2gf", "regnet_x_800mf", "regnet_x_400mf",
            "resnet18", "resnet34", "resnet50", "resnet101",
            "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
            "squeezenet1_0"

            ####### SUPPORTED BUT NO CHECKPOINT IN TORCHVISION 0.11.0 ###########
            # "mnasnet0_75", "mnasnet1_3",

            ########################## BANNED ##########################
            # "densenet161", "densenet169", "densenet201",
            # "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
            # "googlenet",
            # "inception_v3",
            # "regnet_y_3_2gf", "regnet_x_1_6gf", "regnet_x_8gf", "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf", "regnet_x_16gf", "regnet_x_32gf",
            # "resnet152",
            # "resnext50_32x4d", "resnext101_32x8d",
            # "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
            # "squeezenet1_1",
            # "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"
            # "wide_resnet50_2", "wide_resnet101_2"
        ]


# def get_model(model_name: str):
#     return getattr(models, model_name)(weights=None)

def get_model(model_name: str, jetson_device: str = "Tx2"):

    if jetson_device in ["Xavier", "Orin"]:
        return getattr(models, model_name)(weights=None)
    
    else:# jetson_device == "Tx2":
        if model_name == "alexnet":
            return models.alexnet(pretrained=True)
        elif model_name == "densenet121":
            return models.densenet121(pretrained=True)
        elif model_name == "efficientnet_b0":
            return models.efficientnet_b0(pretrained=True)
        elif model_name == "efficientnet_b1":
            return models.efficientnet_b1(pretrained=True)
        elif model_name == "efficientnet_b2":
            return models.efficientnet_b2(pretrained=True)
        elif model_name == "mnasnet0_5":
            return models.mnasnet0_5(pretrained=True)
        elif model_name == "mnasnet1_0":
            return models.mnasnet1_0(pretrained=True)
        elif model_name == "mobilenet_v2":
            return models.mobilenet_v2(pretrained=True)
        elif model_name == "mobilenet_v3_large":
            return models.mobilenet_v3_large(pretrained=True)
        elif model_name == "mobilenet_v3_small":
            return models.mobilenet_v3_small(pretrained=True)
        elif model_name == "regnet_y_400mf":
            return models.regnet_y_400mf(pretrained=True)
        elif model_name == "regnet_y_800mf":
            return models.regnet_y_800mf(pretrained=True)
        elif model_name == "regnet_y_1_6gf":
            return models.regnet_y_1_6gf(pretrained=True)
        elif model_name == "regnet_x_3_2gf":
            return models.regnet_x_3_2gf(pretrained=True)
        elif model_name == "regnet_x_800mf":
            return models.regnet_x_800mf(pretrained=True)
        elif model_name == "regnet_x_400mf":
            return models.regnet_x_400mf(pretrained=True)
        elif model_name == "resnet18":
            return models.resnet18(pretrained=True)
        elif model_name == "resnet34":
            return models.resnet34(pretrained=True)
        elif model_name == "resnet50":
            return models.resnet50(pretrained=True)
        elif model_name == "resnet101":
            return models.resnet101(pretrained=True)
        elif model_name == "shufflenet_v2_x0_5":
            return models.shufflenet_v2_x0_5(pretrained=True)
        elif model_name == "shufflenet_v2_x1_0":
            return models.shufflenet_v2_x1_0(pretrained=True)
        elif model_name == "squeezenet1_0":
            return models.squeezenet1_0(pretrained=True)

def check_memory_usage(model, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()  # Empty cached memory to get accurate measurements

    initial_allocated = torch.cuda.memory_allocated(device)
    initial_cached = torch.cuda.memory_reserved(device)
    model = model.eval()
    model = model.to(device)
    dummy_input = torch.randn(1, *input_size).to(device)
    model(dummy_input)  # forward pass to initialize all buffers

    final_allocated = torch.cuda.memory_allocated(device)
    final_cached = torch.cuda.memory_reserved(device)

    model_memory = final_allocated - initial_allocated
    cached_memory_diff = final_cached - initial_cached

    print(f"Model Name: {model.__class__.__name__}")
    print(f"Memory Allocated: {model_memory / (1024 ** 2):.2f} MB")
    print(f"Change in Cached Memory: {cached_memory_diff / (1024 ** 2):.2f} MB")
    print("-------------------------------------------------")

    # Check if memory allocated is greater than 1 GB
    assert model_memory / (1024 ** 2) < 1024, f"Model {model.__class__.__name__} is using more than 1 GB of memory"


def get_dummy_input_size(is_inception, is_regnet_y_128gf):
    if is_inception:
        return (3, 299, 299)
    elif is_regnet_y_128gf:
        return (3, 384, 384)
    else:
        return (3, 224, 224)


def main():
    Error_model_list = {}
    model_list = get_model_list()
    for model_name in model_list:
        try:
            model = get_model(model_name)
            print(f"Checking {model_name}")
            check_memory_usage(model, get_dummy_input_size(
                model_name == "inception_v3", model_name == "regnet_y_128gf"))
        except Exception as e:
            Error_model_list[model_name] = e
    print("Error_model_list: ", Error_model_list)


if __name__ == "__main__":
    main()
