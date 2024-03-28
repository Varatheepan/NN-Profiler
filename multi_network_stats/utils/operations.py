
import torch
import torch.nn.functional as F

from torch import nn

from torchvision import transforms
from torchvision.transforms import InterpolationMode

class _alexnet_op(nn.Module):
    @torch.no_grad()
    def forward(self,x: torch.Tensor):
        return torch.flatten(x, 1)

class _densenet_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

class _efficientnet_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return torch.flatten(x, 1)

class _googlenet_prep_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        return torch.cat((x_ch0, x_ch1, x_ch2), 1)

class _googlenet_post_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return torch.flatten(x, 1)
    
class _inception_prep_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        return torch.cat((x_ch0, x_ch1, x_ch2), 1)

class _inception_post_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return torch.flatten(x, 1)
    
class _mnasnet_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return x.mean([2, 3])
    
class _mobilenet_v2_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

class _mobilenet_v3_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return torch.flatten(x, 1)
    
class _regnet_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return torch.flatten(x, 1)
    
class _resnet_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return torch.flatten(x, 1)
    
class _shufflenet_v2_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return x.mean([2, 3])
    
class _vgg_op(nn.Module):
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        return torch.flatten(x, 1)
    

class CustomOpExecutor:
    def __init__(self, preprocess: bool = False, verbose: bool = False):
        self.preprocess = preprocess
        self.operations = {}
        self.model_family_ops = {}
        self.preprocess_ops = {}
        self.verbose = verbose

    # @staticmethod
    # @torch.no_grad()
    # def _alexnet_op(x: torch.Tensor):
    #     return torch.flatten(x, 1)

    # @staticmethod
    # @torch.no_grad()
    # def _densenet_op(x: torch.Tensor):
    #     x = F.relu(x, inplace=True)
    #     x = F.adaptive_avg_pool2d(x, (1, 1))
    #     return torch.flatten(x, 1)
    
    # @staticmethod
    # @torch.no_grad()
    # def _efficientnet_op(x: torch.Tensor):
    #     return torch.flatten(x, 1)

    # @staticmethod
    # @torch.no_grad()
    # def _googlenet_prep_op(x: torch.Tensor):
    #     x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    #     x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    #     x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    #     return torch.cat((x_ch0, x_ch1, x_ch2), 1)

    # @staticmethod
    # @torch.no_grad()
    # def _googlenet_post_op(x: torch.Tensor):
    #     return torch.flatten(x, 1)
    
    # @staticmethod
    # @torch.no_grad()
    # def _inception_prep_op(x: torch.Tensor):
    #     x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    #     x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    #     x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    #     return torch.cat((x_ch0, x_ch1, x_ch2), 1)
    
    # @staticmethod
    # @torch.no_grad()
    # def _inception_post_op(x: torch.Tensor):
    #     return torch.flatten(x, 1)
    
    # @staticmethod
    # @torch.no_grad()
    # def _mnasnet_op(x: torch.Tensor):
    #     return x.mean([2, 3])
    
    # @staticmethod
    # @torch.no_grad()
    # def _mobilenet_v2_op(x: torch.Tensor):
    #     x = F.adaptive_avg_pool2d(x, (1, 1))
    #     return torch.flatten(x, 1)

    # @staticmethod
    # @torch.no_grad()
    # def _mobilenet_v3_op(x: torch.Tensor):
    #     return torch.flatten(x, 1)
    
    # @staticmethod
    # @torch.no_grad()
    # def _regnet_op(x: torch.Tensor):
    #     return torch.flatten(x, 1)
    
    # @staticmethod
    # @torch.no_grad()
    # def _resnet_op(x: torch.Tensor):
    #     return torch.flatten(x, 1)
    
    # @staticmethod
    # @torch.no_grad()
    # def _shufflenet_v2_op(x: torch.Tensor):
    #     return x.mean([2, 3])
    
    # @staticmethod
    # @torch.no_grad()
    # def _vgg_op(x: torch.Tensor):
    #     return torch.flatten(x, 1)
    
    def register_preprocess(self, model_name: str, transform):
        """Registers a preprocessing operation for a specific model family."""
        self.preprocess_ops[model_name] = transform

    def register_family_operation(self, family_name, operation: nn.Module):
        """Associates a family with a specific operation."""
        if family_name not in self.model_family_ops:
            self.model_family_ops[family_name] = []
        self.model_family_ops[family_name].append(operation)

    def register_operation(self, model_name, family_name, layer_idx, operation_indices: set = None):
        """Registers specific operations of a family for a specific model and layer."""
        family_operations = self.model_family_ops.get(family_name)
        if family_operations:
            if operation_indices is not None:
                # Filter operations based on indices
                selected_operations = [family_operations[i] for i in operation_indices if i < len(family_operations)]
            else:
                # If no specific indices provided, use all operations
                selected_operations = family_operations
            key = f"{model_name}_{layer_idx}"
            self.operations[key] = selected_operations
        else:
            if self.verbose:
                print(f"No operations registered for family {family_name}")

    def finalize_registration(self):
        """Finalizes the registration process by appending universal preprocessing to layer 0."""
        if self.preprocess:
            for model_name, preprocess_op in self.preprocess_ops.items():
                key = f"{model_name}_0"
                # If there are already operations for layer 0
                if key in self.operations:
                    # Prepend the preprocessing op
                    self.operations[key].insert(0, preprocess_op)
                else:
                    # Otherwise, just set the preprocessing op for layer 0
                    self.operations[key] = [preprocess_op]
    
    def execute(
        self, model_name: str, layer_idx: int, x: torch.Tensor, 
        device: torch.device = torch.device('cpu')
    ):
        key = f"{model_name}_{layer_idx}"
        operations = self.operations.get(key, [lambda x: x])
        # print("operations: ",[op for op in operations])
        cntr = 0
        for op in operations:
            if layer_idx == 0 and cntr == 0:
                if self.preprocess:
                    x = op(x).unsqueeze(0).to(device)
                    cntr += 1
            else:
                x = op(x)
        return x


def get_model_list():
    return [
        "alexnet",
        "densenet121", "densenet161", "densenet169", "densenet201",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
        "googlenet",
        "inception_v3",
        "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3",
        "mobilenet_v2",
        "mobilenet_v3_large", "mobilenet_v3_small",
        "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf", "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf", "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf", "regnet_x_8gf", "regnet_x_16gf", "regnet_x_32gf",
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "resnext50_32x4d", "resnext101_32x8d",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
        "squeezenet1_0", "squeezenet1_1",
        "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
        "wide_resnet50_2", "wide_resnet101_2"
    ]


def database_spawn(preprocess: bool = False, pretrained: bool = False, verbose: bool = False):
    # Define preprocessing stack for Inception models
    inception_preprocess = transforms.Compose([
        transforms.Resize(342),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
    ])

    # Define preprocessing stack for all other models
    default_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
    ])

    swag_preprocess = transforms.Compose([
        transforms.Resize(384, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    
    # Custom operation registration
    op_executor = CustomOpExecutor(preprocess=preprocess, verbose=verbose)

    # Registering family operations
    op_executor.register_family_operation("alexnet", _alexnet_op())
    op_executor.register_family_operation("densenet", _densenet_op())
    op_executor.register_family_operation("efficientnet", _efficientnet_op())
    op_executor.register_family_operation("googlenet", _googlenet_prep_op())
    op_executor.register_family_operation("googlenet", _googlenet_post_op())
    op_executor.register_family_operation("inception", _inception_prep_op())
    op_executor.register_family_operation("inception", _inception_post_op())
    op_executor.register_family_operation("mnasnet", _mnasnet_op())
    op_executor.register_family_operation("mobilenet_v2", _mobilenet_v2_op())
    op_executor.register_family_operation("mobilenet_v3", _mobilenet_v3_op())
    op_executor.register_family_operation("regnet_x", _regnet_op())
    op_executor.register_family_operation("regnet_y", _regnet_op())
    op_executor.register_family_operation("resnet", _resnet_op())
    op_executor.register_family_operation("resnext", _resnet_op())
    op_executor.register_family_operation("shufflenet_v2", _shufflenet_v2_op())
    op_executor.register_family_operation("vgg", _vgg_op())

    # Register preprocessing step
    model_list = get_model_list()
    for model_name in model_list:
        if model_name in ["inception_v3"]:
            op_executor.register_preprocess(model_name, inception_preprocess)
        elif model_name in ["regnet_y_128gf"]:
            op_executor.register_preprocess(model_name, swag_preprocess)
        else:
            op_executor.register_preprocess(model_name, default_preprocess)

    # Registering model operations based on family
    op_executor.register_operation("alexnet", "alexnet", 14)
    op_executor.register_operation("densenet121", "densenet", 12)
    op_executor.register_operation("densenet161", "densenet", 12)
    op_executor.register_operation("densenet169", "densenet", 12)
    op_executor.register_operation("densenet201", "densenet", 12)
    op_executor.register_operation("efficientnet_b0", "efficientnet", 10)
    op_executor.register_operation("efficientnet_b1", "efficientnet", 10)
    op_executor.register_operation("efficientnet_b2", "efficientnet", 10)
    op_executor.register_operation("efficientnet_b3", "efficientnet", 10)
    op_executor.register_operation("efficientnet_b4", "efficientnet", 10)
    op_executor.register_operation("efficientnet_b5", "efficientnet", 10)
    op_executor.register_operation("efficientnet_b6", "efficientnet", 10)
    op_executor.register_operation("efficientnet_b7", "efficientnet", 10)
    op_executor.register_operation("efficientnet_v2_l", "efficientnet", 10)
    op_executor.register_operation("efficientnet_v2_m", "efficientnet", 10)
    op_executor.register_operation("efficientnet_v2_s", "efficientnet", 10)
    op_executor.register_operation("googlenet", "googlenet", 0, {0})
    op_executor.register_operation("googlenet", "googlenet", 17, {1})
    op_executor.register_operation("inception_v3", "inception", 0, {0})
    op_executor.register_operation("inception_v3", "inception", 20, {1})
    op_executor.register_operation("mnasnet0_5", "mnasnet", 17)
    op_executor.register_operation("mnasnet0_75", "mnasnet", 17)
    op_executor.register_operation("mnasnet1_0", "mnasnet", 17)
    op_executor.register_operation("mnasnet1_3", "mnasnet", 17)
    op_executor.register_operation("mobilenet_v2", "mobilenet_v2", 19)
    op_executor.register_operation("mobilenet_v3_large", "mobilenet_v3", 18)
    op_executor.register_operation("mobilenet_v3_small", "mobilenet_v3", 14)
    op_executor.register_operation("regnet_x_16gf", "regnet_x", 8)
    op_executor.register_operation("regnet_x_1_6gf", "regnet_x", 8)
    op_executor.register_operation("regnet_x_32gf", "regnet_x", 8)
    op_executor.register_operation("regnet_x_3_2gf", "regnet_x", 8)
    op_executor.register_operation("regnet_x_400mf", "regnet_x", 8)
    op_executor.register_operation("regnet_x_800mf", "regnet_x", 8)
    op_executor.register_operation("regnet_x_8gf", "regnet_x", 8)
    op_executor.register_operation("regnet_y_128gf", "regnet_y", 8)
    op_executor.register_operation("regnet_y_16gf", "regnet_y", 8)
    op_executor.register_operation("regnet_y_1_6gf", "regnet_y", 8)
    op_executor.register_operation("regnet_y_32gf", "regnet_y", 8)
    op_executor.register_operation("regnet_y_3_2gf", "regnet_y", 8)
    op_executor.register_operation("regnet_y_400mf", "regnet_y", 8)
    op_executor.register_operation("regnet_y_800mf", "regnet_y", 8)
    op_executor.register_operation("regnet_y_8gf", "regnet_y", 8)
    op_executor.register_operation("resnet101", "resnet", 38)
    op_executor.register_operation("resnet152", "resnet", 55)
    op_executor.register_operation("resnet18", "resnet", 13)
    op_executor.register_operation("resnet34", "resnet", 21)
    op_executor.register_operation("resnet50", "resnet", 21)
    op_executor.register_operation("wide_resnet101_2", "resnet", 38)
    op_executor.register_operation("wide_resnet50_2", "resnet", 21)
    op_executor.register_operation("resnext101_32x8d", "resnext", 38)
    op_executor.register_operation("resnext101_64x4d", "resnext", 38)
    op_executor.register_operation("resnext50_32x4d", "resnext", 21)
    op_executor.register_operation("shufflenet_v2_x0_5", "shufflenet_v2", 23)
    op_executor.register_operation("shufflenet_v2_x1_0", "shufflenet_v2", 23)
    op_executor.register_operation("shufflenet_v2_x1_5", "shufflenet_v2", 23)
    op_executor.register_operation("shufflenet_v2_x2_0", "shufflenet_v2", 23)
    op_executor.register_operation("vgg11", "vgg", 22)
    op_executor.register_operation("vgg11_bn", "vgg", 30)
    op_executor.register_operation("vgg13", "vgg", 26)
    op_executor.register_operation("vgg13_bn", "vgg", 36)
    op_executor.register_operation("vgg16", "vgg", 32)
    op_executor.register_operation("vgg16_bn", "vgg", 45)
    op_executor.register_operation("vgg19", "vgg", 38)
    op_executor.register_operation("vgg19_bn", "vgg", 54)

    # Finalize registration
    op_executor.finalize_registration()

    # Return the executor
    return op_executor
