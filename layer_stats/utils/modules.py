
import torch
import torch.nn as nn


class VitConvOp(nn.Module):
    def __init__(self, layer, class_token, patch_size, image_size, hidden_dim):
        super(VitConvOp, self).__init__()
        self.layer = layer
        self.class_token = class_token
        self.patch_size = patch_size
        self.image_size = image_size
        self.hidden_dim = hidden_dim

    def forward(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.layer(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        return torch.cat([batch_class_token, x], dim=1)


class VitPosOp(nn.Module):
    def __init__(self, layer):
        super(VitPosOp, self).__init__()
        self.layer = layer
    
    def forward(self, x):
        torch._assert(x.dim(
        ) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        return x + self.layer
