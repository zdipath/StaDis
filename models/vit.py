from torchvision.models.vision_transformer import ConvStemConfig,Encoder
from torchvision.utils import _log_api_usage_once
from torchvision.ops.misc import Conv2dNormActivation
import torch
from torch import nn
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from functools import partial
from collections import OrderedDict

from .IBOTvit import VisionTransformer


def base_vit(path_model,device,num_classes):
    #torch.cuda.set_device(device)
    model=vit_base(patch_size=16, num_classes=num_classes, use_mean_pooling=False)
    
    state_dict = torch.load(path_model, map_location="cpu")
    '''
    state_dict = state_dict['teacher']
    state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
            }
    state_dict = {
            k.replace("backbone.", ""): v for k, v in state_dict.items()
            }'''
    model.load_state_dict(state_dict, strict=False)

    return model







def vit_base(patch_size: int = 16, **kwargs):
    """Create a Base Vision Transformer model (ViT-B).

    Parameters
    ----------
    patch_size : int = 16
        Size of the patches in the input image.
    **kwargs : keyword arguments
        Additional arguments to be passed to the VisionTransformer constructor.

    Returns
    -------
    model : VisionTransformer
        The Base Vision Transformer model.
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs
    )
    return model


class ViT_B_16(VisionTransformer):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 num_layers=12,
                 num_heads=12,
                 hidden_dim=768,
                 mlp_dim=3072,
                 num_classes=1000):
        super(ViT_B_16, self).__init__(image_size=image_size,
                                       patch_size=patch_size,
                                       num_layers=num_layers,
                                       num_heads=num_heads,
                                       hidden_dim=hidden_dim,
                                       mlp_dim=mlp_dim,
                                       num_classes=num_classes)
        self.feature_size = hidden_dim

    def forward(self, x, return_feature=False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        if return_feature:
            return self.heads(x), x
        else:
            return self.heads(x)

    def forward_threshold(self, x, threshold):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        feature = x.clip(max=threshold)
        logits_cls = self.heads(feature)

        return logits_cls

    def get_fc(self):
        fc = self.heads[0]
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.heads[0]