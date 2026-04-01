# Copyright (c) Yukun Zhou. # All rights reserved.
# --------------------------------------------------------

import timm
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


def Dinov2(args, **kwargs):
    
    if args.model_arch == 'dinov2_vits14':
        arch = 'vit_small_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitb14':
        arch = 'vit_base_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitl14':
        arch = 'vit_large_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitg14':
        arch = 'vit_giant_patch14_dinov2.lvd142m'
    else:
        raise ValueError(f"Unknown model_arch '{args.model_arch}'. "
                         f"Expected one of: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14")
        
    model = timm.create_model(
        arch,
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model


def RETFound_dinov2(args, **kwargs):
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=True,
        img_size=224,
        **kwargs
    )
    return model


def Dinov3(args, **kwargs):
    # Load ViT-L/16 backbone (hub model has `head = Identity` by default)
    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model=args.model_arch,
        pretrained=False,   # main() will load your checkpoint
        trust_repo=True,
    )

    # Figure out feature dimension for the probe
    feat_dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    if feat_dim is None:
        # Fallback: infer from a dummy forward
        model.eval()
        with torch.no_grad():
            H = W = getattr(args, "input_size", 224)
            dummy = torch.zeros(1, 3, H, W)
            out = model.forward_features(dummy)  # dinov3 returns dict or tensor
            if isinstance(out, dict):
                if args.global_pool and "x_norm_patchtokens" in out:
                    z = out["x_norm_patchtokens"].mean(1)
                elif "x_norm_clstoken" in out:
                    z = out["x_norm_clstoken"]
                    if z.dim() == 3:  # (B,1,C)
                        z = z.squeeze(1)
                else:
                    z = next(iter(out.values()))
                    if z.dim() == 3:
                        z = z[:, 0]
            else:
                z = out[:, 0] if out.dim() == 3 else out
            feat_dim = z.shape[-1]

    # Replace identity head with a linear probe
    model.head = nn.Linear(feat_dim, args.nb_classes)
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)

    return model




def MED_dinov3(args, **kwargs):
    # Load ViT-L/16 backbone (hub model has `head = Identity` by default)
    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model="dinov3_vitl16",
        pretrained=True,   # main() will load your checkpoint
        trust_repo=True,
    )

    # Figure out feature dimension for the probe
    feat_dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    if feat_dim is None:
        # Fallback: infer from a dummy forward
        model.eval()
        with torch.no_grad():
            H = W = getattr(args, "input_size", 224)
            dummy = torch.zeros(1, 3, H, W)
            out = model.forward_features(dummy)  # dinov3 returns dict or tensor
            if isinstance(out, dict):
                if args.global_pool and "x_norm_patchtokens" in out:
                    z = out["x_norm_patchtokens"].mean(1)
                elif "x_norm_clstoken" in out:
                    z = out["x_norm_clstoken"]
                    if z.dim() == 3:  # (B,1,C)
                        z = z.squeeze(1)
                else:
                    z = next(iter(out.values()))
                    if z.dim() == 3:
                        z = z[:, 0]
            else:
                z = out[:, 0] if out.dim() == 3 else out
            feat_dim = z.shape[-1]

    # Replace identity head with a linear probe
    model.head = nn.Linear(feat_dim, args.nb_classes)
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)

    return model
