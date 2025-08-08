import importlib

from ptflops import get_model_complexity_info
from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
import torch.nn as nn


def get_model_info(model, cfg):
    input_shape = (cfg['model']['in_channels'], cfg['dataset']['patch_size'], cfg['dataset']['patch_size'])
    flops, params = get_model_complexity_info(
        model, 
        input_shape, 
        print_per_layer_stat=False,
        as_strings=False, 
        verbose=False
    )

    if params is None:
        print('Pretrained UNet')
        return 
    
    
    params_m = params / 1e6 
    flops_g = flops / 1e9

    print(f'| Model | Params (M) | FLOPs (G) | Batch Size | Learning Rate | Optimizer |')
    print(f"| {cfg['model']['name']} | {params_m:.1f} | {flops_g:.1f} | {cfg['dataloader']['train_batch_size']} | {cfg['trainer']['learning_rate']} | Adam |")


def get_model(cfg):
    model_name = cfg['model']['name']
    module = importlib.import_module(f"model.{model_name}")
    model_class = getattr(module, model_name)
    # Pass only the arguments your model needs
    model = model_class(
        in_channels=cfg['model']['in_channels'],
        n_classes=cfg['model']['n_classes'],
        depth=cfg['model']['depth'],
    ) 

    model = model_class()

    get_model_info(model, cfg)


    return model

def get_pretrain_model(cfg):
    cache_dir = '/srv/scratch/CRUISE/shawn/cache/huggingface'
    processor = AutoImageProcessor.from_pretrained(cfg['model']['name'], cache_dir=cache_dir)
    model = MaskFormerForInstanceSegmentation.from_pretrained(cfg['model']['name'], cache_dir=cache_dir, use_safetensors=True )
    
    # Freeze backbone and transformer
    for param in model.model.pixel_level_module.parameters():
        param.requires_grad = False
    for param in model.model.transformer_module.parameters():
        param.requires_grad = False
    
    # Replace class predictor for binary segmentation
    model.class_predictor = nn.Linear(model.class_predictor.in_features, cfg['model']['n_classes'])
    
    get_model_info(model, cfg)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model, processor