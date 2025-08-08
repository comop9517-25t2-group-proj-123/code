import importlib

from ptflops import get_model_complexity_info
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