def get_config():
    """Configuration dictionary for easy parameter tweaking"""
    cfg = {
        # Dataset configuration
        'dataset': {
            'data_root': "data/datasets/USA_segmentation",
            'patch_size': 128,
            'stride': 64,
            'nrg': True,
            'test_ratio': 0.2
        },
        
        # Dataloader configuration
        'dataloader': {
            'train_batch_size': 16,
            'test_batch_size': 1,
            'num_workers': 4
        },
        
        # Model configuration
        'model': {
            'name': 'UNet',
            'in_channels': 4,
            'n_classes': 3,
            'depth': 4,
        },
        
        # Trainer configuration
        'trainer': {
            'learning_rate': 1e-3,
            'epochs': 10,
            'hybrid_loss': True
        },
        
        # Output configuration
        'output': {
            'model_save_path': 'output/checkpoints/trained_model.pth',
            'vis_dir': 'output/visualizations'
        },

        'postprocess_methods': {
            'initial_segmentation_refinement': False,
        }
    }
    
    return cfg