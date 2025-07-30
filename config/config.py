def get_config():
    """Configuration dictionary for easy parameter tweaking"""
    cfg = {
        # Dataset configuration
        'dataset': {
            'data_root': "data/datasets/USA_segmentation",
            'patch_size': 128,
            'stride': 32,
            'val_ratio': 0.2,
            'test_ratio': 0.1
        },
        
        # Dataloader configuration
        'dataloader': {
            'train_batch_size': 16,
            'val_batch_size': 8,
            'test_batch_size': 1,
            'num_workers': 4
        },
        
        # Model configuration
        'model': {
            'in_channels': 4,
            'n_classes': 3,
            'depth': 4,
        },
        
        # Trainer configuration
        'trainer': {
            'learning_rate': 1e-4,
            'epochs': 10,
        },
        
        # Output configuration
        'output': {
            'model_save_path': 'output/checkpoints/trained_model.pth',
            'vis_dir': 'output/visualizations'
        },

        # Test configuration
        'test': {
            'postprocess_methods': [
                'initial_segmentation_refinement',
                # 'hybrid_filtering'
            ],
            'seg_thresh': 0.5,
            'hyb_thresh': 0.5,
            'min_area': 50,
        }
    }
    
    return cfg