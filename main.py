import torch
# import kagglehub
import os

from data.loader import prepare_dataloaders
from engine.trainer import Trainer
from engine.tester import Tester
from config.config import get_config
from model.loss import BCEDiceLoss, HybridLoss
from utils.model_utils import get_model

def main():
    # path = kagglehub.dataset_download("meteahishali/aerial-imagery-for-standing-dead-tree-segmentation")
    # image_root_idr = os.path.join(path, 'USA_segmentation')
    cfg = get_config()
    # cfg['dataset']['data_root'] = image_root_idr

    print('Cfg: \n', cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ################
    # Training #
    ################
    model = get_model(cfg).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg['trainer']['learning_rate']
    )
    loss_fn = HybridLoss() if cfg['trainer']['hybrid_loss'] else BCEDiceLoss()
    train_loader, test_loader = prepare_dataloaders(cfg)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )
    
    # Train the model
    epochs = cfg['trainer']['epochs']
    print(f"\nStarting training for {epochs} epochs...")
    trainer.fit(epochs)

    # Save the trained model
    model_path = cfg['output']['model_save_path']
    trainer.save(model_path)
    
    # Get training history
    history = trainer.get_history()
    print(f"\nTraining completed!")
    print(f"Final train loss: {history['train'][-1]['loss']:.4f}")

    ################
    # Testing #
    ################

    # Load trained model
    # model_path = cfg['output']['model_save_path']
    # try:
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    #     print(f"Loaded model from {model_path}")
    # except FileNotFoundError:
    #     print(f"Warning: Model file {model_path} not found. Using untrained model.")

    model.eval()

    # Get postprocess methods
    postprocess = [k for k, v in cfg['postprocess_methods'].items() if v]

    # Create tester with config
    tester = Tester(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
        postprocess=postprocess
    )

    print(f"Using postprocessing methods: {postprocess}")
    
    results = tester.evaluate()
    # tester.visualize_sample()

if __name__ == "__main__":
    main()