import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

from utils.postprocess import apply_postprocess

class Tester:
    def __init__(self, model, dataloader, loss_fn, device, postprocess):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.device = device
        self.postprocess = postprocess

    def evaluate(self):
        self.model.eval()
        total_iou, total_precision, total_recall, total_f1, total_loss = 0, 0, 0, 0, 0
        count = 0

        with torch.no_grad():
            for img, labels in self.dataloader:
                img = img.to(self.device)
                labels = labels.to(self.device)
                pred = self.model(img)
                loss = self.loss_fn(pred, labels)

                if pred.shape[1] == 3:
                    pred_seg = torch.sigmoid(pred[:, 0:1])
                    gt_seg = labels[:, 0:1]
                else:
                    pred_seg = torch.sigmoid(pred)
                    gt_seg = labels

                for p in self.postprocess:
                    pred_seg = apply_postprocess(pred_seg, p)
                
                pred_binary = pred_seg > 0.5
                gt_binary = gt_seg > 0.5

                for b in range(pred_binary.shape[0]):
                    pred_flat = pred_binary[b].cpu().numpy().astype(np.uint8).flatten()
                    gt_flat = gt_binary[b].cpu().numpy().astype(np.uint8).flatten()

                    if gt_flat.sum() == 0 and pred_flat.sum() == 0:
                        continue

                    total_iou += jaccard_score(gt_flat, pred_flat, zero_division=0)
                    total_precision += precision_score(gt_flat, pred_flat, zero_division=0)
                    total_recall += recall_score(gt_flat, pred_flat, zero_division=0)
                    total_f1 += f1_score(gt_flat, pred_flat, zero_division=0)
                    count += 1

                total_loss += loss.item()

        if count == 0:
            return {'mean_iou': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'loss': 0, 'count': 0}

        results = {
            'mean_iou': total_iou / count,
            'precision': total_precision / count,
            'recall': total_recall / count,
            'f1': total_f1 / count,
            'loss': total_loss / len(self.dataloader),
            'count': count
        }
        
        print(f"Eval: | mIoU | F1 | Precision | Recall |")
        print(f"      | {results['mean_iou']:.3f} | {results['f1']:.3f} | {results['precision']:.3f} | {results['recall']:.3f} |")
        return results

    def visualize_sample(self, k=10, vis_dir='output/visualizations'):
        self.model.eval()
        os.makedirs(vis_dir, exist_ok=True)
        method_str = ' + '.join(self.postprocess) if self.postprocess else 'none'
        count = 0

        with torch.no_grad():
            for img, labels in self.dataloader:
                if count >= k:
                    break
                img = img.to(self.device)
                labels = labels.to(self.device)
                pred = self.model(img)
                
                if pred.shape[1] == 3:
                    pred_seg = torch.sigmoid(pred[:, 0:1])
                    gt_seg = labels[:, 0:1]
                else:
                    pred_seg = torch.sigmoid(pred)
                    gt_seg = labels
                
                for p in self.postprocess:
                    pred_seg = apply_postprocess(pred_seg, p)
                
                img_np = img[0].cpu().numpy().transpose(1, 2, 0)
                pred_mask = pred_seg[0, 0].cpu().numpy()
                gt_mask = gt_seg[0, 0].cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img_np[..., :3])
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                axes[1].imshow(pred_mask, cmap='gray')
                axes[1].set_title(f'Prediction | Postprocess: {method_str}')
                axes[1].axis('off')
                axes[2].imshow(gt_mask, cmap='gray')
                axes[2].set_title('Ground Truth')
                axes[2].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'sample_{count+1:02d}.png'), dpi=150)
                plt.close(fig)
                count += 1

        print(f"Saved {count} visualizations to {vis_dir}")