import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from utils.postprocess import initial_segmentation_refinement, hybrid_filtering
from utils.metrics import pixel_iou


class Tester:
    def __init__(self, model, device, cfg):
        self.model = model
        self.device = device
        self.cfg = cfg
        self.test_cfg = cfg['test']

    def evaluate(self, test_loader):
        self.model.eval()
        
        results = {}
        results['baseline'] = {'total_iou': 0.0, 'count': 0}
        
        methods = self.test_cfg['postprocess_methods']
        method_key = '+'.join(methods) if methods else 'none'
        results[method_key] = {'total_iou': 0.0, 'count': 0}
        
        with torch.no_grad():
            for i, (img, labels) in enumerate(test_loader):
                img = img.to(self.device)
                
                pred = self.model(img)
                seg_pred = torch.sigmoid(pred[:, 0:1])
                cent_pred = torch.sigmoid(pred[:, 1:2])
                hyb_pred = pred[:, 2:3]
                
                seg_gt = labels[:, 0:1]
                gt_bin = (seg_gt[0, 0] > 0.5).cpu().numpy()
                
                seg_base = (seg_pred[0, 0] > self.test_cfg['seg_thresh']).cpu().numpy()
                iou_base = pixel_iou(seg_base, gt_bin)
                results['baseline']['total_iou'] += iou_base
                results['baseline']['count'] += 1
                
                seg_post = self._postprocess(seg_pred[0], cent_pred[0], hyb_pred[0])
                iou_post = pixel_iou(seg_post, gt_bin)
                results[method_key]['total_iou'] += iou_post
                results[method_key]['count'] += 1
        
        avg_base = results['baseline']['total_iou'] / results['baseline']['count']
        avg_post = results[method_key]['total_iou'] / results[method_key]['count']
        
        print(f"Baseline: {avg_base:.4f}")
        print(f"{method_key}: {avg_post:.4f}")
        
        self._visualize_sample(test_loader)
        
        return {
            'baseline_iou': avg_base,
            'postprocessed_iou': avg_post,
            'methods': methods
        }

    def _visualize_sample(self, test_loader, k=10):
        self.model.eval()
        output_dir = self.cfg['output'].get('vis_dir', 'output/visualizations')
        os.makedirs(output_dir, exist_ok=True)
        
        method_str = ' + '.join(self.test_cfg['postprocess_methods']) if self.test_cfg['postprocess_methods'] else 'none'
        
        with torch.no_grad():
            for i, (img, labels) in enumerate(test_loader):
                if i >= k:
                    break
                    
                img = img.to(self.device)
                
                pred = self.model(img)
                seg_pred = torch.sigmoid(pred[:, 0:1])
                cent_pred = torch.sigmoid(pred[:, 1:2])
                hyb_pred = pred[:, 2:3]
                
                seg_base = (seg_pred[0, 0] > self.test_cfg['seg_thresh']).cpu().numpy()
                seg_post = self._postprocess(seg_pred[0], cent_pred[0], hyb_pred[0])
                
                img_np = img[0].cpu().numpy().transpose(1, 2, 0)[:,:,:3]
                gt_np = labels[0, 0].cpu().numpy()
                
                iou_base = pixel_iou(seg_base, gt_np)
                iou_post = pixel_iou(seg_post, gt_np)
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                axes[0, 0].imshow(img_np)
                axes[0, 0].set_title('Input')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(gt_np, cmap='gray')
                axes[0, 1].set_title('GT')
                axes[0, 1].axis('off')
                
                axes[1, 0].imshow(seg_base, cmap='gray')
                axes[1, 0].set_title(f'Base ({iou_base:.3f})')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(seg_post, cmap='gray')
                axes[1, 1].set_title(f'Post ({iou_post:.3f})')
                axes[1, 1].axis('off')
                
                plt.suptitle(f'Sample {i+1} - {method_str}')
                plt.tight_layout()
                
                save_path = os.path.join(output_dir, f'test_sample_{i+1:02d}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"Saved {min(k, len(test_loader))} visualizations to {output_dir}")

    def _postprocess(self, seg, cent_map, hyb_map):
        methods = self.test_cfg['postprocess_methods']
        
        s = seg.squeeze().cpu().detach().numpy()
        h = hyb_map.squeeze().cpu().detach().numpy()
        
        mask = (s > self.test_cfg['seg_thresh']).astype(np.uint8)
        
        if 'initial_segmentation_refinement' in methods:
            mask = initial_segmentation_refinement(
                s, self.test_cfg['seg_thresh'], self.test_cfg['min_area']
            )
        
        if 'hybrid_filtering' in methods:
            mask = hybrid_filtering(mask, h, self.test_cfg['hyb_thresh'])
        
        return mask