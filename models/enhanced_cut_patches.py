import torch
from torch import nn
import torch.nn.functional as F
import random
from .cut_model import CUTModel

class PatchAugmentor:
    def __init__(self, aug_prob=0.5):
        self.aug_prob = aug_prob
        
    def augment_patch(self, patch):
        """Apply augmentations to a single patch"""
        if random.random() < self.aug_prob:
            # Random feature perturbation
            noise = torch.randn_like(patch) * 0.1
            patch = patch + noise
            
        if random.random() < self.aug_prob:
            # Feature dropout
            mask = torch.bernoulli(torch.ones_like(patch) * 0.9)
            patch = patch * mask
            
        # Normalize feature vector
        patch = F.normalize(patch, dim=-1)
        return patch

    def create_positive_patches(self, patch, num_augmentations=2):
        """Create multiple positive patches through augmentation"""
        positive_patches = [patch]  # Original patch
        for _ in range(num_augmentations):
            aug_patch = self.augment_patch(patch.clone())
            positive_patches.append(aug_patch)
        return torch.cat(positive_patches, dim=0)  # Changed from stack to cat

class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.patch_augmentor = PatchAugmentor(aug_prob=getattr(opt, 'patch_aug_prob', 0.5))
        self.num_augmentations = getattr(opt, 'num_augmentations', 2)
        
    def forward(self, feat_q, feat_k):
        batch_size, dim = feat_q.shape
        
        total_loss = 0
        for idx in range(batch_size):
            # Extract current query and key patches
            q_patch = feat_q[idx:idx+1]  # [1, dim]
            k_patch = feat_k[idx:idx+1]  # [1, dim]
            
            # Create augmented positive patches [num_aug+1, dim]
            positive_patches = self.patch_augmentor.create_positive_patches(
                k_patch, 
                self.num_augmentations
            )
            
            # Calculate positive logits
            # [1, dim] x [num_aug+1, dim]T -> [1, num_aug+1]
            l_pos = torch.mm(q_patch, positive_patches.t())
            
            # Create negative pairs from other patches in the batch
            negatives = torch.cat([feat_k[:idx], feat_k[idx+1:]], dim=0)  # [batch_size-1, dim]
            
            # Calculate negative logits
            # [1, dim] x [batch_size-1, dim]T -> [1, batch_size-1]
            l_neg = torch.mm(q_patch, negatives.t())
            
            # Combine positive and negative logits
            logits = torch.cat([l_pos, l_neg], dim=1)  # [1, num_aug+1+batch_size-1]
            
            # Create labels (first num_augmentations + 1 are positive)
            labels = torch.zeros(1, dtype=torch.long, device=logits.device)
            
            # Calculate NCE loss for this patch
            loss = self.cross_entropy_loss(logits / self.opt.nce_T, labels)
            total_loss += loss
            
        return total_loss / batch_size

class PatchAugmentedCUTModel(CUTModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--patch_aug_prob', type=float, default=0.5,
                          help='probability of applying augmentation to each patch')
        parser.add_argument('--num_augmentations', type=int, default=2,
                          help='number of augmented positive patches to create')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.criterionNCE = []
        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, 
                                            self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers