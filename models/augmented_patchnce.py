import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random

class AugmentedPatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool
        self.patch_augmentor = PatchAugmentor()
        
    def compute_single_loss(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # Compute positive logits
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), 
            feat_k.view(num_patches, -1, 1)
        )
        l_pos = l_pos.view(num_patches, 1)

        # Handle batch organization for negatives
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # Reshape features for batch processing
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        
        # Compute negative logits
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # Mask out self-similarities
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss

    def forward(self, feat_q, feat_k):
        # Compute regular patch loss
        regular_loss = self.compute_single_loss(feat_q, feat_k)
        
        # Create augmented features
        feat_q_aug = self.patch_augmentor(feat_q)
        feat_k_aug = self.patch_augmentor(feat_k)
        
        # Compute augmented patch loss
        augmented_loss = self.compute_single_loss(feat_q_aug, feat_k_aug)
        
        # Combine losses
        total_loss = regular_loss + augmented_loss
        
        return total_loss


class PatchAugmentor(nn.Module):
    def __init__(self, aug_prob=0.5):
        super().__init__()
        self.aug_prob = aug_prob
        
    def forward(self, x):
        """Apply augmentations to patch features"""
        if random.random() < self.aug_prob:
            # Add random noise
            noise = torch.randn_like(x) * 0.1
            x = x + noise
        
        if random.random() < self.aug_prob:
            # Feature dropout
            mask = torch.bernoulli(torch.ones_like(x) * 0.9)
            x = x * mask
        
        # Normalize feature vectors
        x = F.normalize(x, dim=1)
        
        return x