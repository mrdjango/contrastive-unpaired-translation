from packaging import version
import torch
from torch import nn
import torch.nn.functional as F
import math

class TangentPatchNCELoss(nn.Module):
    """
    PatchNCE Loss with Tangent Distance incorporating multiple differentiable augmentations.
    
    The tangent distance is computed as:
    D_T(x, y) = min_{α} ||x + Σ(α_i * T_i(x)) - y||²
    where T_i are the tangent vectors corresponding to various transformations.
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.l2_norm = lambda x: F.normalize(x, p=2, dim=-1)
        self.epsilon = 1e-6
        
    def compute_tangent_vectors(self, feat):
        """
        Compute tangent vectors for various differentiable transformations.
        
        Args:
            feat: Feature tensor [B, C, H, W]
        Returns:
            List of tangent vectors for different transformations
        """
        B, C, H, W = feat.shape
        tangent_vectors = []
        
        # Compute coordinate grids
        x_coords = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(B, 1, H, W).to(feat.device)
        y_coords = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(B, 1, H, W).to(feat.device)
        
        # Compute partial derivatives
        dx = (torch.roll(feat, shifts=1, dims=3) - torch.roll(feat, shifts=-1, dims=3)) / (2*self.epsilon)
        dy = (torch.roll(feat, shifts=1, dims=2) - torch.roll(feat, shifts=-1, dims=2)) / (2*self.epsilon)
        
        # 1. Translation tangent vectors
        tangent_vectors.extend([dx, dy])
        
        # 2. Rotation tangent vector
        rotation = -y_coords * dx + x_coords * dy
        tangent_vectors.append(rotation)
        
        # 3. Scaling transformations
        scale = x_coords * dx + y_coords * dy  # isotropic
        scale_x = x_coords * dx  # anisotropic x
        scale_y = y_coords * dy  # anisotropic y
        tangent_vectors.extend([scale, scale_x, scale_y])
        
        # 4. Shearing transformations
        shear_x = y_coords * dx
        shear_y = x_coords * dy
        tangent_vectors.extend([shear_x, shear_y])
        
        # 5. Perspective transformation approximations
        persp_x = x_coords.pow(2) * dx
        persp_y = y_coords.pow(2) * dy
        tangent_vectors.extend([persp_x, persp_y])
        
        # 6. Color transformations (if applicable)
        if C >= 3 and getattr(self.opt, 'use_color_tangents', True):
            # Brightness
            brightness = feat * 0.1
            
            # Contrast
            mean = feat.mean(dim=[2, 3], keepdim=True)
            contrast = (feat - mean) * 0.1
            
            # Color saturation
            if C >= 3:
                luminance = (0.299 * feat[:, 0:1] + 0.587 * feat[:, 1:2] + 0.114 * feat[:, 2:3])
                saturation = (feat[:, :3] - luminance) * 0.1
                tangent_vectors.extend([brightness, contrast, saturation])
        
        return tangent_vectors

    def compute_tangent_distance(self, feat_q, feat_k):
        """
        Compute tangent distance between two feature sets.
        
        Args:
            feat_q: Query features [B, C, H, W]
            feat_k: Key features [B, C, H, W]
        Returns:
            Tangent distance tensor [B]
        """
        # Compute tangent vectors
        tangent_q = self.compute_tangent_vectors(feat_q)
        tangent_k = self.compute_tangent_vectors(feat_k)
        
        # Stack and normalize tangent vectors
        T_q = torch.stack([t.reshape(feat_q.shape[0], -1) for t in tangent_q], dim=2)
        T_k = torch.stack([t.reshape(feat_k.shape[0], -1) for t in tangent_k], dim=2)
        
        T_q = F.normalize(T_q, p=2, dim=1)
        T_k = F.normalize(T_k, p=2, dim=1)
        
        # Compute projection matrices
        I = torch.eye(T_q.shape[1], device=feat_q.device).unsqueeze(0)
        
        P_q = I - torch.bmm(
            torch.bmm(T_q, torch.inverse(torch.bmm(T_q.transpose(1, 2), T_q) + 
                     self.epsilon * I[:, :T_q.shape[2], :T_q.shape[2]])),
            T_q.transpose(1, 2)
        )
        
        P_k = I - torch.bmm(
            torch.bmm(T_k, torch.inverse(torch.bmm(T_k.transpose(1, 2), T_k) + 
                     self.epsilon * I[:, :T_k.shape[2], :T_k.shape[2]])),
            T_k.transpose(1, 2)
        )
        
        # Compute difference vector
        diff = feat_q.reshape(feat_q.shape[0], -1) - feat_k.reshape(feat_k.shape[0], -1)
        
        # Compute distances using both projections
        dist_q = torch.sum(torch.bmm(diff.unsqueeze(1), P_q) * diff.unsqueeze(1), dim=(1, 2))
        dist_k = torch.sum(torch.bmm(diff.unsqueeze(1), P_k) * diff.unsqueeze(1), dim=(1, 2))
        
        return torch.min(dist_q, dist_k)

    def forward(self, feat_q, feat_k):
        """
        Forward pass computing the tangent-aware NCE loss.
        
        Args:
            feat_q: Query features
            feat_k: Key features
        Returns:
            NCE loss combined with tangent distance
        """
        num_patches = feat_q.shape[0]
        feat_k = feat_k.detach()
        
        # Process in batches to handle memory efficiently
        l_neg = []
        batch_size = min(self.opt.batch_size, num_patches)
        for i in range(0, num_patches, batch_size):
            end_i = min(i + batch_size, num_patches)
            batch_q = feat_q[i:end_i]
            batch_k = feat_k[i:end_i]
            l_neg.append(self.compute_tangent_distance(batch_q, batch_k))
        l_neg = torch.cat(l_neg, dim=0)
        
        # Temperature scaling
        l_neg = -l_neg / self.opt.nce_T
        
        # Positive pairs
        l_pos = -self.compute_tangent_distance(
            feat_q.unsqueeze(1), 
            feat_k.unsqueeze(1)
        ) / self.opt.nce_T
        
        # Combine and compute final loss
        out = torch.cat((l_pos, l_neg), dim=1)
        loss = self.cross_entropy_loss(
            out, 
            torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        )
        
        return loss