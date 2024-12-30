import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from .cut_model import CUTModel


class EnhancedCUTModel(CUTModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train)
        # Add augmentation-specific options
        parser.add_argument('--use_augmentation', type=bool, default=True, help='whether to use data augmentation')
        parser.add_argument('--aug_prob', type=float, default=0.5, help='probability of applying each augmentation')
        parser.add_argument('--color_jitter_params', type=dict, default={
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2,
            'hue': 0.1
        }, help='color jitter parameters')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.augment_params = {
            'brightness': opt.color_jitter_params['brightness'],
            'contrast': opt.color_jitter_params['contrast'],
            'saturation': opt.color_jitter_params['saturation'],
            'hue': opt.color_jitter_params['hue']
        }
        self.aug_prob = opt.aug_prob
        self.use_augmentation = opt.use_augmentation

    def augment(self, x):
        """Apply a series of augmentations to the input tensor"""
        if not self.use_augmentation:
            return x

        # Convert to PIL Image for some transformations
        if isinstance(x, torch.Tensor):
            # Ensure the input is in the correct format [0, 1]
            if x.min() < -0.5:  # If using normalization [-1, 1]
                x = (x + 1) / 2

        # Random horizontal flip
        if random.random() < self.aug_prob:
            x = TF.hflip(x)

        # Color jitter
        if random.random() < self.aug_prob:
            x = TF.adjust_brightness(x, random.uniform(1-self.augment_params['brightness'], 
                                                     1+self.augment_params['brightness']))
        if random.random() < self.aug_prob:
            x = TF.adjust_contrast(x, random.uniform(1-self.augment_params['contrast'], 
                                                   1+self.augment_params['contrast']))
        if random.random() < self.aug_prob:
            x = TF.adjust_saturation(x, random.uniform(1-self.augment_params['saturation'], 
                                                     1+self.augment_params['saturation']))
        if random.random() < self.aug_prob:
            x = TF.adjust_hue(x, random.uniform(-self.augment_params['hue'], 
                                              self.augment_params['hue']))

        # Random rotation
        if random.random() < self.aug_prob:
            angle = random.uniform(-10, 10)
            x = TF.rotate(x, angle)

        # Ensure the output is in the correct format [-1, 1] if needed
        if isinstance(x, torch.Tensor):
            if self.opt.input_nc == 3:  # Only normalize if working with images
                x = x * 2 - 1

        return x

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # Apply augmentations
        if self.isTrain and self.use_augmentation:
            self.real_A = self.augment(self.real_A)
            self.real_B = self.augment(self.real_B)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # First, generate fake images
        self.forward()

        # Update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # Update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()