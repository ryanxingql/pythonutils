import math
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as tmp
from collections import OrderedDict
from torchvision.models import vgg as vgg

# ===
# Multi-processing
# ===

def init_dist(local_rank=0, backend='nccl'):
    tmp.set_start_method('spawn', force=True)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)

# ===
# Loss
# ===

loss_lst = ['CharbonnierLoss', 'GANLoss', 'RelativisticGANLoss', 'VGGLoss']

def return_loss_func(name, opts):
    assert (name in loss_lst), '> Unsupported loss fn!'
    loss_func_cls = globals()[name]
    return loss_func_cls(**opts)

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, inp, ref):
        assert len(inp.shape) <= 4, '> Unsupported input!'
        if len(inp.shape) < 3:
            inp = inp.unsqueeze(0)
        n = inp.shape[0]
        diff = [torch.add(inp[k], -ref[k]) for k in range(n)]
        error = [torch.sqrt(diff[k] * diff[k] + self.eps) for k in range(n)]
        loss = torch.mean(torch.stack([torch.mean(error[k]) for k in range(n)]))
        return loss

class GANLoss(nn.Module):
    def __init__(
            self,
            real_label_val=1.0,
            fake_label_val=0.0,
            ):
        """
        Args:
            real_label_val (float): The value for real label. Default: 1.0.
            fake_label_val (float): The value for fake label. Default: 0.0.
        """
        super().__init__()

        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')  # sigmoid -> Binary Cross Entropy

    def forward(self, input_t, if_real):
        """
        Args:
            input_t: a list of scores from GANs discriminator.
            if_real (True|False): if the sample is real (gt, not enhanced sample).
        """
        target_val = (
            self.real_label_val if if_real else self.fake_label_val
            )
        target_label = input_t.new_ones(input_t.size()) * target_val
        loss = self.loss(input_t, target_label)
        return loss

class RelativisticGANLoss(nn.Module):
    def __init__(
            self,
            real_label_val=1.,
            fake_label_val=0.,
            ):
        """
        gan_loss_func(x - y, target)
        if x is more real than y, then two d out distance aims to be 1. for dis.
        if x is less real than y, then two d out distance aims to be 0. for dis.

        Args:
            real_label_val (float): The value for real label. Default: 1..
            fake_label_val (float): The value for fake label. Default: 0..
        """
        super().__init__()

        self.gan_loss = GANLoss(real_label_val, fake_label_val)

    def forward(self, dis, data_real, data_fake, inter_step, mode='gen'):
        """
        For gen loss:
            gt -> dis -> real_pred (can detach)
            lq -> gen -> enhanced -> dis -> fake_pred
        For dis loss:
            gt -> dis -> real_pred
            lq -> gen -> enhanced (can detach) -> dis -> fake_pred

        Args:
            mode (gen|dis)

        If we use VGG discriminator which includes inplace BN operation, we should make sure that we use dis only once before backward.
        """
        assert mode in ['gen', 'dis'], '> Unsupported mode!'
        if mode == 'gen':
            real_pred = dis(data_real).detach()  # unrelated to dis
            fake_pred = dis(data_fake)  # use dis here; only once
            loss_real = self.gan_loss(real_pred - torch.mean(fake_pred), False)
            loss_fake = self.gan_loss(fake_pred - torch.mean(real_pred), True)
            loss = loss_real * 0.5 + loss_fake * 0.5
            loss /= inter_step.float()  # multiple backwards and step once, thus mean
            return loss
            
        elif mode == 'dis':
            # fake_pred must be in front of real_pred, so that dis is only used once
            fake_pred_d = dis(data_fake.detach()).detach()
            real_pred = dis(data_real)  # use dis here
            loss_real = self.gan_loss(real_pred - torch.mean(fake_pred_d), True) * 0.5
            loss_real /= inter_step.float()  # multiple backwards and step once, thus mean
            loss_real.backward()

            fake_pred = dis(data_fake.detach())  # use dis here; 
            real_pred_d = real_pred.detach()
            loss_fake = self.gan_loss(fake_pred - torch.mean(real_pred_d), False) * 0.5  # for loss_fake, detach as constant
            loss_fake /= inter_step.float()  # multiple backwards and step once, thus mean
            loss_fake.backward()
        
            return (loss_real + loss_fake).item()

class _VGGFeatureExtractor(nn.Module):
    """VGG network for feature extraction.
    In this implementation, we allow users to choose whether use normalization
    in the input feature and the type of vgg network. Note that the pretrained
    path must fit the vgg type.
    Args:
        layer_name_list (list[str]): Forward function returns the corresponding
            features according to the layer_name_list.
            Example: {'relu1_1', 'relu2_1', 'relu3_1'}.
        vgg_type (str): Set the type of vgg network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize the input image. Importantly,
            the input feature must in the range [0, 1]. Default: True.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
        remove_pooling (bool): If true, the max pooling operations in VGG net
            will be removed. Default: False.
        pooling_stride (int): The stride of max pooling operation. Default: 2.
    """
    def __init__(
            self,
            layer_name_list,
            vgg_type='vgg19',
            use_input_norm=True,
            requires_grad=False,
            remove_pooling=False,
            pooling_stride=2
            ):
        super().__init__()

        NAMES = {
            'vgg11': [
                'conv1_1', 'relu1_1', 'pool1', 'conv2_1', 'relu2_1', 'pool2',
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3', 'conv4_1',
                'relu4_1', 'conv4_2', 'relu4_2', 'pool4', 'conv5_1', 'relu5_1',
                'conv5_2', 'relu5_2', 'pool5'
            ],
            'vgg13': [
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                'conv3_2', 'relu3_2', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
                'relu4_2', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5'
            ],
            'vgg16': [
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
                'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
                'pool5'
            ],
            'vgg19': [
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
                ]
            }

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm

        self.names = NAMES[vgg_type]
        if 'bn' in vgg_type:
            self.names = self.insert_bn(self.names)

        # only borrow layers that will be used to avoid unused params
        max_idx = 0
        for v in layer_name_list:
            idx = self.names.index(v)
            if idx > max_idx:
                max_idx = idx
        vgg_net = getattr(vgg, vgg_type)(pretrained=True) # get pre-trained vgg19
        features = vgg_net.features[:max_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(self.names, features):
            if 'pool' in k:
                # if remove_pooling is true, pooling operation will be removed
                if remove_pooling:
                    continue
                else:
                    # in some cases, we may want to change the default stride
                    modified_net[k] = nn.MaxPool2d(
                        kernel_size=2, stride=pooling_stride)
            else:
                modified_net[k] = v

        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def insert_bn(names):
        """Insert bn layer after each conv.
        Args:
            names (list): The list of layer names.
        Returns:
            list: The list of layer names with bn layers.
        """
        names_bn = []
        for name in names:
            names_bn.append(name)
            if 'conv' in name:
                position = name.replace('conv', '')
                names_bn.append('bn' + position)
        return names_bn

    def forward(self, x):
        if self.use_input_norm:  # normalize
            x = (x - self.mean) / self.std

        output = {}  # output desired feature maps
        for key, layer in self.vgg_net._modules.items():
            x = layer(x)
            if key in self.layer_name_list:
                output[key] = x.clone()
        return output
    
class VGGLoss(nn.Module):
    """Perceptual loss with commonly used style loss.
    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
    
    perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
        loss will be calculated and the loss will multiplied by the
        weight. Default: 1.0.
    style_weight (float): If `style_weight > 0`, the style loss will be
        calculated and the loss will multiplied by the weight.
        Default: 0.
    """
    def __init__(
            self,
            vgg_type='vgg19',
            layer_weights={'conv5_4': 1.0},
            use_input_norm=True,
            perceptual_weight=1.,
            style_weight=0.,
            ):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        
        self.layer_weights = layer_weights

        self.vgg = _VGGFeatureExtractor(
            layer_name_list=list(
                layer_weights.keys()
                ),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm
            )

        self.criterion = CharbonnierLoss()

    @staticmethod
    def _gram_mat(x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def forward(self, inp, ref):
        """
        inp (Tensor): Input tensor with shape (n, c, h, w).
        ref (Tensor): Ground-truth tensor with shape (n, c, h, w).
        
        Returns tensor.
        """
        # extract vgg features
        x_features = self.vgg(inp)
        gt_features = self.vgg(ref.detach())

        # calculate perceptual loss
        percep_loss = 0
        if self.perceptual_weight > 0:
            for k in x_features.keys():
                percep_loss += self.criterion(
                    x_features[k], gt_features[k]
                    ) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight

        # calculate style loss
        style_loss = 0
        if self.style_weight > 0:
            for k in x_features.keys():
                style_loss += self.criterion(
                    self._gram_mat(x_features[k]),
                    self._gram_mat(gt_features[k])
                    ) * self.layer_weights[k]
            style_loss *= self.style_weight

        return percep_loss + style_loss

# ===
# Optimizer
# ===

optim_lst = ['Adam']

def return_optimizer(name, params, opts):
    assert (name in optim_lst), '> Unsupported optimizer!'
    if name == 'Adam':
        return torch.optim.Adam(params, **opts)

# ===
# Scheduler
# ===

scheduler_lst = ['CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'MultiStepLR']

def return_scheduler(name, optim, opts):
    """
    CosineAnnealingLR:
        Real Cosine: Cosine down, Cosine up. Cosine cycle is 2*T_max.
        opts:
            T_max: half iters of one cycle.
            eta_min=0
            last_epoch=-1: resuming state. -1 indicates training from the stracth.
                Will automatically +1 after each .step()

    CosineAnnealingWarmRestarts:
        lr drops from lr_init to eta_min within T_i, and then repeat this process.
        Could be called after each epoch*/iter. Default to restart after each epoch/iter.
        opts:
            T_0: the first cycle before the first restart.
            T_mult=1: T_i = T_{i-1}*T_mult
            eta_min=0
            last_epoch=-1
        Example: the first cycle is 10 epochs, and then 20, 40, 80, ... epochs.
            Then T_0, T_mult should be 10 and 2.

    https://zhuanlan.zhihu.com/p/261134624
    
    MultiStepLR:
        lr drops by the factor gamma after each milestone.
        opts:
            milestones
            gamma=0.1
            last_epoch=-1
    """
    assert (name in scheduler_lst), '> Unsupported scheduler!'
    if name == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, **opts)
    elif name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, **opts)
    elif name == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optim, **opts)
