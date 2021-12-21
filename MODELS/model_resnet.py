import os
import torch
import torch.nn as nn
from .iterative_normalization import IterNormRotation as cw_layer
from torch import Tensor


# define some layers that are used more frequently in the models
def conv3x3(in_planes: int, out_planes: int, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding used in DeepMir models"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation)


def maxpool_2x2():
    """2x2 maxpool layer used in DeepMir models"""
    return nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))


class DeepMir_v2(nn.Module):
    """Version of DeepMir model that includes BN layers + ReLU activations_test instead of dropout ones"""
    def __init__(self) -> None:
        super(DeepMir_v2, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(48, 48)
        self.relu2 = nn.ReLU()
        self.pool1 = maxpool_2x2()
        self.bn1 = nn.BatchNorm2d(48, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(48, 60)
        self.relu4 = nn.ReLU()
        self.conv4 = conv3x3(60, 60)
        self.relu5 = nn.ReLU()
        self.pool2 = maxpool_2x2()
        self.bn2 = nn.BatchNorm2d(60, track_running_stats=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv5 = conv3x3(60, 72)
        self.relu7 = nn.ReLU()
        self.conv6 = conv3x3(72, 72)
        self.relu8 = nn.ReLU()
        self.pool3 = maxpool_2x2()
        self.bn3 = nn.BatchNorm2d(72, track_running_stats=True)
        self.relu9 = nn.ReLU(inplace=True)

        self.linear1 = nn.Linear(2 * 12 * 72, 256)
        self.relu10 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool1(out)
        out = self.bn1(out)
        out = self.relu3(out)

        out = self.conv3(out)
        out = self.relu4(out)
        out = self.conv4(out)
        out = self.relu5(out)
        out = self.pool2(out)
        out = self.bn2(out)
        out = self.relu6(out)

        out = self.conv5(out)
        out = self.relu7(out)
        out = self.conv6(out)
        out = self.relu8(out)
        out = self.pool3(out)
        out = self.bn3(out)
        out = self.relu9(out)

        out = out.contiguous()
        x = out.view(out.size(0), -1)
        x = self.linear1(x)
        x = self.relu10(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class DeepMir_v2_BN(nn.Module):
    """DeepMir model used for pre-training and fine-tuning"""
    def __init__(self, args, model_file=None):
        super(DeepMir_v2_BN, self).__init__()
        self.model = DeepMir_v2()
        if model_file is not None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k, 'bw', 'bn'): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)


class DeepMir_v2_Transfer(nn.Module):
    """DeepMir model (without mean of max pool values) that is converted to a concept whitening model based on which
    BN layer is chosen to be converted"""
    def __init__(self, args, whitened_layers=None, model_file=None):
        super(DeepMir_v2_Transfer, self).__init__()
        self.model = DeepMir_v2()
        # load pre-trained weights into the model
        if model_file is not None:
            if not os.path.exists(model_file):
                raise Exception("checkpoint {} not found!".format(model_file))
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k, 'bw', 'bn'): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

        # convert one of the BN layers to a CW layer
        self.whitened_layers = whitened_layers
        for whitened_layer in whitened_layers:
            if whitened_layer == 1:
                self.model.bn1 = cw_layer(48, activation_mode=args.act_mode)
            elif whitened_layer == 2:
                self.model.bn2 = cw_layer(60, activation_mode=args.act_mode)
            elif whitened_layer == 3:
                self.model.bn3 = cw_layer(72, activation_mode=args.act_mode)

    def change_mode(self, mode):
        """
        Change the training mode
        mode = -1, no update for gradient matrix G
             = 0 to k-1, the column index of gradient matrix G that needs to be updated
        """
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 1:
                self.model.bn1.mode = mode
            elif whitened_layer == 2:
                self.model.bn2.mode = mode
            elif whitened_layer == 3:
                self.model.bn3.mode = mode

    def update_rotation_matrix(self):
        """
        update the rotation R using accumulated gradient G
        """
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 1:
                self.model.bn1.update_rotation_matrix()
            elif whitened_layer == 2:
                self.model.bn2.update_rotation_matrix()
            elif whitened_layer == 3:
                self.model.bn3.update_rotation_matrix()

    def forward(self, x):
        """forward pass of model, the pass is defined in iterative_normalization.py"""
        return self.model(x)


class DeepMir_vfinal(nn.Module):
    """Version of DeepMir model that includes BN layers + ReLU activations_test instead of dropout ones, as well as the
    mean max pool operation after the conv base. This model is the final model used in the experiments."""

    def __init__(self) -> None:
        super(DeepMir_vfinal, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(48, 48)
        self.relu2 = nn.ReLU()
        self.pool1 = maxpool_2x2()
        self.bn1 = nn.BatchNorm2d(48, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(48, 60)
        self.relu4 = nn.ReLU()
        self.conv4 = conv3x3(60, 60)
        self.relu5 = nn.ReLU()
        self.pool2 = maxpool_2x2()
        self.bn2 = nn.BatchNorm2d(60, track_running_stats=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv5 = conv3x3(60, 72)
        self.relu7 = nn.ReLU()
        self.conv6 = conv3x3(72, 72)
        self.relu8 = nn.ReLU()
        self.pool3 = maxpool_2x2()
        self.bn3 = nn.BatchNorm2d(72, track_running_stats=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))

        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(72, 2)  # note: if you want a 1-node output than the 2 has to changed into a 1

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool1(out)
        out = self.bn1(out)
        out = self.relu3(out)

        out = self.conv3(out)
        out = self.relu4(out)
        out = self.conv4(out)
        out = self.relu5(out)
        out = self.pool2(out)
        out = self.bn2(out)
        out = self.relu6(out)

        out = self.conv5(out)
        out = self.relu7(out)
        out = self.conv6(out)
        out = self.relu8(out)
        out = self.pool3(out)
        out = self.bn3(out)
        out = self.relu9(out)
        # extra pool layer and mean of the pool values
        out = self.pool4(out)
        out = torch.mean(out, dim=3)
        out = torch.flatten(out, 1)

        x = self.dropout(out)
        x = self.linear2(x)

        return x


class DeepMir_vfinal_BN(nn.Module):
    """DeepMir model that applies maxpool and mean operation after conv base. This model is used for pre-training and
         fine-tuning"""

    def __init__(self, args, model_file=None):
        super(DeepMir_vfinal_BN, self).__init__()
        self.model = DeepMir_vfinal()
        if model_file is not None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k, 'bw', 'bn'): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)


class DeepMir_vfinal_Transfer(nn.Module):
    """DeepMir model (with mean of max pool values) that is converted to a concept whitening model based on which
        BN layer is chosen to be converted"""
    def __init__(self, args, whitened_layers=None, model_file=None):
        super(DeepMir_vfinal_Transfer, self).__init__()
        self.model = DeepMir_vfinal()
        # load pre-training weights
        if model_file is not None:
            if not os.path.exists(model_file):
                raise Exception("checkpoint {} not found!".format(model_file))
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k, 'bw', 'bn'): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)

        # convert one of the BN layers to a CW one
        self.whitened_layers = whitened_layers
        for whitened_layer in whitened_layers:
            if whitened_layer == 1:
                self.model.bn1 = cw_layer(48, activation_mode=args.act_mode)
            elif whitened_layer == 2:
                self.model.bn2 = cw_layer(60, activation_mode=args.act_mode)
            elif whitened_layer == 3:
                self.model.bn3 = cw_layer(72, activation_mode=args.act_mode)

    def change_mode(self, mode):
        """
        Change the training mode
        mode = -1, no update for gradient matrix G
             = 0 to k-1, the column index of gradient matrix G that needs to be updated
        """
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 1:
                self.model.bn1.mode = mode
            elif whitened_layer == 2:
                self.model.bn2.mode = mode
            elif whitened_layer == 3:
                self.model.bn3.mode = mode

    def update_rotation_matrix(self):
        """
        update the rotation R using accumulated gradient G
        """
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 1:
                self.model.bn1.update_rotation_matrix()
            elif whitened_layer == 2:
                self.model.bn2.update_rotation_matrix()
            elif whitened_layer == 3:
                self.model.bn3.update_rotation_matrix()

    def forward(self, x):
        """forward pass of model, the pass is defined in iterative_normalization.py"""
        return self.model(x)
