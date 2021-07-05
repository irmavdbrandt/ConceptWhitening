import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torch.nn import init
from .iterative_normalization import IterNormRotation as cw_layer
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor


class ResidualNetTransfer(nn.Module):
    def __init__(self, num_classes, args, whitened_layers=None, arch='resnet18', layers=None, model_file=None):

        super(ResidualNetTransfer, self).__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        self.layers = layers
        self.model = models.__dict__[arch](num_classes=num_classes)
        if model_file is not None:
            if not os.path.exists(model_file):
                raise Exception("checkpoint {} not found!".format(model_file))
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            # state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            # if using my own pretrained resnet model, use this below
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}

            state_dict = {str.replace(k, 'bw', 'bn'): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

        self.whitened_layers = whitened_layers

        for whitened_layer in whitened_layers:
            if whitened_layer <= layers[0]:
                self.model.layer1[whitened_layer - 1].bn1 = cw_layer(64, activation_mode=args.act_mode)
            elif whitened_layer <= layers[0] + layers[1]:
                self.model.layer2[whitened_layer - layers[0] - 1].bn1 = cw_layer(128, activation_mode=args.act_mode)
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.model.layer3[whitened_layer - layers[0] - layers[1] - 1].bn1 = cw_layer(256,
                                                                                             activation_mode=args.act_mode)
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.model.layer4[whitened_layer - layers[0] - layers[1] - layers[2] - 1].bn1 = cw_layer(512,
                                                                                                         activation_mode=args.act_mode)

    def change_mode(self, mode):
        """
        Change the training mode
        mode = -1, no update for gradient matrix G
             = 0 to k-1, the column index of gradient matrix G that needs to be updated
        """
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            if whitened_layer <= layers[0]:
                self.model.layer1[whitened_layer - 1].bn1.mode = mode
            elif whitened_layer <= layers[0] + layers[1]:
                self.model.layer2[whitened_layer - layers[0] - 1].bn1.mode = mode
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.model.layer3[whitened_layer - layers[0] - layers[1] - 1].bn1.mode = mode
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.model.layer4[whitened_layer - layers[0] - layers[1] - layers[2] - 1].bn1.mode = mode

    def update_rotation_matrix(self):
        """
        update the rotation R using accumulated gradient G
        """
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            if whitened_layer <= layers[0]:
                self.model.layer1[whitened_layer - 1].bn1.update_rotation_matrix()
            elif whitened_layer <= layers[0] + layers[1]:
                self.model.layer2[whitened_layer - layers[0] - 1].bn1.update_rotation_matrix()
            elif whitened_layer <= layers[0] + layers[1] + layers[2]:
                self.model.layer3[whitened_layer - layers[0] - layers[1] - 1].bn1.update_rotation_matrix()
            elif whitened_layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                self.model.layer4[whitened_layer - layers[0] - layers[1] - layers[2] - 1].bn1.update_rotation_matrix()

    def forward(self, x):
        return self.model(x)


class DenseNetTransfer(nn.Module):
    def __init__(self, num_classes, args, whitened_layers=None, arch='densenet161', model_file=None):

        super(DenseNetTransfer, self).__init__()
        self.model = models.__dict__[arch](num_classes=num_classes)
        if model_file is not None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            import re

            def repl(matchobj):
                return matchobj.group(0)[1:]

            state_dict = {re.sub('\.\d+\.', repl, str.replace(k, 'module.', '')): v for k, v in
                          checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)

        self.whitened_layers = whitened_layers
        for whitened_layer in whitened_layers:
            if whitened_layer == 1:
                self.model.features.norm0 = cw_layer(64, activation_mode=args.act_mode)
            elif whitened_layer == 2:
                self.model.features.transition1.norm = cw_layer(384, activation_mode=args.act_mode)
            elif whitened_layer == 3:
                self.model.features.transition2.norm = cw_layer(768, activation_mode=args.act_mode)
            elif whitened_layer == 4:
                self.model.features.transition3.norm = cw_layer(2112, activation_mode=args.act_mode)
            elif whitened_layer == 5:
                self.model.features.norm5 = cw_layer(2208, activation_mode=args.act_mode)

    def change_mode(self, mode):
        """
        Change the training mode
        mode = -1, no update for gradient matrix G
             = 0 to k-1, the column index of gradient matrix G that needs to be updated
        """
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 1:
                self.model.features.norm0.mode = mode
            elif whitened_layer == 2:
                self.model.features.transition1.norm.mode = mode
            elif whitened_layer == 3:
                self.model.features.transition2.norm.mode = mode
            elif whitened_layer == 4:
                self.model.features.transition3.norm.mode = mode
            elif whitened_layer == 5:
                self.model.features.norm5.mode = mode

    def update_rotation_matrix(self):
        """
        update the rotation R using accumulated gradient G
        """
        for whitened_layer in self.whitened_layers:
            if whitened_layer == 1:
                self.model.features.norm0.update_rotation_matrix()
            elif whitened_layer == 2:
                self.model.features.transition1.norm.update_rotation_matrix()
            elif whitened_layer == 3:
                self.model.features.transition2.norm.update_rotation_matrix()
            elif whitened_layer == 4:
                self.model.features.transition3.norm.update_rotation_matrix()
            elif whitened_layer == 5:
                self.model.features.norm5.update_rotation_matrix()

    def forward(self, x):
        return self.model(x)


class VGGBNTransfer(nn.Module):
    def __init__(self, num_classes, args, whitened_layers=None, arch='vgg16_bn', model_file=None):
        super(VGGBNTransfer, self).__init__()
        self.model = models.__dict__[arch](num_classes=num_classes)
        if model_file is not None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)

        self.whitened_layers = whitened_layers
        self.layers = [1, 4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]
        for whitened_layer in whitened_layers:
            whitened_layer -= 1
            if whitened_layer in range(0, 2):
                channel = 64
            elif whitened_layer in range(2, 4):
                channel = 128
            elif whitened_layer in range(4, 7):
                channel = 256
            else:
                channel = 512
            self.model.features[self.layers[whitened_layer]] = cw_layer(channel, activation_mode=args.act_mode)

    def change_mode(self, mode):
        """
        Change the training mode
        mode = -1, no update for gradient matrix G
             = 0 to k-1, the column index of gradient matrix G that needs to be updated
        """
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            self.model.features[layers[whitened_layer - 1]].mode = mode

    def update_rotation_matrix(self):
        """
        update the rotation R using accumulated gradient G
        """
        layers = self.layers
        for whitened_layer in self.whitened_layers:
            self.model.features[layers[whitened_layer - 1]].update_rotation_matrix()

    def forward(self, x):
        return self.model(x)


class DeepMirResNetTransfer(nn.Module):
    def __init__(self, args, whitened_layers=None, model_file=None):
        super(DeepMirResNetTransfer, self).__init__()
        self.model = DeepMirResNet()
        if model_file is not None:
            if not os.path.exists(model_file):
                raise Exception("checkpoint {} not found!".format(model_file))
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            # if using my own pretrained resnet model, use this below
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k, 'bw', 'bn'): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)

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
        return self.model(x)


class DeepMirResNetTransferv2(nn.Module):
    def __init__(self, args, whitened_layers=None, model_file=None):
        super(DeepMirResNetTransferv2, self).__init__()
        self.model = DeepMirResNetv2()
        if model_file is not None:
            if not os.path.exists(model_file):
                raise Exception("checkpoint {} not found!".format(model_file))
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k, 'bw', 'bn'): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict)

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
        return self.model(x)


# I am going to change the dropout layers to BN
# convert model from keras to pytorch
class DeepMir(nn.Module):
    def __init__(self):
        super(DeepMir, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3, 3), stride=(1, 1)),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(3, 3), padding=(1, 1)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      nn.BatchNorm2d(48),

                                      nn.Conv2d(in_channels=48, out_channels=60, kernel_size=(3, 3), padding=(1, 1)),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=60, out_channels=60, kernel_size=(3, 3), padding=(1, 1)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      nn.BatchNorm2d(60),

                                      nn.Conv2d(in_channels=60, out_channels=72, kernel_size=(3, 3), padding=(1, 1)),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=72, out_channels=72, kernel_size=(3, 3), padding=(1, 1)),
                                      nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                                      nn.BatchNorm2d(72))
        self.classifier = nn.Sequential(nn.Linear(2 * 12 * 72, 256),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(256, 2),
                                        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class DeepMirTransfer(nn.Module):
    def __init__(self, args, whitened_layers=None, model_file=None):
        super(DeepMirTransfer, self).__init__()
        self.model = DeepMir()
        if model_file is not None:
            if not os.path.exists(model_file):
                raise Exception(f"checkpoint {model_file} not found!")
            # here I load the weights, and other stuff related to pretraining
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']  # best_prec1 is the acc
            print('best accuracy from loaded model', checkpoint['best_prec1'])
            new_state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(new_state_dict)

        self.whitened_layers = whitened_layers
        for whitened_layer in whitened_layers:
            whitened_layer -= 1
            # note that the other layers such as dropout, activation.. are also counted
            if whitened_layer in range(0, 6):
                channel = 48
            elif whitened_layer in range(6, 12):
                channel = 60
            else:
                channel = 72
            self.model.features[whitened_layer] = cw_layer(channel, activation_mode=args.act_mode)

    def change_mode(self, mode):
        """
        Change the training mode
        mode = -1, no update for gradient matrix G
             = 0 to k-1, the column index of gradient matrix G that needs to be updated
        """
        # layers = self.layers
        for whitened_layer in self.whitened_layers:
            self.model.features[whitened_layer - 1].mode = mode

    def update_rotation_matrix(self):
        """
        update the rotation R using accumulated gradient G
        """
        # layers = self.layers
        for whitened_layer in self.whitened_layers:
            self.model.features[whitened_layer - 1].update_rotation_matrix()

    def forward(self, x):
        return self.model(x)


class ResidualNetBN(nn.Module):
    def __init__(self, num_classes, args, arch='resnet18', layers=None, model_file=None):

        super(ResidualNetBN, self).__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        self.layers = layers
        self.model = models.__dict__[arch](num_classes=num_classes)
        if model_file is not None:
            if not os.path.exists(model_file):
                raise Exception("checkpoint {} not found!".format(model_file))
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            print(args.best_prec1)
            # state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}

            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)


class DenseNetBN(nn.Module):
    def __init__(self, num_classes, args, arch='densenet161', model_file=None):
        super(DenseNetBN, self).__init__()
        self.model = models.__dict__[arch](num_classes=num_classes)
        if model_file is not None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            import re

            def repl(matchobj):
                return matchobj.group(0)[1:]

            state_dict = {re.sub('\.\d+\.', repl, str.replace(k, 'module.', '')): v for k, v in
                          checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)


class VGGBN(nn.Module):
    def __init__(self, args, arch='vgg16_bn', model_file=None):
        super(VGGBN, self).__init__()
        self.model = models.__dict__[arch](num_classes=365)
        if model_file == 'vgg16_bn_places365.pt':
            state_dict = torch.load(model_file, map_location='cpu')
            args.start_epoch = 0
            args.best_prec1 = 0
            d = self.model.state_dict()
            new_state_dict = {k: state_dict[k] if k in state_dict.keys() else d[k] for k in d.keys()}
            self.model.load_state_dict(new_state_dict)
        elif model_file is not None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)


class DeepMirBN(nn.Module):
    def __init__(self, args, model_file=None):
        super(DeepMirBN, self).__init__()
        self.model = DeepMir()
        # add here the pretraining weights from mirbase
        if model_file == 'deepmir_pretrain.pth':
            state_dict = torch.load(model_file, map_location='cpu')['state_dict']
            args.start_epoch = 0
            args.best_prec1 = 0
            d = self.model.state_dict()
            new_state_dict = {k: state_dict[k] if k in state_dict.keys() else d[k] for k in d.keys()}
            self.model.load_state_dict(new_state_dict)
        elif model_file is not None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)


def conv3x3(in_planes: int, out_planes: int, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation)


class DeepMirResNet(nn.Module):
    def __init__(self) -> None:
        super(DeepMirResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(48, 48)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(48, track_running_stats=True)

        self.conv3 = conv3x3(48, 60)
        self.relu3 = nn.ReLU()
        self.conv4 = conv3x3(60, 60)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(60, track_running_stats=True)

        self.conv5 = conv3x3(60, 72)
        self.relu5 = nn.ReLU()
        self.conv6 = conv3x3(72, 72)
        self.relu6 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(72, track_running_stats=True)

        self.linear1 = nn.Linear(2 * 12 * 72, 256)
        self.relu7 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool1(out)
        out = self.bn1(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.pool2(out)
        out = self.bn2(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.conv6(out)
        out = self.relu6(out)
        out = self.pool3(out)
        out = self.bn3(out)

        out = out.contiguous()
        x = out.view(out.size(0), -1)
        x = self.linear1(x)
        x = self.relu7(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class DeepMirResNetBN(nn.Module):
    def __init__(self, args, model_file=None):
        super(DeepMirResNetBN, self).__init__()
        self.model = DeepMirResNet()
        # add here the pretraining weights from mirbase
        if model_file is not None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}

            state_dict = {str.replace(k, 'bw', 'bn'): v for k, v in state_dict.items()}

            # randomly initialize bn mean and var
            # TODO: pretrain again to get running mean and running var
            bn1 = torch.empty(48)
            bn2 = torch.empty(60)
            bn3 = torch.empty(72)
            print(bn1.shape, bn2.shape, bn3.shape)
            state_dict['bn1.running_mean'] = nn.init.zeros_(bn1)
            state_dict['bn1.running_var'] = nn.init.zeros_(bn1)
            state_dict['bn2.running_mean'] = nn.init.zeros_(bn2)
            state_dict['bn2.running_var'] = nn.init.zeros_(bn2)
            state_dict['bn3.running_mean'] = nn.init.zeros_(bn3)
            state_dict['bn3.running_var'] = nn.init.zeros_(bn3)
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)


class DeepMirResNetv2(nn.Module):
    def __init__(self) -> None:
        super(DeepMirResNetv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=3, stride=(1, 1))
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(48, 48)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(48, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(48, 60)
        self.relu4 = nn.ReLU()
        self.conv4 = conv3x3(60, 60)
        self.relu5 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(60, track_running_stats=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv5 = conv3x3(60, 72)
        self.relu7 = nn.ReLU()
        self.conv6 = conv3x3(72, 72)
        self.relu8 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
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


class DeepMirResNetBNv2(nn.Module):
    def __init__(self, args, model_file=None):
        super(DeepMirResNetBNv2, self).__init__()
        self.model = DeepMirResNetv2()
        if model_file is not None:
            checkpoint = torch.load(model_file, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.best_prec1 = checkpoint['best_prec1']
            state_dict = {str.replace(k, 'module.model.', ''): v for k, v in checkpoint['state_dict'].items()}

            state_dict = {str.replace(k, 'bw', 'bn'): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)
