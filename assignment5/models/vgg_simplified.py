import torch
import torch.nn as nn
import math


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        B, F, _, _ = x.shape
        return x.view(B, F)

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class Vgg(nn.Module):
    def __init__(self, fc_layer=512, classes=10):
        super(Vgg, self).__init__()
        """ Initialize VGG simplified Module
        Args: 
            fc_layer: input feature number for the last fully MLP block
            classes: number of image classes
        """
        self.fc_layer = fc_layer
        self.classes = classes

        # todo: construct the simplified VGG network blocks
        # input shape: [bs, 3, 32, 32]
        # layers and output feature shape for each block:
        # # conv_block1 (Conv2d, ReLU, MaxPool2d) --> [bs, 64, 16, 16]
        # # conv_block2 (Conv2d, ReLU, MaxPool2d) --> [bs, 128, 8, 8]
        # # conv_block3 (Conv2d, ReLU, MaxPool2d) --> [bs, 256, 4, 4]
        # # conv_block4 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 2, 2]
        # # conv_block5 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 1, 1]
        # # classifier (Linear, ReLU, Dropout2d, Linear) --> [bs, 10] (final output)

        # hint: stack layers in each block with nn.Sequential, e.x.:
        # # self.conv_block1 = nn.Sequential(
        # #     layer1,
        # #     layer2,
        # #     layer3,
        # #     ...)

        tuple1, tuple2, tuple3 = (1,1), (2,2), (3,3)
        sizes = [3, 64, 128, 256, 512, 512, 10]
        convrelupool = lambda args : [ nn.Conv2d(args[0], args[1], tuple3, tuple1, tuple1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(tuple2)]
        self.model = nn.Sequential(
            *convrelupool([sizes[0], sizes[1]]),
            *convrelupool([sizes[1], sizes[2]]),
            *convrelupool([sizes[2], sizes[3]]),
            *convrelupool([sizes[3], sizes[4]]),
            *convrelupool([sizes[4], sizes[5]]),
            View(),
            nn.Linear(sizes[5], sizes[5]),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(sizes[5], sizes[6]),
        )
        print('Simplified VGG Model: ', self.model)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        """
        :param x: input image batch tensor, [bs, 3, 32, 32]
        :return: score: predicted score for each class (10 classes in total), [bs, 10]
        """
        # todo
        return self.model(x)

