# models/shufflenetv2.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, width_mult=1.0, out_stages=(2, 3, 4), frozen_stages=-1, norm_cfg=dict(type='BN')):
        super(ShuffleNetV2, self).__init__()
        self.width_mult = width_mult
        self.out_stages = out_stages
        self.frozen_stages = frozen_stages
        self.norm_cfg = norm_cfg
        
        # 输入通道数
        input_channels = 24
        # 各阶段输出通道数
        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [
            -1,  # 0 不使用
            24,  # 1 阶段1
            48,  # 2 阶段2
            96,  # 3 阶段3
            192, # 4 阶段4
            1024 # 5 阶段5
        ]

        # 调整宽度
        if width_mult != 1.0:
            # 确保第一阶段通道数为8的倍数
            self.stage_out_channels[1] = 24
            # 计算新的通道数
            for i in range(2, len(self.stage_out_channels)):
                self.stage_out_channels[i] = int(self.stage_out_channels[i] * width_mult)
            # 调整最后阶段的通道数
            self.stage_out_channels[-1] = int(self.stage_out_channels[-1] * width_mult) if width_mult > 1.0 else self.stage_out_channels[-1]

        # 第一阶段
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建阶段2-4
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for i, (name, repeats, output_channels) in enumerate(
                zip(stage_names, self.stage_repeats, self.stage_out_channels[2:])):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        # 冻结指定阶段
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        
        for i in range(1, self.frozen_stages + 1):
            if i >= 2 and i <= 4:
                m = getattr(self, 'stage{}'.format(i))
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        
        outs = []
        for i in range(2, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                outs.append(x)
        
        return tuple(outs)

    def train(self, mode=True):
        super(ShuffleNetV2, self).train(mode)
        self._freeze_stages()