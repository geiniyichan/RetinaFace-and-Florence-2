import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),  # 修复了这里
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, padding, expand_ratio=1):
        super(DepthwiseSeparableConv, self).__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # 点卷积
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # 深度卷积
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # 点卷积（投影）
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # 第一次Ghost卷积
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # 深度卷积
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False)
            if stride == 2 else nn.Sequential(),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True) if stride == 2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # 第二次Ghost卷积（无ReLU）
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        self.shortcut = nn.Sequential()
        if stride == 1 and inp != oup:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x) if self.shortcut else self.conv(x)


class GhostNet(nn.Module):
    def __init__(self, width_mult=1.0, pretrained=False):
        super(GhostNet, self).__init__()
        cfgs = [
            # k, t, c, SE, s 
            # stage1
            [[3,  16,  16, 0, 1]],
            # stage2
            [[3,  48,  24, 0, 2]],
            [[3,  72,  24, 0, 1]],
            # stage3
            [[5,  72,  40, 1, 2]],
            [[5, 120,  40, 1, 1]],
            # stage4
            [[3, 240,  80, 0, 2]],
            [[3, 200,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 184,  80, 0, 1],
             [3, 480, 112, 1, 1],
             [3, 672, 112, 1, 1]
            ],
            # stage5
            [[5, 672, 160, 1, 2]],
            [[5, 960, 160, 0, 1],
             [5, 960, 160, 1, 1],
             [5, 960, 160, 0, 1],
             [5, 960, 160, 1, 1]
            ]
        ]

        # 构建第一个卷积层
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU6(inplace=True)
        input_channel = output_channel

        # 构建瓶颈层
        stages = []
        block = GhostBottleneck
        for cfg in cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width_mult, 4)
                hidden_channel = _make_divisible(exp_size * width_mult, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s, se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        # 为RetinaFace添加的层，用于特征提取
        self.layer1 = stages[0]  # stride=2
        self.layer2 = stages[1]  # stride=4
        self.layer3 = stages[2]  # stride=8
        self.layer4 = stages[3]  # stride=16
        self.layer5 = stages[4]  # stride=32

        # 初始化权重
        self._initialize_weights()
        if pretrained:
            # 加载预训练权重的逻辑在RetinaFace类中处理
            pass

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x2)
        x = self.layer4(x3)
        x = self.layer5(x4)
        
        # 返回多个尺度的特征图
        return {
            'layer2': x3,
            'layer3': x4,
            'layer4': x5,
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                m.bias.data.zero_()