import torch
import torch.nn as nn
import pickle


class Shufflenet(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]

        self.base_mid_channel = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize,
                      stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(
                    inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


class Shuffle_Xception(nn.Module):

    def __init__(self, inp, oup, mid_channels, *, stride):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]

        self.base_mid_channel = mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        outputs = oup - inp

        branch_main = [
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3,
                      1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw
            nn.Conv2d(mid_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, 3,
                      1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]

        self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class ShuffleNetV2_OneShot(nn.Module):

    def __init__(self, input_size=224, n_class=1000, architecture=None, channels_scales=None):
        super(ShuffleNetV2_OneShot, self).__init__()

        assert input_size % 32 == 0
        assert architecture is not None and channels_scales is not None

        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                blockIndex = architecture[archIndex]
                base_mid_channels = outp // 2
                mid_channels = int(
                    base_mid_channels * channels_scales[archIndex])
                archIndex += 1
                if blockIndex == 0:
                    self.features.append(
                        Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride))
                elif blockIndex == 1:
                    self.features.append(
                        Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride))
                elif blockIndex == 2:
                    self.features.append(
                        Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride))
                elif blockIndex == 3:
                    self.features.append(
                        Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride))
                else:
                    raise NotImplementedError
                input_channel = output_channel

        assert archIndex == len(architecture)
        self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                input_channel, self.stage_out_channels[
                    -1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)

        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def get_flops(model, input_shape=(3, 224, 224)):
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        assert self.in_channels % self.groups == 0

        kernel_ops = self.kernel_size[0] * self.kernel_size[
            1] * (self.in_channels // self.groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement()

        flops = batch_size * weight_ops
        list_linear.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    input = torch.autograd.Variable(
        torch.rand(*input_shape).unsqueeze(0), requires_grad=True)
    out = model(input)

    total_flops = sum(sum(i) for i in [list_conv, list_linear])
    return total_flops

'''
def get_cand_flops(cand):
    model=ShuffleNetV2_OneShot(architecture=tuple(cand),channels_scales=(1.0,)*20)
    return get_flops(model)
'''

op_flops_dict = pickle.load(open('./data/op_flops_dict.pkl', 'rb'))
backbone_info = [  # inp, oup, img_h, img_w, stride
        (3,     16,     224,    224,    2),  # conv1
        (16,    64,     112,    112,    2),
        (64,    64,     56,     56,     1),
        (64,    64,     56,     56,     1),
        (64,    64,     56,     56,     1),
        (64,    160,    56,     56,     2),  # stride = 2
        (160,   160,    28,     28,     1),
        (160,   160,    28,     28,     1),
        (160,   160,    28,     28,     1),
        (160,   320,    28,     28,     2),  # stride = 2
        (320,   320,    14,     14,     1),
        (320,   320,    14,     14,     1),
        (320,   320,    14,     14,     1),
        (320,   320,    14,     14,     1),
        (320,   320,    14,     14,     1),
        (320,   320,    14,     14,     1),
        (320,   320,    14,     14,     1),
        (320,   640,    14,     14,     2),  # stride = 2
        (640,   640,    7,      7,      1),
        (640,   640,    7,      7,      1),
        (640,   640,    7,      7,      1),
        (640,   1000,   7,      7,      1),  # rest_operation
]
blocks_keys = [
    'shufflenet_3x3',
    'shufflenet_5x5',
    'shufflenet_7x7',
    'xception_3x3',
]


def get_cand_flops(cand):
    conv1_flops = op_flops_dict['conv1'][(3, 16, 224, 224, 2)]
    rest_flops = op_flops_dict['rest_operation'][(640, 1000, 7, 7, 1)]
    total_flops = conv1_flops + rest_flops
    for i in range(len(cand)):                                              # 遍历20个Choices Blocks, 每个CB的选择是根据cand这个随机生成的op id 序列
        op_ids = cand[i]                                                    # 获取第i层的id
        inp, oup, img_h, img_w, stride = backbone_info[i + 1]               # 获取backbone输入，输出，stride等信息
        key = blocks_keys[op_ids] + '_stride_' + str(stride)                # 选择选用的op的类型，根据其所在层位置选择stride，生成名字，进入op_flops_dict查表
        mid = int(oup // 2)
        mid = int(mid)
        total_flops += op_flops_dict[key][
            (inp, oup, mid, img_h, img_w, stride)]
    return total_flops


def main():
    for i in range(4):
        print(i, get_cand_flops((i,) * 20))

if __name__ == '__main__':
    main()
