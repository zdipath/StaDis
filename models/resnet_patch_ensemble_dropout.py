import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torch.nn import Parameter
import torch.nn.init as init



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            
        self.block = block
        self.groups = groups
        self.base_width = width_per_group

        # 224,224,3 -> 112,112,64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 112,112,64 -> 56,56,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 56,56,64 -> 56,56,256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 56,56,256 -> 28,28,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        # 28,28,512 -> 14,14,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # 14,14,1024 -> 7,7,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 7,7,2048 -> 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2048 -> num_classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Conv_block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #y = self.fc(out)
        return out, out_list
    
    def forward_threshold(self, x, threshold=1e6):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.clip(max=threshold)
        x = torch.flatten(x, 1)#输出为B*C
        x = self.fc(x)
        return x
    def forward_fea(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)#输出为B*C
        return x
    

    def forward(self, x, fea_if = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)#输出为B*C
        if fea_if:
            fea = x
        x = self.fc(x)
        if fea_if:
            return x,fea
        return x
    
    def freeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = True

class ResNetmhead(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetmhead, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            
        self.block = block
        self.groups = groups
        self.base_width = width_per_group

        # 224,224,3 -> 112,112,64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 112,112,64 -> 56,56,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 56,56,64 -> 56,56,256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 56,56,256 -> 28,28,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        # 28,28,512 -> 14,14,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # 14,14,1024 -> 7,7,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 7,7,2048 -> 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2048 -> num_classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)
        self.fc5 = nn.Linear(512 * block.expansion, num_classes)

        # fc1: 使用 Xavier 初始化
        init.xavier_uniform_(self.fc.weight)  # Xavier 初始化权重
        if self.fc.bias is not None:
            init.zeros_(self.fc.bias)  # 偏置初始化为0

        # fc2: 使用 Kaiming 初始化
        init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 初始化权重
        if self.fc2.bias is not None:
            init.zeros_(self.fc2.bias)  # 偏置初始化为0

        # fc3: 使用标准正态分布初始化
        init.normal_(self.fc3.weight, mean=0, std=0.01)  # 正态分布初始化权重
        if self.fc3.bias is not None:
            init.zeros_(self.fc3.bias)  # 偏置初始化为0

        # fc4: 使用均匀分布初始化
        init.uniform_(self.fc4.weight, a=-0.1, b=0.1)  # 均匀分布初始化权重
        if self.fc4.bias is not None:
            init.zeros_(self.fc4.bias)  # 偏置初始化为0

        # fc5: 使用常数初始化
        init.constant_(self.fc5.weight, 0.1)  # 权重初始化为0.1
        if self.fc5.bias is not None:
            init.zeros_(self.fc5.bias)  # 偏置初始化为0









        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Conv_block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #y = self.fc(out)
        return out, out_list
    
    def forward_threshold(self, x, threshold=1e6):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.clip(max=threshold)
        x = torch.flatten(x, 1)#输出为B*C
        x = self.fc(x)
        return x
    def forward_fea(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)#输出为B*C
        return x
    

    def forward(self, x, fea_if = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)#输出为B*C
        if fea_if:
            fea = x
        x1 = self.fc(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x5 = self.fc5(x)
        if fea_if:
            return x1,x2,x3,x4,x5,fea
        return x1,x2,x3,x4,x5
    
    def freeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = True


class ResNetdropout(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetdropout, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            
        self.block = block
        self.groups = groups
        self.base_width = width_per_group

        # 224,224,3 -> 112,112,64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # 112,112,64 -> 56,56,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 56,56,64 -> 56,56,256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 56,56,256 -> 28,28,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])

        # 28,28,512 -> 14,14,1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # 14,14,1024 -> 7,7,2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 7,7,2048 -> 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 2048 -> num_classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.dropout0 = nn.Dropout(0.1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # Conv_block
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # identity_block
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #y = self.fc(out)
        return out, out_list
    
    def forward_threshold(self, x, threshold=1e6):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.clip(max=threshold)
        x = torch.flatten(x, 1)#输出为B*C
        x = self.fc(x)
        return x
    def forward_fea(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)#输出为B*C
        return x
    

    def forward(self, x, fea_if = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout0(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.layer4(x)
        x = self.dropout4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)#输出为B*C
        if fea_if:
            fea = x
        x = self.fc(x)
        if fea_if:
            return x,fea
        return x
    
    def freeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = False

    def Unfreeze_backbone(self):
        backbone = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        for module in backbone:
            for param in module.parameters():
                param.requires_grad = True

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, momentum_bn=0.1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, momentum=momentum_bn)

        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, momentum=momentum_bn)

        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, momentum=momentum_bn)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




def resnet50Mhead(state_dict,pretrained=False, progress=True, num_classes=1000):
    model = ResNetmhead(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        #state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir='./model_data',progress=progress)
        if state_dict['fc.weight'].shape[0]==1000:
            model.load_state_dict(state_dict, strict=False)
        if num_classes!=1000:
            model.fc = nn.Linear(512 * model.block.expansion, num_classes)
            model.fc2 = nn.Linear(512 * model.block.expansion, num_classes)
            model.fc3 = nn.Linear(512 * model.block.expansion, num_classes)
            model.fc4 = nn.Linear(512 * model.block.expansion, num_classes)
            model.fc5 = nn.Linear(512 * model.block.expansion, num_classes)

            # fc1: 使用 Xavier 初始化
            init.xavier_uniform_(model.fc.weight)  # Xavier 初始化权重
            if model.fc.bias is not None:
                init.zeros_(model.fc.bias)  # 偏置初始化为0

            # fc2: 使用 Kaiming 初始化
            init.kaiming_normal_(model.fc2.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 初始化权重
            if model.fc2.bias is not None:
                init.zeros_(model.fc2.bias)  # 偏置初始化为0

            # fc3: 使用标准正态分布初始化
            init.normal_(model.fc3.weight, mean=0, std=0.01)  # 正态分布初始化权重
            if model.fc3.bias is not None:
                init.zeros_(model.fc3.bias)  # 偏置初始化为0

            # fc4: 使用均匀分布初始化
            init.uniform_(model.fc4.weight, a=-0.1, b=0.1)  # 均匀分布初始化权重
            if model.fc4.bias is not None:
                init.zeros_(model.fc4.bias)  # 偏置初始化为0

            # fc5: 使用常数初始化
            init.constant_(model.fc5.weight, 0.1)  # 权重初始化为0.1
            if model.fc5.bias is not None:
                init.zeros_(model.fc5.bias)  # 偏置初始化为0
        if state_dict['fc.weight'].shape[0]!=1000:
            model.load_state_dict(state_dict)
        
    
    return model

def resnet50dropout(state_dict,pretrained=False, progress=True, num_classes=1000):
    model = ResNetdropout(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        #state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir='./model_data',progress=progress)
        if state_dict['fc.weight'].shape[0]==1000:
            model.load_state_dict(state_dict)
        if num_classes!=1000:
            model.fc = nn.Linear(512 * model.block.expansion, num_classes)
        if state_dict['fc.weight'].shape[0]!=1000:
            model.load_state_dict(state_dict)
        
    
    return model
def resnet50(state_dict,pretrained=False, progress=True, num_classes=1000):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        #state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir='./model_data',progress=progress)
        if state_dict['fc.weight'].shape[0]==1000:
            model.load_state_dict(state_dict)
        if num_classes!=1000:
            model.fc = nn.Linear(512 * model.block.expansion, num_classes)
        if state_dict['fc.weight'].shape[0]!=1000:
            model.load_state_dict(state_dict)
        
    
    return model


def retccl_resnet50(pretrained=False, progress=True,method = None, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if method == 'MCdropout':
        return ResNet_retccldropout(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif method == 'M-heads':
        return ResNet_retcclmhead(Bottleneck, [3, 4, 6, 3], **kwargs)
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

class ResNet_retcclmhead(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, two_branch=False, mlp=False, normlinear=False,
                 momentum_bn=0.1, attention=False, attention_layers=3, return_attn=False):
        super(ResNet_retcclmhead, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.return_attn = return_attn
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.two_branch = two_branch
        self.momentum_bn = momentum_bn
        self.mlp = mlp
        linear = NormedLinear if normlinear else nn.Linear

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes, momentum=momentum_bn)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])


        if attention:
            self.att_branch = self._make_layer(block, 512, attention_layers, 1, attention=True)
        else:
            self.att_branch = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.mlp:
            if self.two_branch:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512 * block.expansion),
                    nn.ReLU()
                ) 
                self.instDis = linear(512 * block.expansion, num_classes)
                self.groupDis = linear(512 * block.expansion, num_classes)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512 * block.expansion),
                    nn.ReLU(),
                    linear(512 * block.expansion, num_classes)
                ) 
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            if self.two_branch:
                self.groupDis = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc2 = nn.Linear(512 * block.expansion, num_classes)
        self.fc3 = nn.Linear(512 * block.expansion, num_classes)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)
        self.fc5 = nn.Linear(512 * block.expansion, num_classes)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, attention=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, momentum=self.momentum_bn),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, momentum_bn=self.momentum_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, momentum_bn=self.momentum_bn))

        if attention:
            layers.append(nn.Sequential(
                conv1x1(self.inplanes, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                conv1x1(128, 1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.att_branch is not None:
            att_map = self.att_branch(x)
            x = x + att_map * x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.mlp and self.two_branch:
            x = self.fc(x)
            x1 = self.instDis(x)
            x2 = self.groupDis(x)
            return [x1, x2]
        else:
            x1 = self.fc(x)
            x22 = self.fc2(x)
            x33 = self.fc3(x)
            x44 = self.fc4(x)
            x55 = self.fc5(x)

            if self.two_branch:
                x2 = self.groupDis(x)
                return [x1, x2]
            return x1, x22, x33, x44, x55
        

    def forward_fea(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward_threshold(self, x, threshold=1e6):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.att_branch is not None:
            att_map = self.att_branch(x)
            x = x + att_map * x

        x = self.avgpool(x)
        x = x.clip(max=threshold)
        x = torch.flatten(x, 1)#输出为B*C
        if self.mlp and self.two_branch:
            x = self.fc(x)
            x1 = self.instDis(x)
            x2 = self.groupDis(x)
            return [x1, x2]
        else:
            x1 = self.fc(x)
            if self.two_branch:
                x2 = self.groupDis(x)
                return [x1, x2]
            return x1
        

    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #y = self.fc(out)
        return out, out_list


class ResNet_retccldropout(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, two_branch=False, mlp=False, normlinear=False,
                 momentum_bn=0.1, attention=False, attention_layers=3, return_attn=False):
        super(ResNet_retccldropout, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.return_attn = return_attn
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.dropout0 = nn.Dropout(0.1)
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.two_branch = two_branch
        self.momentum_bn = momentum_bn
        self.mlp = mlp
        linear = NormedLinear if normlinear else nn.Linear

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes, momentum=momentum_bn)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        if attention:
            self.att_branch = self._make_layer(block, 512, attention_layers, 1, attention=True)
        else:
            self.att_branch = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.mlp:
            if self.two_branch:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512 * block.expansion),
                    nn.ReLU()
                ) 
                self.instDis = linear(512 * block.expansion, num_classes)
                self.groupDis = linear(512 * block.expansion, num_classes)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512 * block.expansion),
                    nn.ReLU(),
                    linear(512 * block.expansion, num_classes)
                ) 
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            if self.two_branch:
                self.groupDis = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, attention=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, momentum=self.momentum_bn),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, momentum_bn=self.momentum_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, momentum_bn=self.momentum_bn))

        if attention:
            layers.append(nn.Sequential(
                conv1x1(self.inplanes, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                conv1x1(128, 1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout0(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.dropout3(x)
        x = self.layer4(x)
        x = self.dropout4(x)
        if self.att_branch is not None:
            att_map = self.att_branch(x)
            x = x + att_map * x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.mlp and self.two_branch:
            x = self.fc(x)
            x1 = self.instDis(x)
            x2 = self.groupDis(x)
            return [x1, x2]
        else:
            x1 = self.fc(x)
            if self.two_branch:
                x2 = self.groupDis(x)
                return [x1, x2]
            return x1
        

    def forward_fea(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward_threshold(self, x, threshold=1e6):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.att_branch is not None:
            att_map = self.att_branch(x)
            x = x + att_map * x

        x = self.avgpool(x)
        x = x.clip(max=threshold)
        x = torch.flatten(x, 1)#输出为B*C
        if self.mlp and self.two_branch:
            x = self.fc(x)
            x1 = self.instDis(x)
            x2 = self.groupDis(x)
            return [x1, x2]
        else:
            x1 = self.fc(x)
            if self.two_branch:
                x2 = self.groupDis(x)
                return [x1, x2]
            return x1
        

    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #y = self.fc(out)
        return out, out_list


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_retccl(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

class ResNet_retccl(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, two_branch=False, mlp=False, normlinear=False,
                 momentum_bn=0.1, attention=False, attention_layers=3, return_attn=False):
        super(ResNet_retccl, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.return_attn = return_attn
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.two_branch = two_branch
        self.momentum_bn = momentum_bn
        self.mlp = mlp
        linear = NormedLinear if normlinear else nn.Linear

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes, momentum=momentum_bn)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        if attention:
            self.att_branch = self._make_layer(block, 512, attention_layers, 1, attention=True)
        else:
            self.att_branch = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.mlp:
            if self.two_branch:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512 * block.expansion),
                    nn.ReLU()
                ) 
                self.instDis = linear(512 * block.expansion, num_classes)
                self.groupDis = linear(512 * block.expansion, num_classes)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512 * block.expansion),
                    nn.ReLU(),
                    linear(512 * block.expansion, num_classes)
                ) 
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            if self.two_branch:
                self.groupDis = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, attention=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, momentum=self.momentum_bn),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, momentum_bn=self.momentum_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, momentum_bn=self.momentum_bn))

        if attention:
            layers.append(nn.Sequential(
                conv1x1(self.inplanes, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                conv1x1(128, 1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.att_branch is not None:
            att_map = self.att_branch(x)
            x = x + att_map * x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.mlp and self.two_branch:
            x = self.fc(x)
            x1 = self.instDis(x)
            x2 = self.groupDis(x)
            return [x1, x2]
        else:
            x1 = self.fc(x)
            if self.two_branch:
                x2 = self.groupDis(x)
                return [x1, x2]
            return x1
        

    def forward_fea(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward_threshold(self, x, threshold=1e6):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.att_branch is not None:
            att_map = self.att_branch(x)
            x = x + att_map * x

        x = self.avgpool(x)
        x = x.clip(max=threshold)
        x = torch.flatten(x, 1)#输出为B*C
        if self.mlp and self.two_branch:
            x = self.fc(x)
            x1 = self.instDis(x)
            x2 = self.groupDis(x)
            return [x1, x2]
        else:
            x1 = self.fc(x)
            if self.two_branch:
                x2 = self.groupDis(x)
                return [x1, x2]
            return x1
        

    def feature_list(self, x):
        out_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out_list.append(out)
        out = self.layer1(out)
        out_list.append(out)
        out = self.layer2(out)
        out_list.append(out)
        out = self.layer3(out)
        out_list.append(out)
        out = self.layer4(out)
        out_list.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #y = self.fc(out)
        return out, out_list

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out