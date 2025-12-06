'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-07-11 20:37:55
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-07-26 15:45:52
FilePath: /paper1/models/conch_classification.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import torch.nn as nn
from functools import partial

# 假设 create_model_from_pretrained 已经定义并可用
# net, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
import os
def has_CONCH():
    HAS_CONCH = False
    CONCH_CKPT_PATH = '/conch/pytorch_model.bin'
    #终端 export CONCH_CKPT_PATH=/data_nas2/zd/paper1/models/conch/pytorch_model.bin
    # check if CONCH_CKPT_PATH is set and conch is installed, catch exception if not
    try:
        #from conch.open_clip_custom import create_model_from_pretrained
        # check if CONCH_CKPT_PATH is set
        if 'CONCH_CKPT_PATH' not in os.environ:
            raise ValueError('CONCH_CKPT_PATH not set')
        HAS_CONCH = True
        CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
    except Exception as e:
        print(e)
        print('CONCH not installed or CONCH_CKPT_PATH not set')
    return HAS_CONCH, CONCH_CKPT_PATH
def model_load(pre_network,pretrained=False, progress=True, num_classes=1000):
    
    HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
    assert HAS_CONCH, 'CONCH is not available'
    from conch.open_clip_custom import create_model_from_pretrained
    net, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
    net.forward = partial(net.encode_image, proj_contrast=False, normalize=False)
    net = ConchNet(net,num_classes)
    state_dict=torch.load(pre_network, map_location='cpu')
        #td = r'/home/zhangdi/mycode/paper1/models/Ctranspa
    net.load_state_dict(state_dict)
    
    return net
        
        #td=pre_network








class ConchNet(nn.Module):
    def __init__(self, original_net, num_classes):
        super(ConchNet, self).__init__()
        self.original_net = original_net
        self.fc = nn.Linear(512, num_classes)
        
        # Override the forward method to call encode_image with additional layers
        self.original_net.encode_image_with_fc = self.encode_image_with_fc

    def encode_image_with_fc(self, x):
        features = self.original_net.encode_image(x, proj_contrast=False, normalize=False)
        logits = self.fc(features)
        return logits
    def forward_threshold(self, x, threshold=1e6):
        features = self.original_net.encode_image(x, proj_contrast=False, normalize=False)
        x = x.clip(max=threshold)
        logits = self.fc(features)
        return logits
    



    def forward_fea(self,x):
        features = self.original_net.encode_image(x, proj_contrast=False, normalize=False)
        return 12,features
        
    def forward(self, x):
        return self.encode_image_with_fc(x)
    def feature_list(self,x):
        features = self.original_net.encode_image(x, proj_contrast=False, normalize=False)
        return x,[features]

'''
num_classes = 10  # 假设类别数为 10
custom_net = CustomNet(net, num_classes)

# 现在调用 custom_net(temp_x_noise) 将输出直接为 logit
logits = custom_net(temp_x_noise)'''