import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
from PIL import Image
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import Module
from tqdm import tqdm
import time
from torchvision import datasets
import ast
import io
import logging
import random

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
class dataset_breast(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.image_paths = [os.path.join(data_path, img) for img in os.listdir(data_path)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image,1001
class imagenetdataset(Dataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 **kwargs):
        super(imagenetdataset, self).__init__(**kwargs)
        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        #if self.data_dir != '' and image_name.startswith('/'):
        #    raise RuntimeError('image_name starts with "/"')
        path = image_name#path = os.path.join(self.data_dir, image_name)
        sample = dict()
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        try:
            # some preprocessor methods require setup
            self.preprocessor.setup(**kwargs)
        except:
            pass

        try:
            if not self.dummy_read:
                with open(path, 'rb') as f:
                    content = f.read()
                filebytes = content
                buff = io.BytesIO(filebytes)
            if self.dummy_size is not None:
                sample['data'] = torch.rand(self.dummy_size)
            else:
                image = Image.open(buff).convert('RGB')
                sample['data'] = self.transform_image(image)
                sample['data_aux'] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample['data'], sample['label']

    def __getitem__(self, index):
        # in some pytorch versions, input index will be torch.Tensor
        index = int(index)

        # if sampler produce pseudo_index,
        # randomly sample an index, and mark it as pseudo
        if index == self.pseudo_index:
            index = random.randrange(len(self))
            pseudo = 1
        else:
            pseudo = 0

        
        sample = self.getitem(index)

        return sample






def dataset_imagenet(path):
    input_TF = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
                                   transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    testsetout = datasets.ImageFolder(path, transform=input_TF)
    return testsetout
def dataset_rare_gastric(train_path,test_path):
    labels=[]
    paths = []
    with open(train_path,"r") as f:
        lines_train = f.readlines()#稀有病例并没有参与训练
    with open(test_path,"r") as f:
        lines_test = f.readlines()
    for index, line in enumerate(lines_train):
        path = line.split(' ')[0].split()[0]
        label=line.split(' ')[1].split()[0]
        label =int(label)
        if label in[3,4]:
            paths.append(path)
            labels.append(label)
    for index, line in enumerate(lines_test):
        path = line.split(' ')[0].split()[0]
        label=line.split(' ')[1].split()[0]
        label =int(label)
        if label in[3,4]:
            paths.append(path)
            labels.append(label)
    paths = np.array(paths)
    labels = np.array(labels)
    image_paths_labels=np.concatenate((paths.reshape(-1,1),labels.reshape(-1,1)),axis=1)
    ood_data = paths_labelsDataset(image_paths_labels, transform)
    return ood_data

class dataset_breast_100(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        #self.image_paths = [os.path.join(data_path, img) for img in os.listdir(data_path)]
        self.image_paths=self.path()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    def path(self):  
        paths=[]
        with open(self.data_path, "r") as file:
            for line in file:
                # 使用空格分割每行，得到路径和标签
                path = line.strip()
                paths.append(path)
        return paths

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image,1001
    



class dataset_lung(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        #self.image_paths = [os.path.join(data_path, img) for img in os.listdir(data_path)]
        self.image_paths=self.path()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    def path(self):  
        paths=[]
        with open(self.data_path, "r") as file:
            for line in file:
                # 使用空格分割每行，得到路径和标签
                path, _ = line.strip().split(' ')
                paths.append(path)
        return paths

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image,1001
class oodTCGA_Gastric(Dataset):
    def __init__(self, paths_labels, transform=None):
        self.paths_labels = paths_labels
        self.transform = transform

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, index):
        path, label = self.paths_labels[index]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image,1001
class paths_labelsDataset(Dataset):
    def __init__(self, paths_labels, transform=None):
        self.paths_labels = paths_labels
        self.transform = transform

    def __len__(self):
        return len(self.paths_labels)

    def __getitem__(self, index):
        path, label = self.paths_labels[index]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(label)
class path_Dataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))#获取 root_dir 目录下的所有文件和文件夹的名称，并存储在 self.classes 列表中。然后，通过 sorted 函数对类别进行排序。
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self.load_images()

    def load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                images.append((image_path, self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:

            image = self.transform(image)
            #print(f"After transform: {type(image)}, {image.size()}")
        #print(type(image))
        return image, label
    
def softmax_file(model,test_loader,temperature,magnitude,net_type,outputfile,id_flag=True):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    if id_flag == True:
        temp_file_name_val = '%s/Probability_val_In.txt'%(outputfile)
        temp_file_name_test = '%s/Probability_test_In.txt'%(outputfile)
    else:
        temp_file_name_val = '%s/Probability_val_Out.txt'%(outputfile)
        temp_file_name_test = '%s/Probability_test_Out.txt'%(outputfile)
        
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')
    item=0
    t0=time.time()
    for data, _ in test_loader:
        item+=1
        if item%1000==0:
            print('第%d组数据，1000组数据总共花费了%.2f秒'%(item,time.time()-t0))
            t0=time.time()
        total += data.size(0)
        data = data.cuda()
        data=data.requires_grad_(True)
        #data = Variable(data, requires_grad = True)
        batch_output = model(data)
            
        # temperature scaling
        outputs = batch_output / temperature
        labels = outputs.data.max(1)[1]
        #labels = labels.requires_grad_(True)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2#变为正负1
        if net_type == 'densenet':
            #gradient.index_select将第0列的元素除以(63.0/255.0)，
            #index_copy_将上一步得到的结果复制回原始梯度张量的第0列，表示直接修改原始张量而不创建新的张量
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        tempInputs = torch.add(data.data, gradient, alpha=-magnitude)
        outputs = model(tempInputs)
        outputs = outputs / temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out, dim=1)
        
        for i in range(data.size(0)):
            if total <= 1000:
                g.write("{}\n".format(soft_out[i]))#使用前1000个元素来作为验证集
            else:
                f.write("{}\n".format(soft_out[i]))
                
    f.close()
    g.close()

def oodtest_loader(out_dist,path,batch_size,shuffle,num_workers):#评价方式
    if out_dist=='TCGA_breast':
        ood_data=dataset_breast(path,transform)
    elif out_dist=='TCGA_lung':
        ood_data=dataset_lung(path,transform)
    else:
        ood_data=path_Dataset(path,transform)
    ood_loader = DataLoader(ood_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return ood_loader

def get_curve(dir_name, stypes = ['Baseline', 'Gaussian_LDA']):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    for stype in stypes:
        known = np.loadtxt('{}/Probability_{}_In.txt'.format(dir_name, stype))#, delimiter='\\n')#output_path/Probability_val_In.txt
        novel = np.loadtxt('{}/Probability_{}_Out.txt'.format(dir_name, stype))#, delimiter='\\n')
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95

def metric(dir_name, stypes = ['Bas', 'Gau'], verbose=False):
    tp, fp, tnr_at_tpr95 = get_curve(dir_name, stypes)
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    for stype in stypes:
        if verbose:
            print('{stype:5s} '.format(stype=stype), end='')
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = tnr_at_tpr95[stype]
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = -np.trapz(1.-fpr, tpr)
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = .5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max()
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = np.trapz(pout[pout_ind], 1.-fpr[pout_ind])
        if verbose:
            print(' {val:6.3f}'.format(val=100.*results[stype][mtype]), end='')
            print('')
    
    return results

def ood_baseline(net, num_classes, feature_list, train_loader):
    net1=1


def jilu(net):#获取特征的方式
    temp_x = torch.rand(1,3,224,224).cuda()
    intermediate_outputs = {}
    #with torch.no_grad():
    #    intermediate_outputs['input'] = temp_x.cpu().detach().numpy()

    # Iterate through the layers
    for name, module in net.named_children():
        if name=='head':
            continue
        temp_x = module(temp_x)
        intermediate_outputs[name] = temp_x.cpu().detach().numpy()
    # Print the intermediate outputs




















