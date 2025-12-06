import torch
import torch.nn as nn
import torch.nn.functional as F
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class ABMIL_dropout(nn.Module):
    def __init__(self,n_classes,embed_dim=1024,act='relu',bias=False,dropout=False):
        super(ABMIL_dropout, self).__init__()
        self.L = 512
        self.D = 128 #128
        self.K = 1

        self.feature = [nn.Linear(embed_dim, 512)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)
    def forward(self, x):#N*1024
        x = self.feature(x)#N*512

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)#N*128
        A = self.attention_c(A)#N*1

        A = torch.transpose(A, -1, -2)  # 1xN
        A = F.softmax(A, dim=-1)  #1*N
        x = torch.matmul(A,x)
        A_fea = x # 1*512
        logits = self.classifier(x)
        logits = self.dropout(logits)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat,'A_raw':A}
        return logits, Y_prob, Y_hat, A_fea, results_dict
    
    def forward_threshold(self, x, threshold=1e6):
        x = self.feature(x)#N*512

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)#N*128
        A = self.attention_c(A)#N*1

        A = torch.transpose(A, -1, -2)  # 1xN
        A = F.softmax(A, dim=-1)  #1*N
        x = torch.matmul(A,x)
        x = x.clip(max=threshold)
        logits = self.classifier(x)
        return logits
        



    def probability_patch(self,h):
        x = self.feature(x)
        logits = self.classifier(x)
        return logits

class ABMIL_ensemble(nn.Module):
    def __init__(self,n_classes,embed_dim=1024,act='relu',bias=False,dropout=False):
        super(ABMIL_ensemble, self).__init__()
        self.L = 512
        self.D = 128 #128
        self.K = 1

        self.feature = [nn.Linear(embed_dim, 512)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )
        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)
    def forward(self, x):#N*1024
        x = self.feature(x)#N*512

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)#N*128
        A = self.attention_c(A)#N*1

        A = torch.transpose(A, -1, -2)  # 1xN
        A = F.softmax(A, dim=-1)  #1*N
        x = torch.matmul(A,x)
        A_fea = x # 1*512
        logits = self.classifier(x)
        logits2 = self.classifier2(x)
        logits3 = self.classifier3(x)
        logits4 = self.classifier4(x)
        logits5 = self.classifier5(x)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat,'A_raw':A}
        return logits,logits2,logits3,logits4,logits5, Y_prob, Y_hat, A_fea, results_dict
    
    def forward_threshold(self, x, threshold=1e6):
        x = self.feature(x)#N*512

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)#N*128
        A = self.attention_c(A)#N*1

        A = torch.transpose(A, -1, -2)  # 1xN
        A = F.softmax(A, dim=-1)  #1*N
        x = torch.matmul(A,x)
        x = x.clip(max=threshold)
        logits = self.classifier(x)
        return logits
        



    def probability_patch(self,h):
        x = self.feature(x)
        logits = self.classifier(x)
        return logits

class ABMIL(nn.Module):
    def __init__(self,n_classes,embed_dim=1024,act='relu',bias=False,dropout=False):
        super(ABMIL, self).__init__()
        self.L = 512
        self.D = 128 #128
        self.K = 1

        self.feature = [nn.Linear(embed_dim, 512)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, n_classes),
        )

        self.attention_a = [
            nn.Linear(self.L, self.D,bias=bias),
        ]
        if act == 'gelu': 
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D,bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K,bias=bias)

        self.apply(initialize_weights)
    def forward(self, x):#N*1024
        x = self.feature(x)#N*512

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)#N*128
        A = self.attention_c(A)#N*1

        A = torch.transpose(A, -1, -2)  # 1xN
        A = F.softmax(A, dim=-1)  #1*N
        x = torch.matmul(A,x)
        A_fea = x # 1*512
        logits = self.classifier(x)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat,'A_raw':A}
        return logits, Y_prob, Y_hat, A_fea, results_dict
    
    def forward_threshold(self, x, threshold=1e6):
        x = self.feature(x)#N*512

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)#N*128
        A = self.attention_c(A)#N*1

        A = torch.transpose(A, -1, -2)  # 1xN
        A = F.softmax(A, dim=-1)  #1*N
        x = torch.matmul(A,x)
        x = x.clip(max=threshold)
        logits = self.classifier(x)
        return logits
        



    def probability_patch(self,h):
        x = self.feature(x)
        logits = self.classifier(x)
        return logits


