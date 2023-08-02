import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#Thanks to https://www.kaggle.com/code/balraj98/pointnet-for-3d-object-classification-pytorch
#TRiplet model Pointnet follows https://doi.org/10.1007/s11042-020-10160-9

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)
    
class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,2048,1)
        self.fc1 = nn.Linear(2048,256)
        # self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(2048)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn5(self.fc1(flat)))
        # xb = F.relu(self.bn5(self.fc2(xb)))

        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
        if xb.is_cuda:
            init=init.cuda()
        matrix = self.fc3(xb).view(-1,self.k,self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv1_1 = nn.Conv1d(64,64,1)
        self.conv2_1 = nn.Conv1d(64,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,2048,1)


        self.bn1 = nn.BatchNorm1d(64)
        self.bn1_1 = nn.BatchNorm1d(64)
        self.bn1_2 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(2048)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        #Input Transformation Network
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2) 
        #Forward network-1
        xb = F.relu(self.bn1(self.conv1(xb)))
        xb = F.relu(self.bn1_1(self.conv1_1(xb))) 
        #Feature transforamtion Network
        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)
        #Forward Network-2
        xb = F.relu(self.bn1_2(self.conv2_1(xb)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class Model(nn.Module):
    def __init__(self, activation = "relu", out_embedding=4096):
        super().__init__()
        self.transform = Transform()
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 4096)
        self.act = get_activation(activation)
        self.out_embedding = out_embedding
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.dropout = nn.Dropout(p=0.5)
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512, bias = False),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256, bias = False),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.out_embedding, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        output = self.classifier(xb)
        # xb = F.relu(self.bn1(self.fc1(xb)))
        # xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        # output = self.fc3(xb)
        return output, matrix3x3, matrix64x64
    
class Model_2(nn.Module):
    def __init__(self, activation = "relu", out_embedding=4096):
        super().__init__()
        self.transform = Transform()
        self.act = get_activation(activation)
        self.out_embedding = out_embedding
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512, bias = False),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(512, 256, bias = False),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(0.5),
            nn.Linear(256, self.out_embedding, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        output = self.classifier(xb)
        return output
    
def Pointnet(out_embedding = 4096, **kwargs) -> Model:
    return Model(activation = "relu", out_embedding=out_embedding,  **kwargs)

def Pointnet_v2(out_embedding = 4096, **kwargs) -> Model:
    return Model_2(activation = "relu", out_embedding=out_embedding,  **kwargs)

class pointnetTriplet(nn.Module):
    """Constructs a pointnetTriplet model for 3Dface training using triplet loss.
    """

    def __init__(self,out_embedding=4096,embedding_normalize=True):
        super(pointnetTriplet, self).__init__()
        self.model = Pointnet(out_embedding = out_embedding)
        self.normalize = embedding_normalize

    def forward(self, points):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding,matrix3x3,matrix64x64  = self.model(points)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, matrix3x3 , matrix64x64

class pointnetTriplet_v2(nn.Module):
    """Constructs a pointnetTriplet_v2 model for 3DFace training using triplet loss.
    """

    def __init__(self,out_embedding=4096,embedding_normalize=True):
        super(pointnetTriplet_v2, self).__init__()
        self.model = Pointnet_v2(out_embedding = out_embedding)
        self.normalize = embedding_normalize

    def forward(self, points):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding  = self.model(points)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
    
def pointnetloss_Tnet(batch_size, m3x3, m64x64, alpha = 0.0001):
    # bs=outputs.size(0)
    bs = batch_size
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    # if outputs.is_cuda:
    id3x3=id3x3.cuda()
    id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return  alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)

if __name__ == '__main__':
    from torchinfo import summary
    device = "cuda"
    batch_size = 3 * 1
    data = torch.rand(batch_size, 3, 2048)
    print(data.shape)
    # data =data.permute(0, 2, 1)
    # print(data.shape)
    # print("===> testing pointMLP ...")
    # torch.cuda.empty_cache()
    data = data.to(device)
    model = pointnetTriplet().to(device)
    # model = pointMLPTriplet().to(device)
    summary(model, input_size=(batch_size, 3, 2048))
    # for name, param in model.named_parameters():
    #     print(name)
    # print(model)
    out,matrix3x3,matrix64x64 = model(data)
    print(out.shape)
   