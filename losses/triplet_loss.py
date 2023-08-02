"""This code was imported from tbmoon's 'facenet' repository:
    https://github.com/tbmoon/facenet/blob/master/utils.py
"""

import torch
from torch.autograd import Function
from torch.nn.modules.distance import PairwiseDistance


class TripletLoss(Function):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(p=2)

    def forward(self, anchor, positive, negative):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)

        return loss
    
class TripletLossPointnet(Function):

    def __init__(self, margin, batch_size, alpha = 0.0001):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance(p=2)
        self.bs = batch_size
        self.id3x3 = torch.eye(3, requires_grad=True).repeat(self.bs,1,1)
        self.id64x64 = torch.eye(64, requires_grad=True).repeat(self.bs,1,1)
        self.alpha = alpha

    def forward(self, anchor, positive, negative ,m3x3, m64x64):
        pos_dist = self.pdist.forward(anchor, positive)
        neg_dist = self.pdist.forward(anchor, negative)

        hinge_dist = torch.clamp(self.margin + pos_dist - neg_dist, min=0.0)
        loss = torch.mean(hinge_dist)
        #added loss for transforamtion networks
        id3x3=self.id3x3.cuda()
        id64x64=self.id64x64.cuda()
        diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
        diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
        loss2 = self.alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(self.bs)
        return loss + loss2
