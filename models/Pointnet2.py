import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
#idea code from https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2/models/pointnet2_ssg_cls.py

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
    
class Model(nn.Module):
    def __init__(self, use_xyz=True, out_embedding = 4096, first_mlp =[0, 64, 64, 128], second_mlp = [128, 128, 128, 256],
                 last_mlp = [256, 256, 512, 1024], max_sample = 64, number_points = [512, 128], ball_r = [0.2,0.4], activation="relu", **kwargs):
        super(Model, self).__init__()
        self.use_xyz = use_xyz
        self.out_embedding = out_embedding
        self.act = get_activation(activation)
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(        #the original SA(512, 0.2, [64, 64, 128])
            PointnetSAModule( 
                npoint=number_points[0],
                radius=ball_r[0],
                nsample=max_sample,
                # mlp=[0, 64, 64, 128],
                mlp = first_mlp,
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(           #the original 2nd SA(128, 0.4, [128, 128, 256])
                npoint=number_points[1],
                radius=ball_r[1],
                nsample=max_sample,
                # mlp=[128, 128, 128, 256],
                mlp = second_mlp,
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(           #the original last SA([256, 512, 1024])
                # mlp=[256, 256, 512, 1024], 
                mlp = last_mlp,
                use_xyz=self.use_xyz    
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias = False),
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
        
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features
        
    def forward(self, pointcloud):
        """
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        pointcloud = pointcloud.permute(0, 2, 1) #this line code change from  (B, N, 3 + input_channels) to (B, 3 + input_channels, N) 
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))


def pointnet2ssg(out_embedding = 4096, **kwargs) -> Model:
    return Model(use_xyz=True, out_embedding = out_embedding, first_mlp =[0, 64, 64, 128], second_mlp = [128, 128, 128, 256],
                 last_mlp = [256, 256, 512, 1024], max_sample = 64, number_points = [512, 128], ball_r = [0.2,0.4],  **kwargs)
    
def pointnet2ssgelite(out_embedding=4096, **kwargs) -> Model:
    return Model(use_xyz=True, out_embedding = out_embedding, first_mlp =[0, 64, 64, 128], second_mlp = [128, 128, 128, 256],
                 last_mlp = [256, 256, 512, 1024], max_sample = 22, number_points = [512, 128], ball_r = [0.2,0.4],  **kwargs)

class pointnet2ssgeliteTriplet(nn.Module):
    """
    """

    def __init__(self,out_embedding=4096,embedding_normalize=True):
        super(pointnet2ssgeliteTriplet, self).__init__()
        self.model = pointnet2ssgelite(out_embedding = out_embedding)
        self.normalize = embedding_normalize


    def forward(self, points):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(points)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

class pointnet2ssgTriplet(nn.Module):
    """
    """

    def __init__(self,out_embedding=4096,embedding_normalize=True):
        super(pointnet2ssgTriplet, self).__init__()
        self.model = pointnet2ssg(out_embedding = out_embedding)
        self.normalize = embedding_normalize


    def forward(self, points):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(points)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    
if __name__ == '__main__':
    from torchinfo import summary
    device = "cuda"
    batch_size = 3 * 1
    data = torch.rand(batch_size, 3, 2048)
    print(data.shape)
    data = data.to(device)
    model = pointnet2ssgeliteTriplet().to(device)
    summary(model, input_size=(batch_size, 3, 2048))
    # for name, param in model.named_parameters():
    #     print(name)
    # print(model)
    out = model(data)
    print(out.shape)