# %%
from torch.nn import Linear
import torch
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool, MLP, PPFConv
# %%
class RotaInvNet(torch.nn.Module):
    def __init__(self, trial, in_dim, out_dim):
        super().__init__()
        self.num_classes = out_dim
        torch.manual_seed(0)
        h2 = trial.suggest_int("h2", 16, 32, 8)
        h3 = trial.suggest_int("h3", 32, 64, 8)
        h4 = trial.suggest_int("h4", 64, 128, 8)
        self.mlp1 = MLP([in_dim, 16, h2])
        self.conv1 = PPFConv(local_nn=self.mlp1)
        self.mlp2 = MLP([h2 + in_dim, h3, h4])
        self.conv2 = PPFConv(local_nn=self.mlp2)
        self.classifier = Linear(h4, self.num_classes)
        
    def forward(self, pos, normal, batch):
        edge_index = knn_graph(pos, k=16, batch=batch, loop=True)
        
        x = self.conv1(x=None, pos=pos, normal=normal, edge_index=edge_index)
        x = torch.relu(x)
        x = self.conv2(x=x, pos=pos, normal=normal, edge_index=edge_index)
        x = torch.relu(x)

        x = global_max_pool(x, batch)
        return self.classifier(x)