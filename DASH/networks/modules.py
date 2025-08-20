import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool,MessagePassing
from torch_scatter import scatter_mean



class MLP(nn.Module):
    def __init__(self,dims):
        super(MLP, self).__init__()
        self.f = nn.Sequential()
        for i in range(len(dims)-1):
            self.f.append(nn.Linear(dims[i],dims[i+1]))
            self.f.append(nn.LeakyReLU(0.01))
            
    def forward(self,x):
        return self.f(x)
    

class GNN(nn.Module):
    def __init__(self,dims,edge_index,gnn_type="GCN",heads=4):
        super(GNN, self).__init__()
        self.f = nn.ModuleList()
        for i in range(len(dims)-2):
            if gnn_type == "GCN":
                self.f.append(GCNConv(dims[i],dims[i+1]))
            elif gnn_type == "GAT":
                if i == 0:
                    self.f.append(GATConv(dims[i],dims[i+1],heads=heads))
                elif i == len(dims)-3:
                    self.f.append(GATConv(dims[i]*heads,dims[i+1],heads=1))
                else:
                    self.f.append(GATConv(dims[i]*heads,dims[i+1],heads=heads))
            self.f.append(nn.LeakyReLU(0.01))
        self.project = MLP(dims[-2:])
        self.edge_index = edge_index

    def forward(self,x):
        if len(x.shape) == 1:
            x = x.reshape(1,x.shape[0])
        batch_size, n_feat = x.shape
        x = x.reshape(-1,3)
        n_atom = int(n_feat / 3)
        edge_index = torch.concat([self.edge_index + i * n_atom for i in range(batch_size)],dim=1).to(x.device)
        batch = torch.concat([torch.zeros(n_atom,dtype=torch.long) + i for i in range(batch_size)]).to(x.device)
        
        for f in self.f:
            if isinstance(f,nn.LeakyReLU):
                x = f(x)
            else:
                x = f(x, edge_index)
        x = global_mean_pool(x,batch)
        return self.project(x)
    
class EGNN(nn.Module):
    def __init__(self,dims,edge_index,n_layer):
        super(EGNN, self).__init__()
        self.f0 = MLP(dims[0:2])
        self.f = nn.ModuleList([EGNNLayer(dims[1]) for i in range(n_layer)])
        self.project = MLP(dims[-2:])
        self.edge_index = edge_index
    
    def forward(self,x):
        if len(x.shape) == 1:
            x = x.reshape(1,x.shape[0])
        batch_size, n_feat = x.shape
        n_atom = int(n_feat / 3)
        cset = x.reshape(batch_size,n_atom,3)
        x = x.reshape(batch_size,n_atom,3)
        edge_index = self.edge_index.to(x.device)
        x = self.f0(x)
        for f in self.f:
            x,cset = f(x,cset,edge_index)
        x = x.mean(1)
        return self.project(x)



class EGNNLayer(nn.Module):
    def __init__(self, dim):
        super(EGNNLayer, self).__init__()
        self.edge_mlp = MLP([dim*2+1,dim,dim])
        self.coord_mlp = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Tanh())  # 限制更新幅度
        self.node_mlp = MLP([dim*2,dim,dim])

    def forward(self, h, pos, edge_index):
        row = edge_index[0]
        col = edge_index[1]
        # 计算相对位置和距离
        pos_i = pos[:,row,:]
        pos_j = pos[:,col,:]
        rel_pos = pos_i - pos_j
        dist_sq = torch.sum(rel_pos**2, dim=-1, keepdim=True)
        # 准备消息输入
        h_i = h[:,row,:]
        h_j = h[:,col,:]
        edge_input = torch.cat([h_i, h_j, dist_sq], dim=-1)
        # 计算消息
        msg = self.edge_mlp(edge_input)
        # 聚合消息 (按目标节点聚合)
        aggr_msg = scatter_mean(msg, col, dim=-2)
        # 坐标更新
        coord_update = self.coord_mlp(aggr_msg)
        pos_out = pos + coord_update * 0.1
        # 节点特征更新
        node_input = torch.cat([h, aggr_msg], dim=-1)
        h_out = h + self.node_mlp(node_input)
        return h_out, pos_out


    