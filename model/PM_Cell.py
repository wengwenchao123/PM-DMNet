import torch
import torch.nn as nn
from model.DMN import DMN

class PM_Cell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim):
        super(PM_Cell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = DMN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim, time_dim)
        self.update = DMN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim, time_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
