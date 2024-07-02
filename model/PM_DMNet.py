import torch
import torch.nn as nn
from model.PM_Cell import PM_Cell
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F

class PM_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super(PM_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PM_cells = nn.ModuleList()
        self.PM_cells.append(PM_Cell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        for _ in range(1, num_layers):
            self.PM_cells.append(PM_Cell(node_num, dim_out, dim_out, cheb_k, embed_dim, time_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]     #x=[batch,steps,nodes,input_dim]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]   #state=[batch,steps,nodes,input_dim]
            inner_states = []
            for t in range(seq_length):
                state = self.PM_cells[i](current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :], node_embeddings[1]])#state=[batch,steps,nodes,input_dim]
                # state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state,[node_embeddings[0], node_embeddings[1]])
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.PM_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class PM_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim, num_layers=1):
        super(PM_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.PM_cells = nn.ModuleList()
        self.PM_cells.append(PM_Cell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))
        for _ in range(1, num_layers):
            self.PM_cells.append(PM_Cell(node_num, dim_in, dim_out, cheb_k, embed_dim, time_dim))

    def forward(self, xt, init_state, node_embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.PM_cells[i](current_inputs, init_state[i], [node_embeddings[0], node_embeddings[1]])
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class PM_DMNet(nn.Module):
    def __init__(self, args):
        super(PM_DMNet, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.dropout = nn.Dropout(p=0.1)
        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        # self.node_embeddings2 = nn.Parameter(torch.randn(self.num_node, args.time_dim), requires_grad=True)
        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.time_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.time_dim))

        self.encoder = PM_Encoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                  args.embed_dim, args.time_dim, args.num_layers)
        # self.decoder = PM_Decoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
        #                           args.embed_dim, args.time_dim, args.num_layers)
        #predictor
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        # self.end_conv2 = nn.Conv2d(12, 12, kernel_size=(1, 1), bias=True)
        if args.type == 'P':
            self.TA = TransformAttentionModel(self.hidden_dim, args.time_dim,args.embed_dim)
            self.decoder = Parallel_decoder(args)
        elif args.type =='R':
            self.decoder = Recurrent_decoder(args)
    def forward(self, source, traget=None, batches_seen=None):
        #source: B, T_1, N, D
        #target: B, T_2, N, D


        t_i_d_data1 = source[..., 0, -2]
        t_i_d_data2 = traget[..., 0, -2]
        # T_i_D_emb = self.T_i_D_emb[(t_i_d_data[:, -1, :] * 288).type(torch.LongTensor)]
        T_i_D_emb1 = self.T_i_D_emb[(t_i_d_data1 * 288).type(torch.LongTensor)]
        T_i_D_emb2 = self.T_i_D_emb[(t_i_d_data2 * 288).type(torch.LongTensor)]

        d_i_w_data1 = source[..., 0, -1]
        d_i_w_data2 = traget[..., 0, -1]
        # D_i_W_emb = self.D_i_W_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]
        D_i_W_emb1 = self.D_i_W_emb[(d_i_w_data1).type(torch.LongTensor)]
        D_i_W_emb2 = self.D_i_W_emb[(d_i_w_data2).type(torch.LongTensor)]
        node_embedding1 = torch.mul(T_i_D_emb1, D_i_W_emb1)
        node_embedding2 = torch.mul(T_i_D_emb2, D_i_W_emb2)


        en_node_embeddings=[node_embedding1,self.node_embeddings]

        # source = source[..., :self.input_dim].unsqueeze(-1)
        source = source[..., :self.input_dim]

        init_state = self.encoder.init_hidden(source.shape[0])  # [2,64,307,64]
        state, h_n = self.encoder(source, init_state, en_node_embeddings)  # B, T, N, hidden

        output =self.decoder(source, traget,h_n,node_embedding1, node_embedding2,self.node_embeddings,batches_seen)
        # h_n = h_n[0].unsqueeze(1)
        # # output = self.end_conv1(self.dropout(h_n)).reshape(-1,self.horizon,self.num_node,self.output_dim)
        #
        # de_input = self.TA(h_n,node_embedding1[:,-1,:].unsqueeze(1),node_embedding2).flatten(0, 1)
        # # de_input = h_n.expand(-1,self.horizon,-1,-1).flatten(0, 1)
        # # output = self.proj(self.dropout(de_input))
        # node_embedding2 = node_embedding2.flatten(0, 1)
        # # return output
        #
        # go = torch.zeros((source.shape[0]*self.horizon, self.num_node, self.output_dim), device=source.device)
        #
        #
        # state, ht_list = self.decoder(go, [de_input], [node_embedding2, self.node_embeddings])
        # go = self.proj(self.dropout(state))
        # output = go.reshape(source.shape[0],self.horizon,self.num_node,self.output_dim)


        return output


class Parallel_decoder(nn.Module):
    def __init__(self,args=None):
        super(Parallel_decoder,self).__init__()
        self.TA= TransformAttentionModel(args.rnn_units, args.time_dim,args.embed_dim)
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.decoder = PM_Decoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                  args.embed_dim, args.time_dim, args.num_layers)
        # self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.dropout = nn.Dropout(p=0.)
        # self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
        self.weights = nn.Parameter(torch.FloatTensor(self.horizon,self.hidden_dim, self.output_dim))
        self.bias = nn.Parameter(torch.FloatTensor(self.horizon,self.output_dim))
        # self.end_conv = nn.Conv2d(args.horizon, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
    def forward(self, source, traget, h_n,node_embedding1, node_embedding2,node_embeddings,batches_seen):
        h_n = h_n[0]
        h_n = h_n.unsqueeze(1)
        de_input = self.TA(h_n,node_embedding1[:,-1,:].unsqueeze(1),node_embedding2).flatten(0, 1)
        # de_input = h_n.expand(-1,self.horizon,-1,-1).flatten(0, 1)
        # output = self.proj(self.dropout(de_input))
        node_embedding2 = node_embedding2.flatten(0, 1)
        # return output

        go = torch.zeros((source.shape[0]*self.horizon, self.num_node, self.output_dim), device=source.device)


        state, ht_list = self.decoder(go, [de_input], [node_embedding2, node_embeddings])
        

        # go = self.proj(self.dropout(state))
        # output = go.reshape(source.shape[0],self.horizon,self.num_node,self.output_dim)

        state = state.reshape(source.shape[0],self.horizon,self.num_node,self.hidden_dim)
        output = torch.matmul(self.dropout(state), self.weights)+self.bias.unsqueeze(dim=-2)

        # output = self.end_conv(self.dropout(state)).reshape(source.shape[0],self.horizon,self.num_node,self.output_dim)
        #
        return output

class Recurrent_decoder(nn.Module):
    def __init__(self,args=None):
        super(Recurrent_decoder,self).__init__()
        # self.cl_decay_steps = args.lr_decay_step
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.teacher_forcing = args.teacher_forcing
        self.teacher_decay_step = args.teacher_decay_step
        self.decoder = PM_Decoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                  args.embed_dim, args.time_dim, args.num_layers)
        self.proj = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim, bias=True))
    def forward(self, source, traget, h_n, node_embedding1,node_embedding2,node_embeddings, batches_seen):
        ht_list= h_n
        go = torch.zeros((source.shape[0], self.num_node, self.output_dim), device=source.device)
        out = []
        for t in range(self.horizon):
            state, ht_list = self.decoder(go, ht_list, [node_embedding2[:, t, :], node_embeddings])
            go = self.proj(state)
            out.append(go)
            if self.training and self.teacher_forcing:     #这里的课程学习用了给予一定概率用真实值代替预测值来学习的教师-学生学习法（名字忘了，大概跟着有关）
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):  #如果满足条件，则用真实值代替预测值训练
                    # go = traget[:, t, :, :self.input_dim].unsqueeze(-1)
                    go = traget[:, t, :, :self.input_dim]
        output = torch.stack(out, dim=1)

        return output

    def _compute_sampling_threshold(self, batches_seen):
        x = self.teacher_decay_step / (
            self.teacher_decay_step + np.exp(batches_seen / self.teacher_decay_step))
        return x



class TransformAttentionModel(torch.nn.Module):
    def __init__(self, hidden_dim, time_dim,embed_dim):
        super(TransformAttentionModel, self).__init__()
        self.fc_Q = torch.nn.Linear(time_dim+hidden_dim, hidden_dim)
        self.fc_K = torch.nn.Linear(time_dim+hidden_dim, hidden_dim)
        self.fc_V = torch.nn.Linear(time_dim+hidden_dim, hidden_dim)
        self.fc24 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc25 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, 2, hidden_dim, hidden_dim))
        self.weights = nn.Parameter(torch.FloatTensor(2, hidden_dim, hidden_dim))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_dim))
        self.bias = nn.Parameter(torch.FloatTensor(hidden_dim))
        self.d = 8
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)


    def forward(self, X, STE_P, STE_Q):
        STE_Q = STE_Q.unsqueeze(2)
        STE_P = STE_P.unsqueeze(2)
        query = F.relu(self.fc_Q(torch.cat((STE_Q.expand(-1,-1,X.shape[2],-1),X.expand(-1,STE_Q.shape[1],-1,-1)),dim=-1)))
        key = F.relu(self.fc_K(torch.cat((STE_P.expand(-1,-1,X.shape[2],-1),X),dim=-1)))
        value = F.relu(self.fc_V(torch.cat((STE_P.expand(-1,-1,X.shape[2],-1),X),dim=-1)))

        query = torch.cat(torch.split(query, int(query.shape[-1]/self.d), dim=-1), dim=0)
        key = torch.cat(torch.split(key, int(key.shape[-1]/self.d), dim=-1), dim=0)
        value = torch.cat(torch.split(value, int(value.shape[-1]/self.d), dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)                            # [K * batch_size, num_nodes, num_steps, d]
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)         # [K * batch_size, num_nodes, d, num_steps]
        value = torch.transpose(value, 2, 1)

        attention = torch.matmul(query, key)                           # [K * batch_size, num_nodes, num_steps, num_steps]
        # attention /= (self.d ** 0.5)
        attention = torch.softmax(attention,dim=-2)


        output = torch.matmul(attention,value)
        output = torch.transpose(output, 2, 1)
        output = torch.cat(torch.split(output, output.shape[0]//self.d, dim=0), dim=-1)
        output = torch.stack((X.expand(-1,output.shape[1],-1,-1),output),dim=3)
        # output = torch.einsum('btnki,nkio->btno', output, weights) + bias  # b, N, dim_out

        output = torch.einsum('btnki,kio->btno', output, self.weights) + self.bias  # b, N, dim_out

        return output


