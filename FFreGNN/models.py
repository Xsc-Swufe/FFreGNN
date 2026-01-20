''' Define the HGTAN model '''
import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
from training.tools import *
import torch.nn.functional as F
from FFreGNN.layers import *
import torch.fft


class  FFreGNN(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
                self,
                num_stock,  n_hid, n_class,
                feature,
                dropout,
                scale_num, path_num, window_size, mem_dim):

        super().__init__()
        self.dropout = dropout

        self.scale_num = scale_num
        self.path_num = path_num
        self.n_hid=n_hid
        self.window_size = window_size




        self.tgt_word_prj = nn.Linear(2*n_hid, n_class, bias=False)

        #self.linear = nn.Linear(scale_num*rnn_unit, n_hid, bias=False)
        #self.tgt_word_prj = nn.Linear(2*rnn_unit, n_class, bias=False)
        #self.tgt_word_prj = nn.Linear(n_hid, n_class, bias=False)



        self.knowledge = KnowledgeGuidedFrequencyAlignment(T=self.window_size, C=feature, P=60, feature_dim=n_hid)


        self.routing_module = StateAwareGraphRouting(input_dim= self.window_size // 2 + 1, num_modes=5, d_k= n_hid, tau=0.5)
        self.dre_module = DynamicRelationExtraction(C=5, tau_factor=1)

        self.ln = nn.LayerNorm(n_hid)
        #self.ln2 = nn.LayerNorm(rnn_unit * scale_num)
        self.ln_in = nn.LayerNorm(feature)
        self.ln3 = nn.LayerNorm(2*n_hid)
        self.ln5 = nn.LayerNorm(n_hid)

    def forward(self,src_seq):
        stock_num = src_seq.size(1)
        seq_len = src_seq.size(0)


        if torch.isnan(src_seq).any():
            print("src_seq1 中存在 NaN 值！")


        orig_amplitude, orig_phase, refined_amplitude, refined_phase, temporal_features = self.knowledge(src_seq)
        temporal_features = F.dropout(temporal_features, p=self.dropout, training=self.training)




        R_nm = self.dre_module(refined_phase, refined_amplitude)

        h_n,s_ng = self.routing_module(temporal_features, refined_phase, orig_phase, refined_amplitude, orig_amplitude, R_nm)
        h_n = self.ln(h_n)

        #combined_output = temporal_features
        combined_output = torch.cat((h_n,temporal_features), dim=1)

        #combined_output = F.dropout(combined_output, self.dropout, training=self.training)
        #combined_output = s_n
        seq_logit = F.elu(self.tgt_word_prj(combined_output))
        #
        output = F.log_softmax(seq_logit, dim=1)
        # y_tem = F.log_softmax(output, dim=1)
        # output: torch.Size([385, 2])



        return output,s_ng


