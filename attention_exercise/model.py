import torch
from torch import nn
import torch.nn.init as init
import math


class WSDModel(nn.Module):
    _MAX_RELATIVE_POSITION = 10

    def __init__(self, V, Y, D=300, dropout_prob=0.2, use_padding=False, use_relative_distance=False, use_causal_attention=False):
        super(WSDModel, self).__init__()
        self.use_padding = use_padding

        self.D = D
        self.pad_id = 0
        self.E_v = nn.Embedding(V, D, padding_idx=self.pad_id)
        self.E_y = nn.Embedding(Y, D, padding_idx=self.pad_id)
        init.kaiming_uniform_(self.E_v.weight[1:], a=math.sqrt(5))
        init.kaiming_uniform_(self.E_y.weight[1:], a=math.sqrt(5))

        self.W_A = nn.Parameter(torch.Tensor(D, D))
        self.W_O = nn.Parameter(torch.Tensor(D, D))
        init.kaiming_uniform_(self.W_A, a=math.sqrt(5))
        init.kaiming_uniform_(self.W_O, a=math.sqrt(5))

        self.dropout_layer = nn.Dropout(p=dropout_prob)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm([self.D])

        self.use_causal_attention = use_causal_attention
        self.use_relative_distance = use_relative_distance
        self.relative_distance_matrices = dict()

    def create_distance_matrix(self, N):
        """
        Creates a square matrix with adjusted distance weights.
        """
        # Implementation from https://github.com/TensorUI/relative-position-pytorch/blob/master/relative_position.py
        if N in self.relative_distance_matrices:
            return self.relative_distance_matrices[N]
        range_vec = torch.arange(N)
        distance_mat = abs(range_vec[None, :] - range_vec[:, None])
        distance_mat_clipped = torch.clamp(distance_mat, -self._MAX_RELATIVE_POSITION, self._MAX_RELATIVE_POSITION)
        final_mat = distance_mat_clipped.float()
        if self.use_causal_attention:
            final_mat[torch.triu(torch.ones_like(distance_mat_clipped, dtype=torch.bool), 1)] = math.inf
        self.relative_distance_matrices[N] = final_mat
        return final_mat

    def attention(self, X, Q, mask):
        """
        Computes the contextualized representation of query Q, given context X, using the attention model.

        :param X:
            Context matrix of shape [B, N, D]
        :param Q:
            Query matrix of shape [B, k, D], where k equals 1 (in single word attention) or N (self attention)
        :param mask:
            Boolean mask of size [B, N] indicating padding indices in the context X.

        :return:
            Contextualized query and attention matrix / vector
        """
        Q_c = None
        A = None        
        
        weights = Q @ self.W_A @ X.transpose(-2, -1)

        if self.use_relative_distance:
            weights -= self.create_distance_matrix(X.size()[1]).cuda()

        if self.use_padding:
            mask = mask.unsqueeze(1)
            mask = mask.expand(*weights.size())
            weights[~mask] = -math.inf

        A = self.softmax(weights)
        Q_c =  torch.matmul(torch.matmul(A, X), self.W_O)
        return Q_c, A.squeeze()

    def forward(self, M_s, v_q=None):
        """
        :param M_s:
            [B, N] dimensional matrix containing token integer ids
        :param v_q:
            [B] dimensional vector containing query word indices within the sentences represented by M_s.
            This argument is only passed in single word attention mode.

        :return: logits and attention tensors.
        """

        X = self.dropout_layer(self.E_v(M_s))   # [B, N, D]
        
        Q = None
        if v_q is not None:
            v_q = v_q.unsqueeze(-1)
            indices = v_q.repeat(1, X.size()[-1])
            indices = indices.unsqueeze(1)
            Q = X.gather(1, indices)
        else:
            Q = X
            

        mask = M_s.ne(self.pad_id)
        Q_c, A = self.attention(X, Q, mask)
        H = self.layer_norm(Q_c + Q)
        
        E_y = self.dropout_layer(self.E_y.weight)
        y_logits = (H @ E_y.T).squeeze()
        return y_logits, A
