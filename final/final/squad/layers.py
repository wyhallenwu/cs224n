"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

import numpy as np


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """

    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        # word_vectors shape(88430, 300)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)  # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)  # (batch_size, seq_len, hidden_size)
        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """

    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.gates = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, input_size, hidden_size, num_layers, drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]  # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)  # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)  # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)  # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob,
                      self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob,
                      self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


# ======================my adding=================================
class CharacterEmbeddingLayer(nn.Module):
    """Character Embedding layer in original paper

    Compute character embeddings following the 4.model details in paper BiDaf
    and layer architecture in paper https://arxiv.org/abs/1408.5882 . It is a little 
    different from the original paper in some details. See more in the code comment.

    Args:
        hidden_size (int): Hidden size used in BiDaf 
        char_vectors (tensor): pre-trained char vectors
        char_idxs (tensor): character indexs of the batch of sentences
    """

    def __init__(self, hidden_size, char_vectors, drop_prob):
        super(CharacterEmbeddingLayer, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.window_size_list = [x for x in range(2, 6)]
        # char_embedding
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        # char_vector shape(1376, 64)
        self.hwy = HighwayEncoder(2, hidden_size=hidden_size)

    def single_conv(self, char_idxs, window_size):
        # embed (batch_size, sent_len, sub_char_len, embedding_size: 64)
        embed = self.char_embed(char_idxs)
        embed_size = embed.size()
        # embed (batch_size * sent_len, 1, sub_char_len * embedding_size)
        embed = embed.view(embed_size[0] * embed_size[1], 1, -1)
        # dropout layer (used in original paper)
        embed = F.dropout(embed, self.drop_prob)
        # in original paper, filter_shape (100 filters)
        # conv1d = nn.Conv1d(1, 100, window_size * embed_size[3])
        # embed = conv1d(embed).view(embed_size[0] * embed_size[1], 100, -1)

        filter = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(100, 1, window_size * embed_size[3]))).cuda()
        # shape (batch_size * sent_len, 100, sub_char_len - window_size + 1)
        embed = F.conv1d(embed, filter, stride=embed_size[3]).view(
            embed_size[0] * embed_size[1], 100, -1)
        # original paper using relu
        embed = F.tanh(embed)
        # max-over-time pooling  (batch_size, sent_len, 100)
        embed = F.max_pool1d(embed, embed_size[2] - window_size + 1).view(
            embed_size[0], embed_size[1], 100)
        return embed

    def multi_conv(self, char_idxs):
        outs = []
        # multi-conv using different window_size
        for _, window_size in enumerate(self.window_size_list):
            outs.append(self.single_conv(char_idxs, window_size))
        # shape (batch_size, sent_len, 100)
        outs = torch.cat(outs, dim=2)
        # (batch_size, sent_len, window_list_len * 100)
        return outs

    def forward(self, char_idxs):
        outs = self.multi_conv(
            char_idxs)  # (batch_size, sent_len, window_list_len * 100)
        # regularization using dropout
        outs = F.dropout(outs, self.drop_prob)
        outs = F.linear(
            outs,
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(self.hidden_size, outs.size(
                        2)))).cuda())  # (batch_size, sent_len, hidden_size)

        # testing adding highwayEncoder
        # different from original embedding, adding highway project
        outs = self.hwy(outs)
        return outs


# currently a idea to test (tesing result is bad, this layer has been removed)
# maybe apply after contextual embedding layer


class QueryMask(nn.Module):

    def __init__(self, mask_prob):
        super(QueryMask, self).__init__()
        self.mask_prob = mask_prob

    def mask(self, input):
        # input (batch_size, sent_len, hidden_size)
        prob = np.random.uniform(0, 1)

        if prob < self.mask_prob:
            input_size = input.size()
            qmask_index = np.random.choice(input_size[1],
                                           int(self.mask_prob * input_size[1]),
                                           False)
            # input (batch_size, hidden_size, after_masked)
            masked_result = input[:, qmask_index, :].permute([0, 2, 1])

            output = F.linear(
                masked_result,
                nn.Parameter(
                    nn.init.xavier_uniform_(
                        torch.empty(
                            (input_size[1], masked_result.size(2))))).cuda())
            # output (batch_size, after_masked, hidden_size)
            # TODO(test leakyrelu)
            output = F.tanh(output.permute([0, 2, 1]))
            # output (batch_size, sent_len, hidden_size)
            return output
        return input

    def forward(self, input):
        return self.mask(input)
