"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=int(hidden_size / 2),
                                    drop_prob=drop_prob)

        # ===============my adding=====================
        # filter shape used in BiDaf with num_filters=100, kernel_size=5
        self.char_emb = layers.CharacterEmbeddingLayer(
            hidden_size=int(hidden_size / 2),
            char_vectors=char_vectors,
            drop_prob=drop_prob)

        # mask layer(result is bad, so this layer has been moved)
        # self.mask_layer = layers.QueryMask(mask_prob=0.85)
        # =============================================

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        # ==========modify============
        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)
        # ========================my adding=========================
        cc_emb = self.char_emb(
            cc_idxs)  # (batch_size, cc_len, hidden_size / 2)
        qc_emb = self.char_emb(
            qc_idxs)  # (batch_size, qc_len, hidden_size / 2)

        # concatenate word_embedding with char_embedding
        c_cat_emb = torch.cat((c_emb, cc_emb),
                              dim=2)  # (batch_size, c_len, hidden_size)
        q_cat_emb = torch.cat((q_emb, qc_emb),
                              dim=2)  # (batch_size, q_len, hidden_size)

        # ==========================================================

        c_enc = self.enc(c_cat_emb,
                         c_len)  # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_cat_emb,
                         q_len)  # (batch_size, q_len, 2 * hidden_size)
        # testing mask(result is bad, so move this part)====================
        # q_after_mask = self.mask_layer(q_enc)

        # ================================

        att = self.att(c_enc, q_enc, c_mask,
                       q_mask)  # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
