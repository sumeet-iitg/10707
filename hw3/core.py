""" This file contains components needed by both with-attention and without-attention variants. """
import torch
import numpy as np
from typing import List, Union
from torch.nn.functional import softmax, log_softmax

SOS_token = 0
EOS_token = 1


class GRU(object):
    """ Gated recurrent unit with input dimension `input_dim` and hidden dimension `hidden_dim`
     """

    def __init__(self, W: torch.tensor, U: torch.tensor, Wr: torch.tensor,
                 Ur: torch.tensor, Wz: torch.tensor, Uz: torch.tensor):
        """
        :param W: tensor of shape [hidden_dim, input_dim] - input weights for the candidate hidden state
        :param U: tensor of shape [hidden_dim, hidden_dim] - hidden weights for the candidate hidden state
        :param Wr: tensor of shape [hidden_dim, input_dim] - input weights for the reset gate
        :param Ur: tensor of shape [hidden_dim, hidden_dim] - hidden weights for the reset gate
        :param Wz: tensor of shape [hidden_dim, input_dim] - input weights for the update gate
        :param Uz: tensor of shape [hidden_dim, hidden_dim] - hidden weights for the update gate
        """
        self.W = W
        self.Wr = Wr
        self.Wz = Wz
        self.U = U
        self.Ur = Ur
        self.Uz = Uz

    def forward(self, input: torch.tensor, last_hidden: torch.tensor) -> torch.tensor:
        """
        :param input: tensor of shape [input_dim], denoted x_j in the assignment
        :param last_hidden: tensor of shape [hidden_dim], denoted h_{j-1} in the assignment
        :return: hidden, a tensor of shape [hidden_dim], denoted h_j in the assignment
        """

        reset_gate = torch.sigmoid(torch.mv(self.Wr, input) + torch.mv(self.Ur,last_hidden))
        forget_gate = torch.sigmoid(torch.mv(self.Wz, input) + torch.mv(self.Uz,last_hidden))
        candidate_hid = torch.tanh(torch.mv(self.W, input) + torch.mv(self.U, reset_gate*last_hidden))
        return (1 - forget_gate)*candidate_hid + forget_gate*last_hidden


class StackedGRU(object):
    """ A multilevel stack of L GRUs.
    """

    def __init__(self, grus: List[GRU]):
        """
        :param grus: the GRUs in the stack, from bottom to top.  The GRUs should share the same
                     hidden unit dimension, `hidden_dim`.  The bottom-most GRU should have an input dimension
                      of `input_dim`; all other GRUs should have an input dimension of `hidden_dim`.
        """
        self.grus = grus

    def forward(self, input: torch.tensor, last_hidden: torch.tensor):
        """
        :param input: a tensor of shape [input_dim], denoted x_j in the assignment
        :param last_hidden: a tensor of shape [L, hidden_dim], denoted H_{j-1} in the assignment
        :return: hidden, a tensor of shape [L, hidden_dim], denoted H_j in the assignment
        """

        gru_input = input
        gru_stack_hiddens = []
        for i in range(len(self.grus)):
            gru_input = self.grus[i].forward(gru_input, last_hidden[i])
            gru_stack_hiddens.append(gru_input)

        return torch.stack(gru_stack_hiddens)


class OutputLayer(object):
    """ A fully-connected layer that returns softmax(W^{out} h + b^{out}), where h is the input.
    """

    def __init__(self, weight: torch.tensor, bias: torch.tensor):
        """
        :param weight: [WITHOUT ATTENTION] tensor of shape [target_vocab_size, hidden_dim],
                       [WITH ATTENTION]    tensor of shape [target_vocab_size, 2 * hidden_dim],
                         denoted W^{out} in the assignment
        :param bias: tensor of shape [target_vocab_size], denoted b^{out} in the assigment
        """
        self.weight = weight
        self.bias = bias

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        :param input: [WITHOUT ATTENTION] a tensor of shape [hidden_size]
                      [WITH ATTENTION]    a tensor of shape [2*hidden_size]
        :return: probs: a tensor of shape [target_vocab_size]
        """
        linear_transform = torch.mv(self.weight, input) + self.bias
        return softmax(linear_transform, dim=0)


class Attention(object):

    def __init__(self, A: torch.tensor):
        """
        :param A: a tensor of shape [hidden_dim, hidden_dim]
        """
        self.A = A

    def forward(self, source_top_hiddens: torch.tensor, target_top_hidden: torch.tensor):
        """
        :param source_top_hiddens: tensor of shape [source_sentence_length, hidden_dim] - the hidden states
                                   from the TOP LAYER of the encoder
        :param target_top_hidden: tensor of shape [hidden_dim] - the current hidden state in the TOP LAYER of the decoder
        :return: a tensor of attention weights of shape [source_sentence_length]

        """
        attention_scores = torch.mv(source_top_hiddens,torch.mv(self.A, target_top_hidden))
        return softmax(attention_scores,dim=0)


class Seq2SeqModel(object):

    def __init__(self, hidden_dim: int, encoder: StackedGRU, decoder: StackedGRU,
                 source_embedding_matrix: torch.tensor, target_embedding_matrix: torch.tensor,
                 output_layer: OutputLayer):
        """
        :param encoder: the encoder StackedGRU, with input dim `source_embedding_dim`, and hidden dim `hidden_dim`
        :param decoder: the decoder StackedGRU, with input dim `target_embedding_dim` and hidden dim `hidden_dim`
        :param source_embedding_matrix: a tensor of shape [source_vocab_size, source_embedding_dim]
        :param target_embedding_matrix: a tensor of shape [target_vocab_size, target_embedding_dim]
        :param output_layer: an OutputLayer with input dimension `hidden_dim` and output dimension `target_vocab_size`
        """
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.decoder_gru = decoder
        self.source_embedding_matrix = source_embedding_matrix
        self.target_embedding_matrix = target_embedding_matrix
        self.output_layer = output_layer


class Seq2SeqAttentionModel(object):

    def __init__(self, hidden_dim: int, encoder: StackedGRU, decoder: StackedGRU,
                 source_embedding_matrix: torch.tensor, target_embedding_matrix: torch.tensor,
                 attention: Attention, output_layer: OutputLayer):
        """
        :param encoder: the encoder StackedGRU, with input dim `source_embedding_dim`, and hidden dim `hidden_dim`
        :param decoder: the decoder StackedGRU, with input dim `target_embedding_dim + hidden_dim`
                        and hidden dim `hidden_dim`
        :param source_embedding_matrix: a tensor of shape [source_vocab_size, source_embedding_dim]
        :param target_embedding_matrix: a tensor of shape [target_vocab_size, target_embedding_dim]
        :param attention: an Attention
        :param output_layer: an OutputLayer with input dimension `2*hidden_dim` and output dimension `target_vocab_size`
        """
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.decoder_gru = decoder
        self.source_embedding_matrix = source_embedding_matrix
        self.target_embedding_matrix = target_embedding_matrix
        self.attention = attention
        self.output_layer = output_layer


def encode_all(source_sentence: List[int], model: Union[Seq2SeqModel, Seq2SeqAttentionModel]) -> torch.tensor:
    """ Encode the whole source sentence.

    :param source_sentence: list of ints of length `source_sentence_length`
    :param model: either a Seq2SeqModel or a Seq2SeqAttentionModel
    :return: tensor `source_hiddens` of shape [source_sentence_length, L, hidden_dim], denoted H^{enc}_1 ... H^{enc}_S
             in the assignment
    """
    # input of shape seq_len x embedding_size
    input_embeddings = model.source_embedding_matrix[source_sentence]
    stack_size = len(model.encoder.grus)
    # stack x hid_dim
    last_hidden = torch.zeros((stack_size, model.hidden_dim), dtype=torch.float)
    hiddens = []
    for pos in range(input_embeddings.shape[0]):
        last_hidden = model.encoder.forward(input_embeddings[pos], last_hidden)
        hiddens.append(last_hidden)

    return torch.stack(hiddens)
