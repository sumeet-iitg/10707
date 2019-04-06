import argparse
import time
import torch
from typing import *
from util import initialize_seq2seq_attention_params, build_seq2seq_attention_model, load_data
from time import time
from torch import optim
from core import Seq2SeqAttentionModel, encode_all
from math import exp
from core import SOS_token, EOS_token


def decode(prev_hidden: torch.tensor, source_hiddens: torch.tensor, prev_context: torch.tensor,
           input: int, model: Seq2SeqAttentionModel) -> (
        torch.tensor, torch.tensor, torch.tensor, torch.tensor):
    """ Run the decoder AND the output layer for a single step.

    :param: prev_hidden: tensor of shape [L, hidden_dim] - the decoder's previous hidden state, denoted H^{dec}_{t-1}
                          in the assignment
    :param: source_hiddens: tensor of shape [source sentence length, L, hidden_dim] - the encoder's hidden states,
                            denoted H^{enc}_1 ... H^{enc}_T in the assignment
    :param: prev_context: tensor of shape [hidden_dim], denoted c_{t-1} in the assignment
    :param input: int - the word being inputted to the decoder.
                            during log-likelihood computation, this is y_{t-1}
                            during greedy decoding, this is yhat_{t-1}
    :param model: a Seq2SeqAttention model
    :return: (1) a tensor `probs` of shape [target_vocab_size], denoted p(y_t | x_1 ... x_S, y_1 .. y_{t-1})
             (2) a tensor `hidden` of shape [L, hidden_size], denoted H^{dec}_t in the assignment
             (3) a tensor `context` of shape [hidden_size], denoted c_t in the assignment
             (4) a tensor `attention_weights` of shape [source_sentence_length], denoted \alpha in the assignment
    """

    decode_in = torch.stack(model.target_embedding_matrix[input], prev_context)
    hidden_out = model.decoder_gru.forward(decode_in, prev_hidden)
    attention_weights = model.attention.forward(source_hiddens, hidden_out[-1])
    context = torch.mul(attention_weights, source_hiddens)
    log_probs = model.output_layer.forward(torch.stack(hidden_out[-1],context))
    return log_probs, hidden_out, context, attention_weights


def log_likelihood(source_sentence: List[int],
                   target_sentence: List[int],
                   model: Seq2SeqAttentionModel) -> torch.Tensor:
    """ Compute the log-likelihood for a (source_sentence, target_sentence) pair.

    :param source_sentence: the source sentence, as a list of words
    :param target_sentence: the target sentence, as a list of words
    :return: log-likelihood of the (source_sentence, target_sentence) pair
    """
    raise NotImplementedError()



def perplexity(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel) -> float:
    """ Compute the perplexity of an entire dataset under a seq2seq model.  Refer to the write-up for the
    definition of perplexity.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: seq2seq attention model
    :return: perplexity of the translation
    """
    raise NotImplementedError()



def translate_greedy_search(source_sentence: List[int],
                            model: Seq2SeqAttentionModel, max_length=10) -> (List[int], torch.tensor):
    """ Translate a source sentence using greedy decoding.

    :param source_sentence: the source sentence, as a list of words
    :param max_length: the maximum length that the target sentence could be
    :return: (1) the translated sentence as a list of ints
             (2) the attention matrix, a tensor of shape [target_sentence_length, source_sentence_length]

    """
    raise NotImplementedError()


def translate_beam_search(source_sentence: List[int], model: Seq2SeqAttentionModel,
                          beam_width: int, max_length=10) -> Tuple[List[int], float]:
    """ Translate a source sentence using beam search decoding.

    :param beam_width: the number of translation candidates to keep at each time step
    :param max_length: the maximum length that the target sentence could be
    :return: (1) the target sentence (translation),
             (2) sum of conditional log-likelihood of the translation, i.e., log p(target sentence|source sentence)
    """
    raise NotImplementedError("this is extra credit")


def train_epoch(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel,
                epoch: int, print_every: int = 100, learning_rate: float = 0.0001, gradient_clip=5):
    """ Train the model for an epoch.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: a Seq2Seq model
    :param epoch: which epoch we're at
    """
    print("epoch\titer\tavg loss\telapsed secs")
    total_loss = 0
    start_time = time()
    optimizer = optim.Adam(model_params.values(), lr=learning_rate)
    for i, (source_sentence, target_sentence) in enumerate(sentences):
        optimizer.zero_grad()
        theloss = -log_likelihood(source_sentence, target_sentence, model)
        total_loss += theloss
        theloss.backward()

        torch.nn.utils.clip_grad_norm_(model_params.values(), gradient_clip)

        optimizer.step()

        if i % print_every == 0:
            avg_loss = total_loss / print_every
            total_loss = 0
            elapsed_secs = time() - start_time
            print("{}\t{}\t{:.3f}\t{:.3f}".format(epoch, i, avg_loss, elapsed_secs))

    return model_params


def print_translations(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel,
                       source_vocab: Dict[int, str], target_vocab: Dict[int, str], beam_width):
    """ Iterate through a dataset, printing (1) the source sentence, (2) the actual target sentence, and (3)
    the translation according to our model.

    :param sentences: a list of (source sentence, target sentence) pairs
    :param model: a Seq2Seq model
    :param source_vocab: the mapping from word index to word string, in the source language
    :param target_vocab: the mapping from word index to word string, in the target language
    """
    for (source_sentence, target_sentence) in sentences:
        if beam_width > 0:
            translation, _ = translate_beam_search(source_sentence, model, beam_width)
        else:
            translation, _ = translate_greedy_search(source_sentence, model)

        print("source sentence:" + " ".join([source_vocab[word] for word in source_sentence]))
        print("target sentence:" + " ".join([target_vocab[word] for word in target_sentence]))
        print("translation:\t" + " ".join([target_vocab[word] for word in translation]))
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Seq2Seq Homework Assignment')
    parser.add_argument("action", type=str,
                        choices=["train", "finetune", "train_perplexity", "test_perplexity",
                                 "print_train_translations", "print_test_translations", "visualize_attention"])
    parser.add_argument("--beam_width", type=int, default=-1,
                        help="number of translation candidates to keep at each time step. "
                             "this option is used for beam search translation (greedy search decoding is used by default).")
    parser.add_argument("--load_model", type=str,
                        help="path to saved model on disk.  if this arg is unset, the weights are initialized randomly")
    parser.add_argument("--save_model_prefix", type=str, help="prefix to save model with, if you're training")
    args = parser.parse_args()

    # load train/test data, and source/target vocabularies
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    # load model weights (if path is specified) or else initialize weights randomly
    model_params = initialize_seq2seq_attention_params() if args.load_model is None \
        else torch.load(args.load_model)  # type: Dict[str, torch.Tensor]

    # build a Seq2SeqAttentionModel object
    model = build_seq2seq_attention_model(model_params)  # type: Seq2SeqAttentionModel

    if args.action == 'train':
        for epoch in range(10):
            train_epoch(train_sentences, model, epoch)
            torch.save(model_params, '{}_{}.pth'.format(args.save_model_prefix, epoch))
    elif args.action == 'finetune':
        train_epoch(train_sentences[:1000], model, 0, learning_rate=1e-5)
        torch.save(model_params, "{}.pth".format(args.save_model_prefix))
    elif args.action == "print_train_translations":
        print_translations(train_sentences, model, source_vocab, target_vocab, args.beam_width)
    elif args.action == "print_test_translations":
        print_translations(test_sentences, model, source_vocab, target_vocab, args.beam_width)
    elif args.action == "train_perplexity":
        print("perplexity: {}".format(perplexity(train_sentences[:1000], model)))
    elif args.action == "test_perplexity":
        print("perplexity: {}".format(perplexity(test_sentences, model)))
    elif args.action == "visualize_attention":
        from plotting import visualize_attention

        # visualize the attention matrix for the first 5 test set sentences
        for i in range(5):
            source_sentence = test_sentences[i][0]
            predicted_sentence, attention_matrix = translate_greedy_search(source_sentence, model)
            source_sentence_str = [source_vocab[w] for w in source_sentence]
            predicted_sentence_str = [target_vocab[w] for w in predicted_sentence]
            visualize_attention(source_sentence_str, predicted_sentence_str,
                                attention_matrix.detach().numpy(), "images/{}.png".format(i))
