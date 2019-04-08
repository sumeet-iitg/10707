import argparse
import time
import torch
from typing import *
from util import load_data, initialize_seq2seq_params, build_seq2seq_model
from time import time
from torch import optim
from core import Seq2SeqModel, encode_all, SOS_token, EOS_token
from math import exp


def decode(prev_hidden: torch.tensor, input: int, model: Seq2SeqModel) -> (torch.tensor, torch.tensor):
    """ Run the decoder AND the output layer for a single step.

    (This function will be used in both log_likelihood and translate_greedy_search.)

    :param prev_hidden: tensor of shape [L, hidden_dim] - the decoder's previous hidden state, denoted H^{dec}_{t-1}
                        in the assignment
    :param input: int - the word being inputted to the decoder.
                            during log-likelihood computation, this is y_{t-1}
                            during greedy decoding, this is yhat_{t-1}
    :param model: a Seq2Seq model
    :return: (1) a tensor `probs` of shape [target_vocab_size], denoted p(y_t | x_1 ... x_S, y_1 .. y_{t-1})
             (2) a tensor `hidden` of shape [L, hidden_dim], denoted H^{dec}_t in the assignment
    """
    decode_in = model.target_embedding_matrix[input]
    hidden_out = model.decoder_gru.forward(decode_in, prev_hidden)
    log_probs = model.output_layer.forward(hidden_out[-1])
    return log_probs, hidden_out

def log_likelihood(source_sentence: List[int], target_sentence: List[int], model: Seq2SeqModel) -> torch.Tensor:
    """ Compute the log-likelihood for a (source_sentence, target_sentence) pair.

    :param source_sentence: the source sentence, as a list of words
    :param target_sentence: the target sentence, as a list of words
    :return: conditional log-likelihood of the (source_sentence, target_sentence) pair
    """
    encoder_hiddens = encode_all(source_sentence, model)
    # input of shape seq_len x embedding_size
    target_sentence = [SOS_token] + target_sentence
    target_embeddings = model.target_embedding_matrix[target_sentence]
    # stack x hid_dim
    prev_hidden = encoder_hiddens[-1]
    target_log_probs = []

    for pos in range(len(target_sentence) - 1):
        log_probs, prev_hidden = decode(prev_hidden, target_sentence[pos], model)
        target_log_probs.append(torch.log(log_probs[target_sentence[pos+1]]))

    return torch.sum(torch.stack(target_log_probs))

@torch.no_grad()
def translate_greedy_search(source_sentence: List[int], model: Seq2SeqModel, max_length=10) -> List[int]:
    """ Translate a source sentence using greedy decoding.

    :param source_sentence: the source sentence, as a list of words
    :param max_length: the maximum length that the target sentence could be
    :return: the translated sentence as a list of word ints
    """

    encoder_hiddens = encode_all(source_sentence, model)
    decode_in = SOS_token
    prev_hidden = encoder_hiddens[-1]
    translate_out = []
    for i in range(max_length):
        log_probs, prev_hidden = decode(prev_hidden, decode_in, model)
        tgt_out = int(torch.argmax(log_probs).item())
        decode_in = tgt_out
        translate_out.append(tgt_out)
        if tgt_out == EOS_token:
            break

    return translate_out

@torch.no_grad()
def perplexity(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqModel):
    """ Compute the perplexity of an entire dataset under a seq2seq model.  Refer to the write-up for the
    definition of perplexity.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: seq2seq model
    :return: perplexity of the dataset
    """

    LL_Total = torch.tensor(0,dtype=torch.float)
    total_words = torch.tensor(0,dtype=torch.float)
    for i, (source_sentence, target_sentence) in enumerate(sentences):
        LL_Total += log_likelihood(source_sentence, target_sentence, model)
        total_words += len(target_sentence)

    return torch.exp(-LL_Total/total_words)

@torch.enable_grad()
def train_epoch(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqModel,
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


def print_translations(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqModel,
                       source_vocab: Dict[int, str], target_vocab: Dict[int, str]):
    """ Iterate through a dataset, printing (1) the source sentence, (2) the actual target sentence, and (3)
    the translation according to our model.

    :param sentences: a list of (source sentence, target sentence) pairs
    :param model: a Seq2Seq model
    :param source_vocab: the mapping from word index to word string, in the source language
    :param target_vocab: the mapping from word index to word string, in the target language
    """
    for (source_sentence, target_sentence) in sentences:
        translation = translate_greedy_search(source_sentence, model)

        print("source sentence:" + " ".join([source_vocab[word] for word in source_sentence]))
        print("target sentence:" + " ".join([target_vocab[word] for word in target_sentence]))
        print("translation:\t" + " ".join([target_vocab[word] for word in translation]))
        print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Seq2Seq Homework Assignment')
    parser.add_argument("action", type=str,
                        choices=["train", "finetune", "train_perplexity", "test_perplexity",
                                 "print_train_translations", "print_test_translations"])
    parser.add_argument("--load_model", type=str,
                        help="path to saved model on disk.  if this arg is unset, the weights are initialized randomly",
                        default="pretrained/seq2seq.pth")
    parser.add_argument("--save_model_prefix", type=str, help="prefix to save model with, if you're training",
                        default="trained/train_seq2seq")
    args = parser.parse_args()

    # load train/test data, and source/target vocabularies
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    # load model weights (if path is specified) or else initialize weights randomly
    model_params = initialize_seq2seq_params() if args.load_model is None \
        else torch.load(args.load_model)  # type: Dict[str, torch.Tensor]

    # build a Seq2SeqModel object
    model = build_seq2seq_model(model_params)  # type: Seq2SeqModel

    if args.action == 'train':
        for epoch in range(10):
            train_epoch(train_sentences, model, epoch)
            torch.save(model_params, '{}_{}.pth'.format(args.save_model_prefix, epoch))
    elif args.action == 'finetune':
        train_epoch(train_sentences[:1000], model, 0, learning_rate=1e-5)
        torch.save(model_params, "{}.pth".format(args.save_model_prefix))
    elif args.action == "print_train_translations":
        print_translations(train_sentences, model, source_vocab, target_vocab)
    elif args.action == "print_test_translations":
        print_translations(test_sentences, model, source_vocab, target_vocab)
    elif args.action == "train_perplexity":
        print("perplexity: {}".format(perplexity(train_sentences[:1000], model)))
    elif args.action == "test_perplexity":
        print("perplexity: {}".format(perplexity(test_sentences, model)))
