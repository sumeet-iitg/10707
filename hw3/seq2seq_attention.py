import argparse
import time
import torch
from typing import *
from util import initialize_seq2seq_attention_params, build_seq2seq_attention_model, load_data
from time import time
from torch import optim
from core import Seq2SeqAttentionModel, encode_all
import math
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

    decode_in = torch.cat((model.target_embedding_matrix[input], prev_context))
    hidden_out = model.decoder_gru.forward(decode_in, prev_hidden)
    # passing the top layer of encoder and decoder hidden dims
    attention_weights = model.attention.forward(source_hiddens[:,-1,:], hidden_out[-1])
    context = torch.mm(attention_weights.unsqueeze(dim=0),source_hiddens[:,-1,:]).squeeze()
    log_probs = model.output_layer.forward(torch.cat((hidden_out[-1].squeeze(),context)))
    return log_probs, hidden_out, context, attention_weights


def log_likelihood(source_sentence: List[int],
                   target_sentence: List[int],
                   model: Seq2SeqAttentionModel) -> torch.Tensor:
    """ Compute the log-likelihood for a (source_sentence, target_sentence) pair.

    :param source_sentence: the source sentence, as a list of words
    :param target_sentence: the target sentence, as a list of words
    :return: log-likelihood of the (source_sentence, target_sentence) pair
    """
    encoder_hiddens = encode_all(source_sentence, model)
    # input of shape seq_len x embedding_size
    target_sentence = [SOS_token] + target_sentence
    # stack x hid_dim
    prev_hidden = encoder_hiddens[-1]
    prev_context = torch.zeros(model.hidden_dim)
    target_log_probs = []

    for pos in range(len(target_sentence) - 1):
        log_probs, prev_hidden, prev_context,_ = decode(prev_hidden, encoder_hiddens, prev_context, target_sentence[pos], model)
        target_log_probs.append(torch.log(log_probs[target_sentence[pos + 1]]))

    return torch.sum(torch.stack(target_log_probs))


@torch.no_grad()
def perplexity(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel) -> float:
    """ Compute the perplexity of an entire dataset under a seq2seq model.  Refer to the write-up for the
    definition of perplexity.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: seq2seq attention model
    :return: perplexity of the translation
    """
    LL_Total = torch.tensor(0, dtype=torch.float)
    total_words = torch.tensor(0, dtype=torch.float)
    for i, (source_sentence, target_sentence) in enumerate(sentences):
        LL_Total += log_likelihood(source_sentence, target_sentence, model)
        total_words += len(target_sentence)

    return torch.exp(-LL_Total / total_words)

@torch.no_grad()
def translate_greedy_search(source_sentence: List[int],
                            model: Seq2SeqAttentionModel, max_length=10) -> (List[int], torch.tensor):
    """ Translate a source sentence using greedy decoding.

    :param source_sentence: the source sentence, as a list of words
    :param max_length: the maximum length that the target sentence could be
    :return: (1) the translated sentence as a list of ints
             (2) the attention matrix, a tensor of shape [target_sentence_length, source_sentence_length]

    """
    encoder_hiddens = encode_all(source_sentence, model)
    # stack x hid_dim
    prev_hidden = encoder_hiddens[-1]
    prev_context = torch.zeros(model.hidden_dim)
    decode_in = SOS_token
    translate_out = []
    attention_wt_list = []
    for i in range(max_length):
        log_probs, prev_hidden, prev_context, attention_weights = decode(prev_hidden, encoder_hiddens, prev_context, decode_in, model)
        decode_in = int(torch.argmax(log_probs).item())
        translate_out.append(decode_in)
        attention_wt_list.append(attention_weights)

    return translate_out, torch.stack(attention_wt_list)


def translate_beam_search(source_sentence: List[int], model: Seq2SeqAttentionModel,
                          beam_width: int, max_length=10) -> Tuple[List[int], float]:
    """ Translate a source sentence using beam search decoding.

    :param beam_width: the number of translation candidates to keep at each time step
    :param max_length: the maximum length that the target sentence could be
    :return: (1) the target sentence (translation),
             (2) sum of conditional log-likelihood of the translation, i.e., log p(target sentence|source sentence)
    """
    encoder_hiddens = encode_all(source_sentence, model)
    beam_elems = []
    # stack x hid_dim
    prev_hidden = encoder_hiddens[-1]
    prev_context = torch.zeros(model.hidden_dim)

    beam_elems= [([SOS_token], float(0), prev_hidden, prev_context)]
    candidate_translations = []
    available_width = beam_width
    for i in range(max_length):
        if available_width >0:
            candidate_beam_elems = []
            for b in range(len(beam_elems)):
                prev_predict, prev_log_prob, prev_hidden, prev_context = beam_elems[b]
                log_probs, prev_hidden, prev_context, _ = decode(prev_hidden, encoder_hiddens, prev_context,
                                                                                 prev_predict[-1], model)
                top_log_probs, top_preds = torch.topk(log_probs,available_width)
                for k in range(len(top_log_probs)):
                    curr_log_prob = prev_log_prob + top_log_probs[k].item()
                    curr_pred_list = prev_predict + top_preds[k].item()
                    candidate = (curr_pred_list, curr_log_prob, prev_hidden, prev_context)
                    candidate_pos = -1
                    for pos in range(len(candidate_beam_elems)):
                        if curr_log_prob > candidate_beam_elems[pos][1]:
                            candidate_pos = pos
                    if not candidate_pos == -1:
                        candidate_beam_elems.insert(candidate_pos+1, candidate)
                    elif len(candidate_beam_elems) < available_width:
                        candidate_beam_elems.append(candidate)
                    if len(candidate_beam_elems) > available_width:
                        candidate_beam_elems.pop()

            beam_elems = []
            for candidate in candidate_beam_elems:
                if candidate[0][-1] == EOS_token:
                    candidate_translations.append(candidate)
                    available_width -= 1
                else:
                    beam_elems.append(candidate)

    max_prob = -math.inf
    best_elem = -1
    for pos in range(len(candidate_translations)):
        norm_prob = candidate_translations[pos][1]/len(candidate_translations[pos][0])
        if norm_prob > max_prob:
            max_prob = norm_prob
            best_elem = pos

    return candidate_translations[best_elem][0]

@torch.enable_grad()
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
