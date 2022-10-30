"""
NLP, assignment 4, 2021 - Omer Keidar

In this assignment you will implement a Hidden Markov model and an LSTM model
to predict the part of speech sequence for a given sentence.
(Adapted from Nathan Schneider)

"""

import torch
import torch.nn as nn
import torchtext
from torchtext import data,legacy
import torch.optim as optim
from math import log, isfinite
from collections import Counter
import numpy as np
import sys, os, time, platform, nltk, random
import copy

# With this line you don't need to worry about the HW  -- GPU or CPU
# GPU cuda cores will be used if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You can call use_seed with other seeds or None (for complete randomization)
# but DO NOT change the default value.
def use_seed(seed = 2512021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_deterministic(True)
    # torch.use_deterministic_algorithms(True)
    #torch.backends.cudnn.deterministic = True

# utility functions to read the corpus
def who_am_i(): #this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Omer Keidar', 'id': '307887984', 'email': 'omerkei@post.bgu.ac.il'}

def read_annotated_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
        line = f.readline()
    return sentence

def load_annotated_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_annotated_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_annotated_sentence(f)
    return sentences


START = "<DUMMY_START_TAG>"
END = "<DUMMY_END_TAG>"
UNK = "<UNKNOWN>"
allTagCounts = Counter()
# use Counters inside these
perWordTagCounts = {}
transitionCounts = {}
emissionCounts = {}
# log probability distributions: do NOT use Counters inside these because
# missing Counter entries default to 0, not log(0)
A = {} #transisions probabilities
B = {} #emmissions probabilities
vocabulary_list = [] # list of all words from the data
alpha_for_smooth = 0.1


def learn_params(tagged_sentences):

    """Populates and returns the allTagCounts, perWordTagCounts, transitionCounts,
     and emissionCounts data-structures.
    allTagCounts and perWordTagCounts should be used for baseline tagging and
    should not include pseudocounts, dummy tags and unknowns.
    The transisionCounts and emmisionCounts
    should be computed with pseudo tags and shoud be smoothed.
    A and B should be the log-probability of the normalized counts, based on
    transisionCounts and  emmisionCounts
    Args:
      tagged_sentences: a list of tagged sentences, each tagged sentence is a
       list of pairs (w,t), as retunred by load_annotated_corpus().
   Return:
      [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B] (a list)
  """
    global allTagCounts, perWordTagCounts, transitionCounts, emissionCounts, A, B, vocabulary_list
    # list of all words in the data
    vocabulary = set([word[0].lower() for sentence in tagged_sentences for word in sentence])
    vocabulary_list = list(vocabulary) #make list from the set of vocabulary
    # dict of all tags and their frequency
    allTagCounts = dict(Counter([tag[1] for sentence in tagged_sentences for tag in sentence]))
    # make a dict of all words with possible tags and their frequency
    perWordTagCounts = dict(Counter([(word_tag[0].lower(), word_tag[1]) for sentence in tagged_sentences for word_tag in sentence]))

    tagged_sentences_copy = copy.deepcopy(tagged_sentences)

    tagged_pad_sentences = padding_start_end(tagged_sentences_copy) #padding all sentences with START and END.

    # make dict of all tags and their frequency from the tagged sentences with padding
    allTagCountsWithDummies = dict(Counter([tag[1] for sentence in tagged_pad_sentences for tag in sentence]))

    #make a dict of all bigram tags and their frequency - Count(t_i-1 , t_i)
    transitionCounts = dict(Counter([(sentence[i][1],sentence[i+1][1]) for sentence in tagged_pad_sentences for i in range(len(sentence) - 1)]))
    # make a dict of all words with possible tags and their frequency - Count(wi,ti)
    emissionCounts = dict(Counter([(word_tag[0].lower(), word_tag[1]) for sentence in tagged_pad_sentences for word_tag in sentence]))

    # Calculate and build A and B dictionaries
    A, B = make_A_B_dicts(allTagCounts, allTagCountsWithDummies, transitionCounts, emissionCounts)

    return [allTagCounts,perWordTagCounts,transitionCounts,emissionCounts,A,B]


def padding_start_end(tagged_sent):
    """
    This function pad all the sentences with START and END
    Args:
      tagged_sent: a list of tagged sentences, each tagged sentence is alist of pairs (w,t)
    Return:
         all the tagged sentences with padding
    """
    tagged_sentences = tagged_sent
    for sentence in tagged_sentences:
        sentence.insert(0,(START,'START'))
        sentence.append((END,'END'))
    return tagged_sentences

def make_A_B_dicts(allTagsCount, allTagCountsWithDum, transCount, emissCount):
    """
    This function make the A and B dictionaries by calculate them using transitionCount and emissionCount.
    This function make a lists of all the options for tuples of tags (tag1,tag2) and all the options for (word,tag)
    and then calculate the probabilities and make A and B.
    Args:
      allTagsCount: dictionary of all tags and their frequency
      transCount: dictionary with all of the transition and their frequency
      emissCount: dictionary with all of the (word, tag) and their frequency
    Return:
         A, B dictionaries
    """

    # Make all the options for (tag1,tag2) from the data
    all_tuples_tags_options = [(tag1, tag2) for tag1 in list(allTagsCount.keys()) + ['START'] for tag2 in list(allTagsCount.keys()) + ['END'] ]
    # Make all options for (word, tag) from data
    all_words_tags_options = [(word, tag) for word in vocabulary_list + [UNK] for tag in allTagsCount.keys()]

    # Make A - transitions probabilities, P(t_i|t_i-1)
    A = copy.deepcopy(transCount)
    for tags in all_tuples_tags_options:
        trans_numerator = transCount.get(tags,0) + alpha_for_smooth # C(t_i-1, t_i)
        trans_denominator = allTagCountsWithDum.get(tags[0], 0) + alpha_for_smooth # C(t_i-1)
        A[tags] = np.log( trans_numerator / trans_denominator) # Calculate log probability of transition

    # # Make B - emissions probabilities, P(w_i|t_i)

    B = copy.deepcopy(emissCount)
    for word_tag in all_words_tags_options:
        if word_tag in B.keys() or word_tag[0] == UNK:
            emiss_numerator = emissCount.get(word_tag, 0) + alpha_for_smooth  # C(ti,wi)
            emiss_denominator = allTagCountsWithDum.get(word_tag[1],0) + alpha_for_smooth # C(ti)
            B[word_tag] = np.log( emiss_numerator / emiss_denominator ) # Calculate log probability of emission
    B[(START, 'START')] = 0
    B[(END, 'END')] = 0

    return A, B

def baseline_tag_sentence(sentence, perWordTagCounts, allTagCounts):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Each word is tagged by the tag most
    frequently associated with it. OOV words are tagged by sampling from the
    distribution of all tags.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        Return:
        list: list of pairs
    """
    sentence_lower = copy.deepcopy(sentence)
    for i in range(len(sentence_lower)):
        sentence_lower[i] = sentence_lower[i].lower()

    tagged_sentence = []
    for word in sentence_lower:
        if word in vocabulary_list: # the word is in the vocabulary
            # make dictionary of optionally tags for each word and take the tag with the max frequency
            tag_options_for_word_dict = {tag: perWordTagCounts[(word,tag)] for tag in allTagCounts if (word,tag) in perWordTagCounts}
            tag = max(tag_options_for_word_dict , key=tag_options_for_word_dict.get)
            tagged_sentence.append((word, tag))
        else: #OOV word - sampling from the distribution
            chosen_tag = (random.choices(list(allTagCounts.keys()), weights=list(allTagCounts.values()), k=1))[0]
            tagged_sentence.append((word, chosen_tag))

    return tagged_sentence


#===========================================
#       POS tagging with HMM
#===========================================


def hmm_tag_sentence(sentence, A, B):
    """Returns a list of pairs (w,t) where each w corresponds to a word
    (same index) in the input sentence. Tagging is done with the Viterbi
    algorithm.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): The HMM Emission probabilities.

    Return:
        list: list of pairs
    """
    sentence_lower = copy.deepcopy(sentence)

    for i in range(len(sentence_lower)):
        sentence_lower[i] = sentence_lower[i].lower()

    sentence_with_pad_and_UNK = []

    #make new sentence from the current sentence by replace the OOV words with UNK
    for word in sentence_lower:
        if word in vocabulary_list:
            sentence_with_pad_and_UNK.append(word)
        else:
            sentence_with_pad_and_UNK.append(UNK)
    #add Dummy END , i dont pad with START because i add it in the viterbi matrix.
    sentence_with_pad_and_UNK.append(END)

    v_last = viterbi(sentence_with_pad_and_UNK, A, B)
    tags_for_sentence = retrace(v_last)

    tagged_sentence = []
    for index, word in enumerate(sentence_lower):
        tagged_sentence.append((word,tags_for_sentence[index]))

    return tagged_sentence

def viterbi(sentence, A,B):
    """Creates the Viterbi matrix, column by column. Each column is a list of
    tuples representing cells. Each cell ("item") is a tupple (t,r,p), were
    t is the tag being scored at the current position,
    r is a reference to the corresponding best item from the previous position,
    and p is a log probabilityof the sequence so far).

    The function returns the END item, from which it is possible to
    trace back to the beginning of the sentence.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

    Return:
        obj: the last item, tagged with END. should allow backtraking.

        """

    viterbi_matrix = [[('START', None, 0)]] # each sentence start with tag START , without backpointer and with prob 0.

    for word in sentence:
        matrix_columns = []
        # Build a list of optional tags depending on whether the word is UNK or not.
        if word != UNK: # word is not OOV - only tags that seen with that word.
            optionally_tags = [tag for tag in (list(allTagCounts.keys())+['END', 'START']) if (word,tag) in B]
        else: # word is OOV - all the tags
            optionally_tags = list(allTagCounts.keys())
        for tag in optionally_tags:
            transsiton_options_list = []
            for prev_tag in viterbi_matrix[-1]:
                key_for_A = (prev_tag[0], tag)
                key_for_B = (word , tag)
                tuple_t_r_p = (tag, prev_tag, prev_tag[2] + A[key_for_A] + B[key_for_B]) #calculate prob and build tuple (t,r,p)
                transsiton_options_list.append(tuple_t_r_p)
            matrix_columns.append(max(transsiton_options_list, key=lambda prob: prob[2]))

        viterbi_matrix.append(matrix_columns)
    # print(max(viterbi_matrix[-1], key=lambda prob: prob[2]))

    v_last = max(viterbi_matrix[-1], key=lambda prob: prob[2])

    return v_last


def retrace(end_item):
    """Returns a list of tags (retracing the sequence with the highest probability,
        reversing it and returning the list). The list should correspond to the
        list of words in the sentence (same indices).
    """
    prev_tag = end_item
    tags_list = [end_item[0]]
    while tags_list[-1] != 'START':
        prev_tag = prev_tag[1]
        tags_list.append(prev_tag[0])
    tags_list.remove('START')
    tags_list.reverse()

    return tags_list

def joint_prob(sentence, A, B):
    """Returns the joint probability of the given sequence of words and tags under
     the HMM model.

     Args:
         sentence (pair): a sequence of pairs (w,t) to compute.
         A (dict): The HMM Transition probabilities
         B (dict): the HMM emmission probabilities.
     """
    p = 0   # joint log prob. of words and tags
    sentence_with_pad_and_UNK = []
    for word_tag in sentence:
        if word_tag[0] in vocabulary_list:
            sentence_with_pad_and_UNK.append(word_tag)
        else:
            sentence_with_pad_and_UNK.append((UNK,word_tag[1]))
    #add Dummy END.
    sentence_with_pad_and_UNK.append((END, 'END'))

    prev_tag = 'START'
    for word_tag in sentence_with_pad_and_UNK:
        word = word_tag[0]
        tag = word_tag[1]
        p += A[(prev_tag, tag)] + B[(word, tag)]
        prev_tag = tag

    assert isfinite(p) and p<0  # Should be negative. Think why!
    return p


#===========================================
#       POS tagging with BiLSTM
#===========================================

""" You are required to support two types of bi-LSTM:
    1. a vanilla biLSTM in which the input layer is based on simple word embeddings
    2. a case-based BiLSTM in which input vectors combine a 3-dim binary vector
        encoding case information, see
        https://arxiv.org/pdf/1510.06168.pdf
"""

# Suggestions and tips, not part of the required API
#
#  1. You can use PyTorch torch.nn module to define your LSTM, see:
#     https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
#  2. You can have the BLSTM tagger model(s) implemented in a dedicated class
#     (this could be a subclass of torch.nn.Module)
#  3. Think about padding.
#  4. Consider using dropout layers
#  5. Think about the way you implement the input representation
#  6. Consider using different unit types (LSTM, GRU,LeRU)

class BLSTM (nn.Module):
    """
    Vanilla biLSTM, based on simple word embeddings
    """
    def __init__(self, embedding_dim, output_dim, num_of_layers, vocabulary_size, pad_idx, hidden_dim=128, dropout=0.25):
        # constructor
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim, padding_idx=pad_idx)
        # Blstm - model
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_of_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout = dropout if num_of_layers > 1 else 0)

        self.tag_layer = nn.Linear(hidden_dim*2, output_dim+1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, text):
        """
        Args:
            text: input text
        """
        embedded = self.dropout(self.embedding(text))
        outputs, _ = self.lstm(embedded)
        tag_predictions = self.tag_layer(self.dropout(outputs))

        return tag_predictions

    def add_stoi(self, stoi):
        self.stoi = stoi

    def add_tag_itos(self, tag_itos):
        self.tag_itos = tag_itos


class CBLSTM (nn.Module):
    """
     CBLSTM, a case-based BiLSTM in which input vectors combine a 3-dim binary vector
    """
    def __init__(self, embedding_dim, output_dim, num_of_layers, vocabulary_size, pad_idx, hidden_dim=128, dropout=0.25):
        # constructor
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_dim, padding_idx=pad_idx)
        # Blstm - model
        self.lstm = nn.LSTM(input_size=embedding_dim + 3,
                            hidden_size=hidden_dim,
                            num_layers=num_of_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_of_layers > 1 else 0)

        self.tag_layer = nn.Linear(hidden_dim*2, output_dim+1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, text, features):
        """
        Args:
            text: input text
        """

        embedded = self.dropout(self.embedding(text))
        embedded = torch.cat([embedded, features], dim=-1)
        outputs, _ = self.lstm(embedded)
        tag_predictions = self.tag_layer(self.dropout(outputs))

        return tag_predictions

    def add_stoi(self, stoi):
        self.stoi = stoi

    def add_tag_itos(self, tag_itos):
        self.tag_itos = tag_itos



def initialize_rnn_model(params_d):
    """Returns a dictionary with the objects and parameters needed to run/train_rnn
       the lstm model. The LSTM is initialized based on the specified parameters.
       thr returned dict is may have other or additional fields.

    Args:
        params_d (dict): a dictionary of parameters specifying the model. The dict
                        should include (at least) the following keys:
                        {'max_vocab_size': max vocabulary size (int),
                        'min_frequency': the occurence threshold to consider (int),
                        'input_rep': 0 for the vanilla and 1 for the case-base (int),
                        'embedding_dimension': embedding vectors size (int),
                        'num_of_layers': number of layers (int),
                        'output_dimension': number of tags in tagset (int),
                        'pretrained_embeddings_fn': str,
                        'data_fn': str
                        }
                        max_vocab_size sets a constraints on the vocab dimention.
                            If the its value is smaller than the number of unique
                            tokens in data_fn, the words to consider are the most
                            frequent words. If max_vocab_size = -1, all words
                            occuring more that min_frequency are considered.
                        min_frequency privides a threshold under which words are
                            not considered at all. (If min_frequency=1 all words
                            up to max_vocab_size are considered;
                            If min_frequency=3, we only consider words that appear
                            at least three times.)
                        input_rep (int): sets the input representation. Values:
                            0 (vanilla), 1 (case-base);
                            <other int>: other models, if you are playful
                        The dictionary can include other keys, if you use them,
                             BUT you shouldn't assume they will be specified by
                             the user, so you should spacify default values.
    Return:
        a dictionary with the at least the following key-value pairs:
                                       {'lstm': torch.nn.Module object,
                                       input_rep: [0|1]}
        #Hint: you may consider adding the embeddings and the vocabulary
        #to the returned dict
    """
    train_corpus = load_annotated_corpus(params_d['data_fn']) #load train data
    train_sentences, train_tags = preprocess_data(train_corpus)
    # Build the vocabulary
    SENTENCES = legacy.data.Field( batch_first=True, pad_first=True, unk_token='<unk>')
    SENTENCES.build_vocab(train_sentences, min_freq=params_d['min_frequency'] if params_d['min_frequency'] != -1 else 1,
                          max_size=params_d['max_vocab_size'] if params_d['max_vocab_size'] != -1 else None)

    vocabulary = SENTENCES.vocab.itos #get all the words of the vocabulary
    vocab_length = len(vocabulary)
    pad_index = SENTENCES.vocab.stoi[SENTENCES.pad_token] #get the index of <pad>
    vectors = load_pretrained_embeddings(params_d['pretrained_embeddings_fn'], vocabulary) #load pretrained embeddings vectors

    if params_d['input_rep'] == 0: #BLSTM Model
        model = BLSTM(embedding_dim=params_d['embedding_dimension'], output_dim=params_d['output_dimension'],
                              num_of_layers=params_d['num_of_layers'], vocabulary_size=vocab_length, pad_idx=pad_index)
    else: # CBLSTM model
        model = CBLSTM(embedding_dim=params_d['embedding_dimension'], output_dim=params_d['output_dimension'],
                      num_of_layers=params_d['num_of_layers'], vocabulary_size=vocab_length, pad_idx=pad_index)

    dict_to_return = {'lstm': model, 'input_rep': params_d['input_rep'], 'field': SENTENCES,'vectors': vectors,
                      'pad_index': pad_index}

    return dict_to_return


def load_pretrained_embeddings(path, vocab=None):
    """ Returns an object with the the pretrained vectors, loaded from the
        file at the specified path. The file format is the same as
        https://www.kaggle.com/danielwillgeorge/glove6b100dtxt
        You can also access the vectors at:
         https://www.dropbox.com/s/qxak38ybjom696y/glove.6B.100d.txt?dl=0
         (for efficiency (time and memory) - load only the vectors you need)
        The format of the vectors object is not specified as it will be used
        internaly in your code, so you can use the datastructure of your choice.

    Args:
        path (str): full path to the embeddings file
        vocab (list): a list of words to have embeddings for. Defaults to None.

    """
    vectors = torchtext.vocab.Vectors(path) #make vectors
    if vocab is None:
        return vectors.vectors
    else:
        return vectors.get_vecs_by_tokens(vocab) #make embedding vectors for the vocabulary

def train_rnn(model, train_data, val_data = None):

    """Trains the BiLSTM model on the specified data.

    Args:
        model (dict): the model dict as returned by initialize_rnn_model()
        train_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus()
        val_data (list): a list of annotated sentences in the format returned
                            by load_annotated_corpus() to be used for validation.
                            Defaults to None
        input_rep (int): sets the input representation. Defaults to 0 (vanilla),
                         1: case-base; <other int>: other models, if you are playful
    """
    input_rep = model['input_rep']
    if input_rep == 0:
        train_sentences , train_tags = preprocess_data(train_data)
        if val_data != None:
            val_sentences, val_tags = preprocess_data(val_data)
            train_dataset, val_dataset = build_dataset(train_sentences, train_tags, model, val_sentences, val_tags)
            val_loader = legacy.data.BucketIterator(val_dataset, batch_size=64, sort_key=lambda x: len(x), shuffle=False)
        else:
            train_dataset = build_dataset(train_sentences, train_tags, model)
            val_loader = None

    else: #input_rep ==1
        train_sentences, train_cases, train_tags = preprocess_data(train_data, input_rep=1)
        if val_data != None:
            val_sentences, val_cases, val_tags = preprocess_data(val_data, input_rep=1)
            train_dataset, val_dataset = build_dataset_with_cases(train_sentences, train_tags, train_cases, model, val_sentences, val_tags, val_cases)
            val_loader = legacy.data.BucketIterator(val_dataset, batch_size=64, sort_key=lambda x: len(x), shuffle=False)
        else:
            train_dataset = build_dataset_with_cases(train_sentences, train_tags, train_cases, model)
            val_loader = None

    train_loader = legacy.data.BucketIterator(train_dataset, batch_size=64, sort_key=lambda x: len(x), shuffle=True,
                                       train=True)
    lstm_model = model['lstm']
    lstm_model.embedding.weight.data.copy_(model['vectors'])
    lstm_model.embedding.weight.data[model['pad_index']] = torch.zeros(100)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
    lstm_model.add_stoi(train_dataset.fields['sentences'].vocab.stoi)
    lstm_model.add_tag_itos(train_dataset.fields['tags'].vocab.itos)
    tag_pad_index = train_dataset.fields['tags'].vocab.stoi['<pad>']


    if input_rep == 0:
        print('Training BLSTM model with 5 epochs')
        for epoch in range(5):
            train_epoch(train_loader, lstm_model, optimizer, epoch + 1, tag_idx=tag_pad_index, val_loader=val_loader)
            print(f'epoch {epoch+1} done')
    else: #input_rep == 1 (CbLSTM)
        print('Training CBLSTM model with 7 epochs')
        for epoch in range(7):
            train_epoch_with_case(train_loader, lstm_model, optimizer, epoch + 1, tag_idx=tag_pad_index, val_loader=val_loader)
            print(f'epoch {epoch+1} done')
    print('Training completed')
    if val_data != None:
        print('Evaluating')
        if input_rep == 0:
            _ = evaluate_model(val_loader, lstm_model, optimizer, tag_pad_index)
        else:
            _ = evaluate_model_with_case(val_loader, lstm_model, optimizer, tag_pad_index)


def rnn_tag_sentence(sentence, model):
    """ Returns a list of pairs (w,t) where each w corresponds to a word
        (same index) in the input sentence and t is the predicted tag.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict):  a dictionary with the trained BiLSTM model and all that is needed
                        to tag a sentence.

    Return:
        list: list of pairs
    """
    sent_lower = [word.lower() for word in sentence]

    input_rep = model['input_rep']
    lstm_model = model['lstm']
    lstm_model.eval()

    dataset = build_test_dataset(sent_lower, lstm_model.stoi, input_rep)
    loader = legacy.data.BucketIterator(dataset, batch_size=1)
    for sent in loader:
        if input_rep == 0:
            predictions = lstm_model(sent.sentences)
            predictions = predictions.view(-1, predictions.shape[-1])
        else:
            predictions = lstm_model(sent.sentences, torch.tensor(sent.cases))
            predictions = predictions.view(-1, predictions.shape[-1])
    labels_idx = predictions.argmax(dim=1, keepdim=True)
    tagged_sentence = make_tags(labels_idx, sentence, lstm_model.tag_itos)
    return tagged_sentence


def get_best_performing_model_params():
    """Returns a disctionary specifying the parameters of your best performing
        BiLSTM model.
        IMPORTANT: this is a *hard coded* dictionary that will be used to create
        a model and train a model by calling
               initialize_rnn_model() and train_lstm()
    """
    best_performing_model_params = {'max_vocab_size': -1,
                                    'min_frequency': 1,
                                    'input_rep': 0,
                                    'embedding_dimension': 100,
                                    'num_of_layers': 2,
                                    'output_dimension': 17,
                                    'pretrained_embeddings_fn': 'glove.6B.100d.txt',
                                    'data_fn': 'en-ud-train.upos.tsv'}

    return best_performing_model_params


#===========================================================
#       Wrapper function (tagging with a specified model)
#===========================================================

def tag_sentence(sentence, model):
    """Returns a list of pairs (w,t) where pair corresponds to a word (same index) in
    the input sentence. Tagging is done with the specified model.

    Args:
        sentence (list): a list of tokens (the sentence to tag)
        model (dict): a dictionary where key is the model name and the value is
           an ordered list of the parameters of the trained model (baseline, HMM)
           or the model isteld and the input_rep flag (LSTMs).

        Models that must be supported (you can add more):
        1. baseline: {'baseline': [perWordTagCounts, allTagCounts]}
        2. HMM: {'hmm': [A,B]}
        3. Vanilla BiLSTM: {'blstm':[model_dict]}
        4. BiLSTM+case: {'cblstm': [model_dict]}
        5. (NOT REQUIRED: you can add other variations, agumenting the input
            with further subword information, with character-level word embedding etc.)

        The parameters for the baseline model are:
        perWordTagCounts (Counter): tags per word as specified in learn_params()
        allTagCounts (Counter): tag counts, as specified in learn_params()

        The parameters for the HMM are:
        A (dict): The HMM Transition probabilities
        B (dict): tthe HMM emmission probabilities.

        Parameters for an LSTM: the model dictionary (allows tagging the given sentence)


    Return:
        list: list of pairs
    """

    if list(model.keys())[0] == 'baseline':
        return baseline_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'hmm':
        return hmm_tag_sentence(sentence, list(model.values())[0][0], list(model.values())[0][1])
    if list(model.keys())[0] == 'blstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])
    if list(model.keys())[0] == 'cblstm':
        return rnn_tag_sentence(sentence, list(model.values())[0][0])

def count_correct(gold_sentence, pred_sentence):
    """Return the total number of correctly predicted tags,the total number of
    correctly predicted tags for oov words and the number of oov words in the
    given sentence.

    Args:
        gold_sentence (list): list of pairs, assume to be gold labels
        pred_sentence (list): list of pairs, tags are predicted by tagger

    """
    assert len(gold_sentence) == len(pred_sentence)

    # Total number of correctly predicted tags and Total number of correctly predicted tags for OOV words:
    correct = 0
    correctOOV = 0
    OOV = 0
    for gold_sent, pred_sent in zip(gold_sentence, pred_sentence):
        if gold_sent[1] == pred_sent[1]:
            if gold_sent[0] not in vocabulary_list:
                correct += 1
                correctOOV += 1
            else:
                correct += 1

    # Total number of OOV words in the given sentence :
    for word in gold_sentence:
        if word[0] not in vocabulary_list:
            OOV += 1

    # print (f'correct: {correct} \t correctOOV: {correctOOV} \t OOV_words: {OOV}')

    return correct, correctOOV, OOV

#===========================================================
#       My functions for Blstm and Cblstm (Helpers)
#===========================================================

def preprocess_data(data, input_rep=0):
    """
    This function split the data to sentences and tags and return lists of each one of them.
    If input_rep = 1 , the function build binary_case for each word in the sentence by using build_3_dim_case function
    Args:
        data: all the sentences with their tags
        input_rep: 0/1 , say if we use BLSTM or CBLSTM model
    Returns:
        list of sentences, list of tags and list of the binary_cases if input_rep = 1.

    """
    sent_list = []
    tags_list = []
    binary_cases = [] #for input_rep == 1 only

    if input_rep == 0:
        for sent in data:
            clean_sentence = []
            sentence_tags = []
            for word_tag in sent:
                clean_sentence.append(word_tag[0].lower())
                sentence_tags.append(word_tag[1])
            sent_list.append(clean_sentence)
            tags_list.append(sentence_tags)

        return sent_list, tags_list

    else: #input_rep == 1
        for sent in data:
            clean_sentence = []
            sentence_tags = []
            sentence_cases = []
            for word_tag in sent:
                clean_sentence.append(word_tag[0].lower())
                sentence_tags.append(word_tag[1])
                bin_case = build_3_dim_case(word_tag[0])
                sentence_cases.append(bin_case)
            sent_list.append(clean_sentence)
            tags_list.append(sentence_tags)
            binary_cases.append(sentence_cases)

        return sent_list, binary_cases, tags_list

def build_3_dim_case(word):
    """
    Args:
      word: word to check and build binary_case
    Return:
        list of 3 dim that say if word is all upper / only first char upper / lower or other
    """
    if word.isupper(): # all the chars are upper case
        return [1,0,0]
    elif word[0].isupper():  # only the first char is upper
        return [0,1,0]
    else: # all lower or other.
        return [0,0,1]


def build_dataset(train_sentences, train_tags, model, val_sentences=None, val_tags=None):
    """
    This function build train and validation data set for train the model for evaluate it,
    Args:
        train_sentences: train data (sentences for train)
        train_tags: tags of the train data
        model: dict - dictionary from initialize_rnn_model
        val_sentences: validation data (sentences for validation)
        val_tags: tags of the validation data
   Returns:
       train and validation datasets.
    """
    SENTENCES = model['field']
    TAGS = legacy.data.Field(batch_first=True, unk_token=None, is_target=True, pad_first=True)
    fields = [('sentences', SENTENCES), ('tags', TAGS)]

    train_examples = []
    for sent, tags in zip(train_sentences, train_tags):
        train_examples.append(legacy.data.Example.fromlist([sent, tags], fields))
    train_dataset = legacy.data.Dataset(train_examples, fields=fields)

    TAGS.build_vocab(train_dataset)
    if val_sentences != None:
        val_examples = []
        for sent, tags in zip(val_sentences, val_tags):
            val_examples.append(legacy.data.Example.fromlist([sent, tags], fields))
        val_dataset = legacy.data.Dataset(val_examples, fields=fields)
        return train_dataset, val_dataset
    else:
        return train_dataset


def build_dataset_with_cases(train_sentences, train_tags, train_cases, model, val_sentences=None, val_tags=None, val_cases=None):

    """
    This function build train and validation data set for train the model for evaluate it,
    Args:
        train_sentences: train data (sentences for train)
        train_tags: tags of the train data
        train_cases: binary_cases for each word in the sentences
        model: dict - dictionary from initialize_rnn_model
        val_sentences: validation data (sentences for validation)
        val_tags: tags of the validation data
   Returns:
       train and validation datasets.
    """
    SENTENCES = model['field']
    TAGS = legacy.data.Field(batch_first=True, unk_token=None, is_target=True, pad_first=True)
    CASES = legacy.data.RawField()
    fields = [('sentences', SENTENCES), ('cases', CASES), ('tags', TAGS)]

    train_examples = []
    for sent, cases, tags in zip(train_sentences, train_cases, train_tags):
        train_examples.append(legacy.data.Example.fromlist([sent, cases, tags], fields))
    train_dataset = legacy.data.Dataset(train_examples, fields=fields)

    TAGS.build_vocab(train_dataset)
    if val_sentences != None:
        val_examples = []
        for sent, cases, tags in zip(val_sentences,  val_cases, val_tags):
            val_examples.append(legacy.data.Example.fromlist([sent, cases, tags], fields))
        val_dataset = legacy.data.Dataset(val_examples, fields=fields)
        return train_dataset, val_dataset
    else:
        return train_dataset


def train_epoch(train_loader, model, optimizer, epoch, tag_idx=0, val_loader=None, verbose=False):
    """
    Train epoch
    Args:
         train_loader: batch for the train data
         model: dict - dictionary from initialize_rnn_model
         optimizer: optimizer instances
         epoch: num of epoch
         tag_idx: index of pad to ignore
         val_loader: validation loader for tuning
         verbose: True for print and False for not.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=tag_idx)
    model = model.to(device)
    criterion = criterion.to(device)
    epoch_losses = 0
    epoch_acc = 0
    model.train()
    for sent_batch, labels_batch in train_loader:
        optimizer.zero_grad()  # clear previous gradients after every batch
        predictions = model(sent_batch)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = labels_batch.view(-1)
        loss = criterion(predictions, tags)  # calculate loss
        loss.backward()  # calculate gradients
        optimizer.step()  # perform updates using calculated gradients
        epoch_losses += loss.item()
        batch_acc = categorical_accuracy(predictions, tags, tag_idx)
        epoch_acc += batch_acc.item()

    if verbose: #for tuning
        print(f'epoch {epoch}\t train loss: {epoch_losses/len(train_loader):.3f}\t train accuracy: {epoch_acc/len(train_loader):.3f}')
        if val_loader != None:
            _ = evaluate_model(val_loader, model, optimizer, tag_idx) #for tuning


def train_epoch_with_case(train_loader, model, optimizer, epoch, tag_idx=0, val_loader=None, verbose=False):
    """
    Train epoch
    Args:
         train_loader: batch for the train data
         model: dict - dictionary from initialize_rnn_model
         optimizer: optimizer instances
         epoch: num of epoch
         tag_idx: index of pad to ignore
         val_loader: validation loader for tuning
         verbose: True for print and False for not.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=tag_idx)
    model = model.to(device)
    criterion = criterion.to(device)
    epoch_losses = 0
    epoch_acc = 0
    model.train()
    for sent_batch, labels_batch in train_loader:
        case_batch = []
        maximum = len(max(sent_batch[1], key=lambda x: len(x)))
        for s in sent_batch[1]:
            x = maximum - len(s)
            case_batch.append(([[0, 0, 0]] * x) + s)
        case_batch = torch.tensor(case_batch)
        optimizer.zero_grad()  # clear previous gradients after every batch
        predictions = model(sent_batch[0], case_batch)
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = labels_batch.view(-1)
        loss = criterion(predictions, tags)  # calculate loss
        loss.backward()  # calculate gradients
        optimizer.step()  # perform updates using calculated gradients
        epoch_losses += loss.item()
        batch_acc = categorical_accuracy(predictions, tags, tag_idx)
        epoch_acc += batch_acc.item()

    if verbose:  # for tuning
        print(
            f'epoch {epoch}\t train loss: {epoch_losses / len(train_loader):.3f}\t train accuracy: {epoch_acc / len(train_loader):.3f}')
        if val_loader != None:
            _ = evaluate_model_with_case(val_loader, model, optimizer, tag_idx)  # for tuning

def evaluate_model(test_loader, model, optimizer, tag_idx):
    """
    Evaluate model
    Args:
         test_loader: test loader
         model: dict - dictionary from initialize_rnn_model
         optimizer: optimizer instances
         tag_idx: index of pad to ignore

    """
    criterion = nn.CrossEntropyLoss(ignore_index=tag_idx)
    model = model.to(device)
    criterion = criterion.to(device)
    test_losses = 0
    test_acc = 0
    model.eval()
    for sent_batch, labels_batch in test_loader:
        with torch.no_grad():
            optimizer.zero_grad()  # clear previous gradients
            predictions = model(sent_batch)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = labels_batch.view(-1)
            loss = criterion(predictions, tags)  # calculate loss
            test_losses += loss.item()
            batch_acc = categorical_accuracy(predictions, tags, tag_idx)
            test_acc += batch_acc.item()

    print(f'test loss: {test_losses/len(test_loader):.3f}\t test accuracy: {test_acc/len(test_loader):.3f}')

    return test_acc/len(test_loader)

def evaluate_model_with_case(test_loader, model, optimizer, tag_idx):
    """
        Evaluate model
        Args:
             test_loader: test loader
             model: dict - dictionary from initialize_rnn_model
             optimizer: optimizer instances
             tag_idx: index of pad to ignore

        """
    criterion = nn.CrossEntropyLoss(ignore_index=tag_idx)
    model = model.to(device)
    criterion = criterion.to(device)
    test_losses = 0
    test_acc = 0
    model.eval()
    for sent_batch, labels_batch in test_loader:
        case_batch = []
        maximum = len(max(sent_batch[1], key=lambda x: len(x)))
        for s in sent_batch[1]:
            x = maximum - len(s)
            case_batch.append(([[0, 0, 0]] * x) + s)
        case_batch = torch.tensor(case_batch)
        with torch.no_grad():
            optimizer.zero_grad()  # clear previous gradients
            predictions = model(sent_batch[0], case_batch)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = labels_batch.view(-1)
            loss = criterion(predictions, tags)  # calculate loss
            test_losses += loss.item()
            batch_acc = categorical_accuracy(predictions, tags, tag_idx)
            test_acc += batch_acc.item()

    print(f'test loss: {test_losses/len(test_loader):.3f}\t test accuracy: {test_acc/len(test_loader):.3f}')

    return test_acc/len(test_loader)

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / y[non_pad_elements].shape[0]

def build_test_dataset(test_sentence, stoi, input_rep):
    """
    Args:
        test_sentence: sentence for test (to predict his tags)
        stoi: dict of all words with their indexes
        input_rep: 0/1  say if the model is BLSTM or CBLSTM

    """
    SENTENCES = legacy.data.Field(use_vocab=False, batch_first=True, pad_first=True, unk_token='<unk>')

    test_example = []
    if input_rep == 0:  #BLSTM
        test_sentence = numericalize(test_sentence, stoi)
        fields = [('sentences', SENTENCES)]
        test_example.append(legacy.data.Example.fromlist([test_sentence], fields))

    else: # CBLSTM
        CASES = legacy.data.RawField()
        fields = [('sentences', SENTENCES), ('cases', CASES)]
        words_cases = []
        for w in test_sentence:
            words_cases.append(build_3_dim_case(w.lower()))
        test_sentence = numericalize(test_sentence, stoi)
        test_example.append(legacy.data.Example.fromlist([test_sentence, words_cases], fields))

    test_dataset = legacy.data.Dataset(test_example, fields=fields)

    return test_dataset

def make_tags(labels_idx, sentence, labels_itos):
    tagged_sentence = []
    for word, tag in zip(sentence, labels_idx):
        tagged_sentence.append((word, labels_itos[tag]))
    return tagged_sentence


def numericalize(sentences, stoi):
    numericlized_sentences = []
    if type(sentences[0]) == list:  # list of sentences
        for sent in sentences:
            num_sent = []
            for word in sent:
                num_sent.append(stoi.get(word, 0))  # if the word is OOV add UNK instead the word
            numericlized_sentences.append(num_sent)
    else:  # one sentence only
        for word in sentences: #in this case sentences is only one sentence
            numericlized_sentences.append(stoi.get(word, 0))  # if the word is OOV add UNK instead the word

    return numericlized_sentences
