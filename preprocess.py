import numpy as np
import _pickle as pickle


def load_data(data_file):
    # Load pickled data
    print("Loading data")
    x = pickle.load(open(data_file, "rb"), encoding="latin1")
    data = {"train": x[0], 
            "valid": x[1], 
            "test": x[2]}
    labels = {"train": np.array(x[3], dtype="float32"), 
              "valid": np.array(x[4], dtype="float32"), 
              "test": np.array(x[5], dtype="float32")}
    dicts = {"word2idx": x[6], 
             "idx2word": x[7]}
    print("Data loaded")
    
    return data, labels, dicts

def load_embeddings(embed_file):
    # Load pickled embeddings
    print("Loading embeddings")
    _W_emb = np.array(pickle.load(open(embed_file, "rb"), encoding="latin1"), dtype="float32")
    print("Embeddings loaded")
    
    return _W_emb

def convert_labels_to_one_hot(label_dict):
    for split, labels in label_dict.items():
        labels = np.array(labels, dtype=np.int32)
        one_hot = np.zeros((len(labels), np.max(labels)+1))
        one_hot[np.arange(len(labels)), labels] = 1
        label_dict[split] = one_hot
    return label_dict

def prepare_data(idx, data_split, maxlen):
    # Extract sentences
    sents = [data_split[i] for i in idx]
    _x, _x_mask = truncate_and_mask_sentences(sents, maxlen)
    return _x, _x_mask

def prepare_labels(idx, labels_split):
    # Extract labels
    labels = np.array([labels_split[i] for i in idx])
    _y = labels.reshape((len(labels), -1))
    return _y

def truncate_and_mask_sentences(seqs_x, maxlen):
    lengths_x = [len(s) for s in seqs_x]
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
            else:
                new_seqs_x.append(s_x[:maxlen])
                new_lengths_x.append(maxlen)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    n_samples = len(seqs_x)
    x = np.zeros((n_samples, maxlen), "int32")
    x_mask = np.zeros((n_samples, maxlen), "float32")
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]] = 1. # change to remove the real END token
    return x, x_mask