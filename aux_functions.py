import numpy as np
import tensorflow as tf
from tqdm import trange

import preprocess


def embedding(features, _W_emb, is_reuse=None):
    # Creating word embeddings variable
    with tf.variable_scope("embed", reuse=is_reuse):
        W = tf.get_variable("W", shape=_W_emb.shape, dtype=tf.float32, initializer=None, trainable=True)
    
    # Initialize word embeddings with pretrained embeddings
    W_init_op = W.assign(_W_emb)

    # Look up word embeddings for selected words
    word_vectors = tf.nn.embedding_lookup(W, features)

    return word_vectors, W, W_init_op

def evaluate_accuracy(sess, acc, x, x_mask, y, eval_data, eval_labels, mb, maxlen):
    """
    Accuracy evaluation 
    """
    N_samples = len(eval_data)
    acc_mb = np.zeros(N_samples // mb)

    for j in trange(len(acc_mb)):
        # Sample data
        idx = np.arange(j*mb, (j+1)*mb)
        _x, _x_mask = preprocess.prepare_data(idx, eval_data, maxlen)
        _y = preprocess.prepare_labels(idx, eval_labels)

        # Test
        feed_dict = {x: _x, x_mask: _x_mask, y: _y}
        acc_mb[j] = sess.run(acc, feed_dict=feed_dict)

    acc_avg = np.mean(acc_mb)
    return acc_avg

def write_acc_to_file(filename, acc):
    with open(filename, "a") as f:
        f.write("{:.4} \n".format(acc))

def write_closest_words_to_anchors_to_file(filename, anchors, _W_emb, idx2word, n=10):
    # Find distances between embeddings and anchors
    anchors = np.expand_dims(anchors, axis=0) # anchors \in [1, m, J]
    _W_emb = np.expand_dims(_W_emb, axis=-1)  # _W_emb \in [V, m, 1]
    dist_sq = np.sum(np.square(anchors - _W_emb), axis=1) # dist_sq \in [V, J]
    dist_sq_rank = np.argsort(dist_sq, axis=0)
    
    with open(filename, "a") as f:
        for j, dist_sq_rank_j in enumerate(dist_sq_rank.T):
            f.write("{}:".format(j))
            for i in range(n):
                idx = dist_sq_rank_j[i]
                f.write("\t{0} {1:.4},".format(idx2word[idx], dist_sq[idx, j]))
            f.write("\n")
        
def log_model(filename, args):
    with open(filename, "w") as f:
        for key, val in args.items():
            f.write(key + ": \t\t" + str(val))
            f.write("\n")