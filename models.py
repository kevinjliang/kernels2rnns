import tensorflow as tf
from tensorflow.contrib import layers
from kernel_cells import LSTMCell_mod


def rnn(x_emb, x_mask, args):
    # Mask input words (sets tokens after last word to zero)
    x_emb = x_emb * tf.expand_dims(x_mask, axis=-1) # x_emb \in [mb, L, m]

    # Various parameters
    L = tf.cast(tf.reduce_sum(x_mask, axis=1), tf.int32)
    dim = args["dim"]

    with tf.variable_scope("rnn1"):
        cell = LSTMCell_mod(
                    dim, 
                    gate_mod=args["model"], 
                    layer_norm=args["layer_norm"],
                )
        
        hs, _ = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=x_emb,
                    sequence_length=L,
                    dtype=tf.float32
                )
    
        # Global pool
        h1 = tf.reduce_mean(hs, axis=1)
        
    return h1


def ngram_rnn(x_emb, x_mask, args):
    # Mask input words (sets tokens after last word to zero)
    x_emb = x_emb * tf.expand_dims(x_mask, axis=-1) # x_emb \in [mb, L, m]
    
    # Various parameters
    L = tf.cast(tf.reduce_sum(x_mask, axis=1), tf.int32)
    dim = args["dim"]

    with tf.variable_scope("rnn1"):
        x_ngram = tf.layers.conv1d(x_emb, 4*dim, args["ngram_length"], padding="same", use_bias=False, activation=None)
        
        cell = LSTMCell_mod(
                    dim, 
                    gate_mod=args["model"], 
                    ngram=True,
                    layer_norm=args["layer_norm"],
                )
        
        hs, _ = tf.nn.dynamic_rnn(cell=cell,
                                  inputs=x_ngram,
                                  sequence_length=L,
                                  dtype=tf.float32)
    
        # Global pool
        h = tf.reduce_mean(hs, axis=1)
        
    return h


def mlp_2layer(h, h_dim, dropout, prefix='', num_outputs=1):
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    
    h = layers.fully_connected(
                tf.nn.dropout(h, keep_prob=dropout), 
                num_outputs=h_dim,
                biases_initializer=biasInit, 
                activation_fn=tf.nn.relu, 
                scope=prefix + 'mlp_1'
            )
    logits = layers.linear(
                tf.nn.dropout(h, keep_prob=dropout), 
                num_outputs=num_outputs,
                biases_initializer=biasInit, 
                scope=prefix + 'mlp_2'
            )
    
    return logits