from argparse import ArgumentParser
import os

import numpy as np
import tensorflow as tf
from tqdm import trange

import aux_functions
import preprocess 
from models import rnn, ngram_rnn, mlp_2layer


# Parse arguments
parser = ArgumentParser()
parser.add_argument("--output_dir", type=str, required=False, default="./Outputs/Default")
parser.add_argument("--dataset", type=str, required=False, default="agnews")
parser.add_argument("--model", type=str, required=False, default="rkm_lstm")
parser.add_argument("--ngram_length", type=int, required=False, default=1)
parser.add_argument("--mb", type=int, required=False, default=16)
parser.add_argument("--dim", type=int, required=False, default=300)
parser.add_argument("--h_dim", type=int, required=False, default=300)
parser.add_argument("--layer_norm", dest="layer_norm", action="store_true")
parser.add_argument("--dropout", type=float, required=False, default=0.5)
parser.add_argument("--train_iters", type=int, required=False, default=40001)
parser.add_argument("--validation_rate", type=int, required=False, default=500)
parser.add_argument("--log_tfevents", dest="log_tfevents", action="store_true")
args = vars(parser.parse_args())


#######################################
# Data/directory set-up
#######################################

# Import data and load embeddings
dataset = args["dataset"].strip().lower()
if dataset == "agnews":
    data_file = os.path.join("Data","AGnews","ag_news.p")
    embed_file = os.path.join("Data","AGnews","ag_news_glove.p")
    args["NUM_CLASS"] = 4
    args["MAX_LEN"] = 128
elif dataset == "dbpedia":
    data_file = os.path.join("Data","DBpedia","dbpedia.p")
    embed_file = os.path.join("Data","DBpedia","dbpedia_glove.p")
    args["NUM_CLASS"] = 14
    args["MAX_LEN"] = 128
elif dataset == "yahoo":
    data_file = os.path.join("Data","Yahoo","yahoo.p")
    embed_file = os.path.join("Data","Yahoo","yahoo_glove.p")
    args["NUM_CLASS"] = 10
    args["MAX_LEN"] = 96
elif dataset == "yelp":
    data_file = os.path.join("Data","Yelp","yelp.p")
    embed_file = os.path.join("Data","Yelp","yelp_glove.p")
    args["NUM_CLASS"] = 2
    args["MAX_LEN"] = 128
elif dataset == "yelp_full":
    data_file = os.path.join("Data","Yelp_full","yelp_full.p")
    embed_file = os.path.join("Data","Yelp_full","yelp_full_glove.p")
    args["NUM_CLASS"] = 5
    args["MAX_LEN"] = 128    
else:
    raise NotImplementedError("Dataset {0} not implemented! \nOptions: 'agnews', 'dbpedia', 'yahoo', 'yelp', 'yelp_full'".format(dataset))

data, labels, dicts = preprocess.load_data(data_file)
if dataset == "yelp":
    labels = preprocess.convert_labels_to_one_hot(labels)
N_train = len(data["train"])
_W_emb = preprocess.load_embeddings(embed_file)
W_EMB_DIMS = _W_emb.shape
args["m"] = W_EMB_DIMS[1]

# Output directory                                                                                                        
output_dir = args["output_dir"]
os.makedirs(output_dir, exist_ok=True)
results_filename = os.path.join(output_dir, "accs.txt")
logs_filename = os.path.join(output_dir, "logs.txt")
aux_functions.log_model(logs_filename, args)


#######################################
# Build model
#######################################

# Model Inputs
x = tf.placeholder(tf.int32, [args["mb"], args["MAX_LEN"]], name="x")
x_mask = tf.placeholder(tf.float32, [args["mb"], args["MAX_LEN"]], name="x_mask")
y = tf.placeholder(tf.float32, [args["mb"], args["NUM_CLASS"]], name='y')

# Word embeddings placeholder
W_emb = tf.placeholder(tf.float32, W_EMB_DIMS)

# Lookup word embeddings
x_emb, W, W_init_op = aux_functions.embedding(x, W_emb)

## Network
if args["ngram_length"] == 1:
    h = rnn(x_emb, x_mask, args)
elif args["ngram_length"] > 1:
    h = ngram_rnn(x_emb, x_mask, args)
else:
    raise ValueError("ngram_length of {0} is invalid".format(args["ngram_length"]))
h = tf.squeeze(h)
logits = mlp_2layer(h, args["h_dim"], args["dropout"], prefix='classify_', num_outputs=args["NUM_CLASS"])

# Loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

# Optimizer (with gradient clipping)
optimizer = tf.train.AdamOptimizer(1e-3)
gradients = optimizer.compute_gradients(cross_entropy)
clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
train_step = optimizer.apply_gradients(clipped_gradients)

# Accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Summaries
if args["log_tfevents"]:
    tf.summary.scalar("cross_entropy", cross_entropy)
    merged_sum = tf.summary.merge_all()

# Variable Initializer
init_op = tf.global_variables_initializer()

# Allow GPU memory allocation growth (instead of defaulting to everything)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


#######################################
# Train
#######################################
max_val_accuracy = 0.
max_test_accuracy = 0.

with tf.Session() as sess:
    # Initialize learnable weights, embeddings
    sess.run([init_op, W_init_op], feed_dict={W_emb: _W_emb})
    
    # Summary writer
    if args["log_tfevents"]:
        writer = tf.summary.FileWriter(os.path.join(output_dir,"TF_logs"), sess.graph)

    for i in trange(args["train_iters"]):
        # Sample minibatch
        idx = np.random.choice(N_train, args["mb"], replace=False)
        _x, _x_mask = preprocess.prepare_data(idx, data["train"], args["MAX_LEN"])
        _y = preprocess.prepare_labels(idx, labels["train"])

        # Run one train step
        feed_dict = {x: _x, x_mask: _x_mask, y: _y}
        if args["log_tfevents"]:
            _, _sum = sess.run([train_step, merged_sum], feed_dict=feed_dict)
            writer.add_summary(_sum, i)
        else:
            sess.run(train_step, feed_dict=feed_dict)        

        # Validate
        if i % args["validation_rate"] == 0:
            val_accuracy = aux_functions.evaluate_accuracy(
                                sess, 
                                accuracy, 
                                x, 
                                x_mask, 
                                y, 
                                data["valid"], 
                                labels["valid"], 
                                args["mb"], 
                                args["MAX_LEN"]
                            )
            aux_functions.write_acc_to_file(results_filename, val_accuracy)
            
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            
            # Evaluate on test set
            test_accuracy = aux_functions.evaluate_accuracy(sess, accuracy, x, x_mask, y, data["test"], labels["test"], args["mb"], args["MAX_LEN"])
            print("Test accuracy at iter {0}: {1}".format(i, test_accuracy))
            
            max_test_accuracy = test_accuracy
            
    print("Max Test accuracy {0}".format(max_test_accuracy))
