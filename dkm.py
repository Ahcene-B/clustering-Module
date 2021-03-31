#!/usr/bin/env python3

__author__ = "Thibaut Thonet, Maziar Moradi Fard"
__license__ = "GPL"

import os
import random
import math
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="Deep k-means algorithm")
parser.add_argument("-d", "--dataset", type=str.upper,
                    help="Dataset on which DKM will be run (one of USPS, MNIST, 20NEWS, RCV1)", required=True)
parser.add_argument("-v", "--validation", help="Split data into validation and test sets", action='store_true')
parser.add_argument("-p", "--pretrain", help="Pretrain the autoencoder and cluster representatives",
                    action='store_true')
parser.add_argument("-a", "--annealing",
                    help="Use an annealing scheme for the values of alpha (otherwise a constant is used)",
                    action='store_true')
parser.add_argument("-s", "--seeded", help="Use a fixed seed, different for each run", action='store_true')
parser.add_argument("-c", "--cpu", help="Force the program to run on CPU", action='store_true')
parser.add_argument("-l", "--lambda", type=float, default=-1, dest="lambda_",
                    help="Value of the hyperparameter weighing the clustering loss against the reconstruction loss")
parser.add_argument("-e", "--p_epochs", type=int, default=50, help="Number of pretraining epochs")
parser.add_argument("-f", "--f_epochs", type=int, default=5, help="Number of fine-tuning epochs per alpha value")
parser.add_argument("-b", "--batch_size", type=int, default=256, help="Size of the minibatches used by the optimizer")
parser.add_argument("-g", "--gpu", type=int, help="Which GPU to use")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from Module.dkm_utils import cluster_acc
from Module.dkm_utils import next_batch
from Module.compgraph import DkmCompGraph
import tensorflow as tf

tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(35)

tf.keras.backend.clear_session()
tf.set_random_seed(0)
random.seed(0)
np.random.seed(0)



# Dataset setting from arguments
#if args.dataset == "USPS":
#    import usps_specs as specs
#elif args.dataset == "MNIST":
#    import mnist_specs as specs
#elif args.dataset == "20NEWS":
#    import _20news_specs as specs
#elif args.dataset == "RCV1":
#    import rcv1_specs as specs
from Tuned_Param import *

#dataset = fetch_mldata("USPS")
LOAD = np.load('data/'+args.dataset.upper()+'.npz',allow_pickle=True)
data = LOAD['x']
target = LOAD['y']
del LOAD
print("Dataset",args.dataset.upper(),"loaded...")

n_samples = data.shape[0] # Number of samples in the dataset
n_clusters = len(set(target)) # Number of clusters to obtain

# Get the split between training/test set and validation set
test_indices = np.arange(n_samples) #read_list("split/"+args.dataset.lower()+"/test")
validation_indices = np.arange(n_samples)  #read_list("split/"+args.dataset.lower()+"/validation")

# Auto-encoder architecture
input_size = data.shape[1]
hidden_1_size = 500
hidden_2_size = 500
hidden_3_size = 2000
embedding_size = int(AECM_UNIF[ args.dataset.upper() ]['OUT']) # n_clusters # 
dimensions = [hidden_1_size, hidden_2_size, hidden_3_size, embedding_size, # Encoder layer dimensions
              hidden_3_size, hidden_2_size, hidden_1_size, input_size] # Decoder layer dimensions
activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu, None, # Encoder layer activations
               tf.nn.relu, tf.nn.relu, tf.nn.relu, None] # Decoder layer activations
names = ['enc_hidden_1', 'enc_hidden_2', 'enc_hidden_3', 'embedding', # Encoder layer names
         'dec_hidden_1', 'dec_hidden_2', 'dec_hidden_3', 'output'] # Decoder layer names


#else:
#    parser.error("Unknown dataset!")
#    exit()

# Parameter setting from arguments
n_pretrain_epochs = args.p_epochs
n_finetuning_epochs = args.f_epochs

#DIC_LBD = {
#    'MNIST':    (1e-2, 1e0),
#    'FMNIST':   (1e-2, 1e-2),
#    'USPS':     (1e-2, 1e-1),
#    'CIFAR10':  (1e-2, 1e-1),
#    'R10K':     (1e-2, 1e0),
#    '20NEWS':   (1e-4, 1e-2),
#    '10X73K':   (1e-4, 1e-4),
#    'PENDIGIT': (0, 0),

DIC_LBD = {
    'MNIST':    (1e-1, 1e-0),
    'FMNIST':   (1e-1, 1e-0),
    'USPS':     (1e-1, 1e-0),
    'CIFAR10':  (1e-1, 1e-0),
    'R10K':     (1e-4, 1e-2),
    '20NEWS':   (1e-4, 1e-1),
    '10X73K':   (1e-4, 1e-2),
    'PENDIGIT': (1e-4, 1e-2),
}

batch_size = args.batch_size # Size of the mini-batches used in the stochastic optimizer
n_batches = int(math.ceil(n_samples / batch_size)) # Number of mini-batches
validation = args.validation # Specify if data should be split into validation and test sets
pretrain = args.pretrain # Specify if DKM's autoencoder should be pretrained
annealing = args.annealing # Specify if annealing should be used
seeded = args.seeded # Specify if runs are seeded

if args.lambda_ == -1:
    lambda_ = DIC_LBD[args.dataset.upper()][ 1 if pretrain else 0 ] # args.lambda_ # MNIST .01
else:
    lambda_ = args.lambda_ # MNIST .01

print("Hyperparameters...")
print("lambda =", lambda_)

if pretrain:
    AE = np.load(args.dataset.upper()+'/save/save-ae.npz',allow_pickle=True)['wgt']

FNAME = args.dataset.upper()+'/save/save-dkm-'+('rand' if annealing else ( 'pre' if pretrain else 'unif') )+'-lbd='+str(lambda_)
print(FNAME)

# Define the alpha scheme depending on if the approach includes annealing/pretraining
if annealing and not pretrain:
    constant_value = 1  # embedding_size # Used to modify the range of the alpha scheme
    max_n = 200  # Number of alpha values to consider
    alphas = np.zeros(max_n, dtype=float)
    alphas[0] = 0.1
    for i in range(1, max_n):
        alphas[i] = (2 ** (1 / (np.log(i + 1)) ** 2)) * alphas[i - 1]
    alphas = alphas / constant_value
elif not annealing and pretrain:
    constant_value = 1  # embedding_size # Used to modify the range of the alpha scheme
    max_n = 200  # Number of alpha values to consider (constant values are used here)
    alphas = 1000*np.ones(max_n, dtype=float) # alpha is constant
    alphas = alphas / constant_value
else:
    constant_value = 1  # embedding_size # Used to modify the range of the alpha scheme
    max_n = 200  # Number of alpha values to consider (constant values are used here)
    alphas = 1000*np.ones(max_n, dtype=float) # alpha is constant
    alphas = alphas / constant_value
    
#    parser.error("Run with either annealing (-a) or pretraining (-p), but not both.")
#    exit()

# Select only the labels which are to be used in the evaluation (disjoint for validation and test)
if validation:
    validation_target = np.asarray([target[i] for i in validation_indices])
    test_target = np.asarray([target[i] for i in test_indices])
else:
    target = target

# Dataset on which the computation graph will be run
data = data

# Hardware specifications
if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Run on CPU instead of GPU if batch_size is small
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
config = tf.ConfigProto(gpu_options=gpu_options)

# Definition of the randomly-drawn (0-10000) seeds to be used for each run
seeds = [8905, 9129, 291, 4012, 1256, 6819, 4678, 6971, 1362, 575]

if validation:
    list_validation_acc = []
    list_validation_ari = []
    list_validation_nmi = []
    list_test_acc = []
    list_test_ari = []
    list_test_nmi = []
else:
    list_acc = []
    list_ari = []
    list_nmi = []
    list_mu = []
    list_gmu = []
    
    list_km_acc = []
    list_km_ari = []
    list_km_nmi = []

n_runs = 20
for run in range(n_runs):
    # Use a fixed seed for this run, as defined in the seed list
    if seeded:
        tf.reset_default_graph()
        tf.set_random_seed(seeds[run])
        np.random.seed(seeds[run])

    print("\n>> Run", run+1)
    
    
    tf.reset_default_graph()
        

    # Define the computation graph for DKM
    cg = DkmCompGraph([dimensions, activations, names], n_clusters, lambda_, (AE[run] if pretrain else None) )

    # Run the computation graph
    with tf.Session(config=config) as sess:
        # Initialization
        init = tf.global_variables_initializer()
        sess.run(init)

        # Variables to save tensor content
        distances = np.zeros((n_clusters, n_samples))

        # Pretrain if specified
        if pretrain:
            print("Starting autoencoder pretraining...")
            """
            # Variables to save pretraining tensor content
            embeddings = np.zeros((n_samples, embedding_size), dtype=float)

            # First, pretrain the autoencoder
            ## Loop over epochs
            for epoch in range(n_pretrain_epochs):
                print("Pretraining step: epoch {}".format(epoch))

                # Loop over the samples
                for _ in range(n_batches):
                    # Fetch a random data batch of the specified size
                    indices, data_batch = next_batch(batch_size, data)

                    # Run the computation graph until pretrain_op (only on autoencoder) on the data batch
                    _, embedding_, ae_loss_ = sess.run((cg.pretrain_op, cg.embedding, cg.ae_loss),
                                                       feed_dict={cg.input: data_batch})

                    # Save the embeddings for batch samples
                    for j in range(len(indices)):
                        embeddings[indices[j], :] = embedding_[j, :]

                    #print("ae_loss_:", float(ae_loss_))
            """
            # Second, run k-means++ on the pretrained embeddings
            print("Running k-means on the learned embeddings...")
            embeddings = sess.run((cg.embedding,), feed_dict={cg.input: data})[0].squeeze()
            kmeans_model = KMeans(n_clusters=n_clusters, init="k-means++").fit(embeddings)

            if validation:
                # Split the cluster assignments into validation and test ones
                validation_cluster_assign = np.asarray([kmeans_model.labels_[i] for i in validation_indices])
                test_cluster_assign = np.asarray([kmeans_model.labels_[i] for i in test_indices])

                # Evaluate the clustering validation performance using the ground-truth labels
                validation_acc = cluster_acc(validation_target, validation_cluster_assign)
                print("Validation ACC", validation_acc)
                validation_ari = adjusted_rand_score(validation_target, validation_cluster_assign)
                print("Validation ARI", validation_ari)
                validation_nmi = normalized_mutual_info_score(validation_target, validation_cluster_assign)
                print("Validation NMI", validation_nmi)

                # Evaluate the clustering test performance using the ground-truth labels
                test_acc = cluster_acc(test_target, test_cluster_assign)
                print("Test ACC", test_acc)
                test_ari = adjusted_rand_score(test_target, test_cluster_assign)
                print("Test ARI", test_ari)
                test_nmi = normalized_mutual_info_score(test_target, test_cluster_assign)
                print("Test NMI", test_nmi)  
            else:
                # Evaluate the clustering performance using the ground-truth labels
                acc = cluster_acc(target, kmeans_model.labels_)
                print("ACC", acc)
                ari = adjusted_rand_score(target, kmeans_model.labels_)
                print("ARI", ari)
                nmi = normalized_mutual_info_score(target, kmeans_model.labels_)
                print("NMI", nmi)

            # The cluster centers are used to initialize the cluster representatives in DKM
            sess.run(tf.assign(cg.cluster_rep, kmeans_model.cluster_centers_))

        # Train the full DKM model
        if (len(alphas) > 0):
            print("Starting DKM training", args.dataset, ('Pretrain' if pretrain else ( 'Annealing' if  annealing else 'Uniform')))
            
        print_val = 10
        ## Loop over alpha (inverse temperature), from small to large values
        o_cluster_assign = np.zeros(n_samples)
        o_loss_ = np.inf
        for k in range(len(alphas)):

            # Loop over epochs per alpha
            for _ in range(n_finetuning_epochs):
                # Loop over the samples
                for _ in range(n_batches):
                    #print("Training step: alpha[{}], epoch {}".format(k, i))

                    # Fetch a random data batch of the specified size
                    indices, data_batch = next_batch(batch_size, data)

                    #print(tf.trainable_variables())
                    #current_batch_size = np.shape(data_batch)[0] # Can be different from batch_size for unequal splits

                    # Run the computation graph on the data batch
                    _, loss_, stack_dist_, cluster_rep_, ae_loss_, kmeans_loss_ =\
                        sess.run((cg.train_op, cg.loss, cg.stack_dist, cg.cluster_rep, cg.ae_loss, cg.kmeans_loss),
                                 feed_dict={cg.input: data_batch, cg.alpha: alphas[k]})

                    if k % print_val == 0 or k == max_n - 1:
                        # Save the distances for batch samples
                        for j in range(len(indices)):
                            distances[:, indices[j]] = stack_dist_[:, j]

            # Evaluate the clustering performance every print_val alpha and for last alpha
            if k == 0 or (k+1) % print_val in [0] or k == max_n - 1:
                print("Training step: alpha[{}]: {}".format(k+1, alphas[k]), "lambda =", lambda_)

                LINE = ''
                LINE += "loss: {:.4f}".format(loss_)
                LINE += "\tae loss: {:.4f}".format(ae_loss_)
                LINE += "\tkmeans loss: {:.4f}".format(kmeans_loss_)

                d_loss = np.abs( o_loss_ - loss_)
                LINE += "\t({:.1f})".format( np.log10(d_loss) )
                o_loss_ = loss_

                # Infer cluster assignments for all samples
                cluster_assign = np.zeros((n_samples), dtype=float)
                for i in range(n_samples):
                    index_closest_cluster = np.argmin(distances[:, i])
                    cluster_assign[i] = index_closest_cluster
                cluster_assign = cluster_assign.astype(np.int64)

                if validation:
                    validation_cluster_assign = np.asarray([cluster_assign[i] for i in validation_indices])
                    test_cluster_assign = np.asarray([cluster_assign[i] for i in test_indices])

                    # Evaluate the clustering validation performance using the ground-truth labels
                    validation_acc = cluster_acc(validation_target, validation_cluster_assign)
                    print("Validation ACC", validation_acc)
                    validation_ari = adjusted_rand_score(validation_target, validation_cluster_assign)
                    print("Validation ARI", validation_ari)
                    validation_nmi = normalized_mutual_info_score(validation_target, validation_cluster_assign)
                    print("Validation NMI", validation_nmi)

                    # Evaluate the clustering test performance using the ground-truth labels
                    test_acc = cluster_acc(test_target, test_cluster_assign)
                    print("Test ACC", test_acc)
                    test_ari = adjusted_rand_score(test_target, test_cluster_assign)
                    print("Test ARI", test_ari)
                    test_nmi = normalized_mutual_info_score(test_target, test_cluster_assign)
                    print("Test NMI", test_nmi)
                else:
                    # Evaluate the clustering performance using the ground-truth labels
                    ari = adjusted_rand_score(target, cluster_assign)
                    LINE += "\tARI {:.1f}".format(ari*100)
                    nmi = normalized_mutual_info_score(target, cluster_assign)
                    LINE += ", NMI {:.1f}".format(nmi*100)
                    acc = cluster_acc(target, cluster_assign)
                    LINE += ", ACC {:.1f}".format(acc*100)
                    
                d_ari = adjusted_rand_score( o_cluster_assign, cluster_assign)
                LINE += "\t({:.3f})".format( d_ari )
                print(LINE)
                o_cluster_assign = cluster_assign[:]
                
#                if d_ari > .99:
#                    break
                if np.log10(d_loss) < -4:
                    print("** CV loss **")
                    break
                    

        print("Running k-means on the learned embeddings...")
        embeddings = sess.run((cg.embedding,), feed_dict={cg.input: data})[0].squeeze()
        kmeans_model = KMeans(n_clusters=n_clusters, init="k-means++").fit(embeddings) 
        km_acc = cluster_acc(target, kmeans_model.labels_)
        print("ACC", km_acc)
        km_ari = adjusted_rand_score(target, kmeans_model.labels_)
        print("ARI", km_ari)
        km_nmi = normalized_mutual_info_score(target, kmeans_model.labels_)
        print("NMI", km_nmi)       

        # Record the clustering performance for the run
        if validation:
            list_validation_acc.append(validation_acc)
            list_validation_ari.append(validation_ari)
            list_validation_nmi.append(validation_nmi)
            list_test_acc.append(test_acc)
            list_test_ari.append(test_ari)
            list_test_nmi.append(test_nmi)
        else:
            list_acc.append(acc)
            list_ari.append(ari)
            list_nmi.append(nmi)
            list_km_acc.append(km_acc)
            list_km_ari.append(km_ari)
            list_km_nmi.append(km_nmi)
            
        if args.dataset == 'MNIST':
            cluster_rep_ = sess.run((cg.cluster_rep, ), feed_dict={})[0]
            list_mu.append( cluster_rep_ )
            
            cluster_rep_out = sess.run((cg.output, ), feed_dict={cg.embedding: cluster_rep_})
            list_gmu.append( cluster_rep_out )
        
        np.savez(FNAME ,
            ari=list_ari,
            nmi=list_nmi,
            acc=list_acc,
            mu=list_mu,
            gmu=list_gmu,
            kari=list_km_ari,
            knmi=list_km_nmi,
            kacc=list_km_acc,
            )

if validation:
    list_validation_acc = np.array(list_validation_acc)
    print("Average validation ACC: {:.3f} +/- {:.3f}".format(np.mean(list_validation_acc), np.std(list_validation_acc)))
    list_validation_ari = np.array(list_validation_ari)
    print("Average validation ARI: {:.3f} +/- {:.3f}".format(np.mean(list_validation_ari), np.std(list_validation_ari)))
    list_validation_nmi = np.array(list_validation_nmi)
    print("Average validation NMI: {:.3f} +/- {:.3f}".format(np.mean(list_validation_nmi), np.std(list_validation_nmi)))

    list_test_acc = np.array(list_test_acc)
    print("Average test ACC: {:.3f} +/- {:.3f}".format(np.mean(list_test_acc), np.std(list_test_acc)))
    list_test_ari = np.array(list_test_ari)
    print("Average test ARI: {:.3f} +/- {:.3f}".format(np.mean(list_test_ari), np.std(list_test_ari)))
    list_test_nmi = np.array(list_test_nmi)
    print("Average test NMI: {:.3f} +/- {:.3f}".format(np.mean(list_test_nmi), np.std(list_test_nmi)))
else:
    list_acc = np.array(list_acc)
    print("Average ACC: {:.3f} +/- {:.3f} ({:.3f})".format(np.mean(list_acc), np.std(list_acc), np.max(list_acc)))
    list_ari = np.array(list_ari)
    print("Average ARI: {:.3f} +/- {:.3f} ({:.3f})".format(np.mean(list_ari), np.std(list_ari), np.max(list_ari)))
    list_nmi = np.array(list_nmi)
    print("Average NMI: {:.3f} +/- {:.3f} ({:.3f})".format(np.mean(list_nmi), np.std(list_nmi), np.max(list_nmi)))
    
np.savez(FNAME ,
    ari=list_ari,
    nmi=list_nmi,
    acc=list_acc,
    mu=list_mu,
    gmu=list_gmu,
    kari=list_km_ari,
    knmi=list_km_nmi,
    kacc=list_km_acc,
    )
