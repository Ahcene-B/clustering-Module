import os,sys,time,copy

import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(35)

tf.keras.backend.clear_session()
tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)

from Module.utils import *

############################################################################

class CM(tf.keras.Model):
    def __init__(self, input_dim, n_clusters, true_labels=None, temperature=1., tracker=''):
        super(CM, self).__init__()

        tf.keras.backend.clear_session()

        self.input_dim = input_dim
        self.n_clusters = n_clusters
        self.true_labels = true_labels
        
        self.gamma = tf.keras.Sequential( [
                tf.keras.layers.InputLayer(
                    input_shape=(self.input_dim,)
                ),
                tf.keras.layers.Dense(
                    units=self.n_clusters, 
                    activation='softmax', 
                    kernel_initializer=tf.initializers.glorot_uniform(), 
                    bias_initializer=tf.initializers.glorot_normal(), 
                    name='imu'
                ),
            ], name="cm_encoder" )
        self.mu = tf.keras.Sequential( [
                tf.keras.layers.InputLayer(
                    input_shape=(self.n_clusters,)
                ),
                tf.keras.layers.Dense(
                    units=self.input_dim, 
                    activation=None, 
                    kernel_initializer=tf.initializers.glorot_uniform(), 
                    bias_initializer=tf.initializers.glorot_normal(), 
                    name='mu' 
                ),
            ], name="cm_d" )
        

    def reset(self):
        self.set_weights( [  tf.initializers.glorot_uniform()(( w.shape )).numpy() for w in self.weights])

    @tf.function
    def call(self, x, training=False):
        g  = self.gamma(x, training)
        tx = self.mu(g, training)
        u  = self.mu( tf.eye( self.n_clusters ), False)
        return g, tx, u
            
    @tf.function
    def loss(self, x, alpha, training=False):
        g  = self.gamma(x, training)
        tx = self.mu(g, training)
        u  = self.mu( np.eye(self.n_clusters).astype('float32'), False)

        loss_op1 = tf.reduce_sum( tf.math.squared_difference(x, tx) )
        loss_op2 = tf.reduce_sum( tf.reduce_sum( g*(1-g),0 ) * tf.reduce_sum(tf.square(u),1) )
        
        gg = tf.linalg.matmul( g,g, transpose_a=True )
        uu = tf.linalg.matmul( u,u, transpose_b=True )
        gu = gg * uu
        gu = gu - tf.linalg.diag(tf.linalg.diag_part(gu))
        loss_op3 = -tf.reduce_sum( gu )

        lmg = tf.math.log( tf.reduce_mean(g,0) +1e-10 )
        slmg = tf.sort(lmg)
        loss_op4 = tf.reduce_sum( slmg * (1-alpha) )

        return tf.stack( (loss_op1 , loss_op2 , loss_op3 , loss_op4) )

    @tf.function
    def update(self, batch, alpha, optimizer):
        with tf.GradientTape() as tape:
            batched_loss = self.loss(batch, alpha, True)
            batched_loss_summed = tf.reduce_sum(batched_loss)
        gradients = tape.gradient( batched_loss_summed, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return batched_loss

    @tf.function
    def tf_predict(self, x):
        z = self.encode(x,False)
        g = self.gamma(z,False)
        p = tf.argmax(g,1)
        return p
        
    def predict(self,x, y_true=None):
        y = tf.argmax(self.gamma(x),1).numpy()
        if y_true is None:
            return y
        else:
            return ari(y_true, y), nmi(y_true, y), acc(y_true, y)
             
        
    def cluster_centers_(self):
        return self.mu( tf.eye( self.n_clusters ), False).numpy()
        
    
    def pre_fit(self, x, 
            y=None, 
            verbose=True
        ):
        if verbose:
            print(">>> Start pre-training with k-means++")
        # Run k-means++ on Input
        kmeans = KMeans(n_clusters=self.n_clusters,init='k-means++',n_init=1,max_iter=1)
        kmeans.fit( x )
        
        if verbose and y is not None:
            p = kmeans.predict( x )
            print_line = "K-means++\t"
            for name, algo in zip(['ARI','NMI','ACC'], [ari,nmi,acc]):
                print_line += name+" = {:.1f} ".format( algo(y,p)*100 )
            print( print_line, flush=True)
        
        u = kmeans.cluster_centers_
        iu = np.linalg.pinv(u)
        
        # Set the Model's Weights
        self.set_weights( [iu, tf.zeros_initializer()((self.n_clusters,)).numpy(), 
                            u, tf.zeros_initializer()((self.input_dim,)).numpy()] )

        if verbose and y is not None:
            n,d = x.shape
            loss = self.loss(x, 0, False).numpy()
            p = self.predict( x )
            print_line = "loss = {:.3f} {:.3f} {:.3f} {:.3f} = {:.3f}\t".format(loss[0]/n, loss[1]/d, loss[2]/d/d, loss[3], loss.sum() )
            for name, algo in zip(['ARI','NMI','ACC'], [ari,nmi,acc]):
                print_line += name+" = {:.1f} ".format( algo(y,p)*100 )
            print( print_line, flush=True)
            
        if verbose:
            print(">>> Done.")
    
    def fit(self, x, alpha,
        y=None, 
        update_p=20, 
        batch_size=256, 
        epoch_size=150, 
        optimizer_name=None, 
        optimizer_step=1000,
        print_interval=0, 
        verbose=True
        ):

        # Prepare
        n,k = x.shape[0],self.n_clusters
        
        if hasattr( alpha, '__iter__'):
            alpha = np.sort(np.array(alpha, dtype='float32'))
        else:
            alpha = alpha*np.ones(self.n_clusters,'float32')
        
        optimizer_name = optimizer_name.lower()
        optimizer = get_optimizer( optimizer_name,  steps=optimizer_step )
            
        # Training
        if verbose:
            print("\n>>> Start training for {} epochs".format(epoch_size), flush=True)
            print(optimizer.get_config(), flush=True)

        old_loss,to,itr = 1.,time.time(),0
        
        for epoch in range(1,epoch_size+1):        
            batched_input = tf.data.Dataset.from_tensor_slices(x).shuffle( int(n*10) ).batch(batch_size)
                
            # # Optimize
            loss = []
            for batch in batched_input:
                batched_loss = self.update(batch, alpha, optimizer)

            # # Will we print?
            if print_interval == 0:
                will_print = ( epoch%50 == 0 or epoch < 10 or (epoch%10==0 and epoch < 100) ) * (print_interval == 0)
            else:
                will_print  = ( print_interval > 0 and epoch % print_interval == 0 )
            will_print = will_print
            
            # # Print
            if will_print * verbose > 0:
                loss = self.loss(x, 0, False).numpy()
                
                print_line = t2s( time.time()-to )
                print_line += "\tepoch = {},\tloss = {:.3f} {:.3f} {:.3f} {:.3f} = {:.3f}, log diff=({:.1f}) lr={:.3f}  ".format(
                    epoch, 
                    loss[0]/n, loss[1]/k, loss[2]/k/k, loss[3], 
                    loss.sum(), 
                    np.log10( -(loss[:2].sum()-old_loss) / old_loss ),
                    np.log10(optimizer._decayed_lr(tf.float32).numpy())
                )
                if y is not None:
                    print_line += "\t"
                    for name, value in zip(['ARI','NMI','ACC'], self.predict( x, y )):
                        print_line += name+" = {:.1f} ".format( value*100 )
                print( print_line, flush=True)
                old_loss,to = loss.sum(),time.time()

        # Averaging Epoch
        if verbose:
            print(">>> Start averaging epoch")

        batched_input = tf.data.Dataset.from_tensor_slices(x).shuffle( int(n*10) ).batch(batch_size)
        temp_weights = [ w.numpy().copy()*0 for w in self.weights ]
        nb,to = 0,time.time()
        for batch in batched_input:
            nb += 1
            self.update(batch, alpha, optimizer)
            for iw,w in enumerate( self.weights ):
                temp_weights[iw] = temp_weights[iw] + w.numpy()
        self.set_weights([ w / nb for w in temp_weights])

        if verbose and y is not None:
            p = self.predict( x )
            print_line = "" + t2s( time.time()-to )
            for name, algo in zip(['ARI','NMI','ACC'], [ari,nmi,acc]):
                print_line += " "+name+" = {:.1f}".format( algo(y,p)*100 )
            print( print_line, flush=True)
            print(">>> Done.", flush=True)
    
        return epoch

