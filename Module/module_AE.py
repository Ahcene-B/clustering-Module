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

class AE(tf.keras.Model):
    def __init__(self, architecture, n_clusters, true_labels=None):
        super(AE, self).__init__()

        tf.keras.backend.clear_session()

        self.input_dim = architecture[0][0][1]
        self.code_dim = architecture[1][0][1]
        
        self.n_clusters = n_clusters
        self.true_labels = true_labels
        
        # architecture = [(type of layer, size), ...]
        self.encode = tf.keras.Sequential(name='encoder')
        for name, param in architecture[0]:
            self.encode.add( archi2layer(name, param) )
        
        self.decode = tf.keras.Sequential(name='decoder')
        for name, param in architecture[1]:
            self.decode.add( archi2layer(name, param) )
            
        self.size_dec = len(architecture[1])
        

    def reset(self):
        self.set_weights( [  tf.initializers.glorot_uniform()(( w.shape )).numpy() for w in self.weights])
            
    @tf.function
    def loss_ee(self, x, training=False):
        z  = self.encode(x, training)
        tx = self.decode(z, training)
        return tf.losses.mean_squared_error(x, tx)

    @tf.function
    def update_ee(self, batch, optimizer):
        with tf.GradientTape() as tape:
            batched_loss = self.loss_ee(batch, True)
        gradients = tape.gradient( batched_loss, self.trainable_variables[1:])
        optimizer.apply_gradients(zip(gradients, self.trainable_variables[1:]))
        return batched_loss
        
    def predict(self,x, y_true=None):
        return self.predict_km(x,y_true)
        
    def predict_km(self,x, y_true=None):
        z = self.encode(x,False).numpy()
        y = KMeans(n_clusters=self.n_clusters).fit_predict(z)
        if y_true is None:
            return y
        else:
            return ari(y_true, y), nmi(y_true, y), acc(y_true, y) 
        
    def cluster_centers_code(self):
        return self.mu( tf.eye( self.n_clusters ), False).numpy()
        
    def cluster_centers_(self):
        return self.decode( self.mu( tf.eye( self.n_clusters ), False), False).numpy()
    
    def fit(self, x, 
        y=None, 
        update_p=20, 
        batch_size=256, 
        epoch_size=1000, 
        optimizer_name=None, 
        optimizer_step=1000,
        print_interval=0, 
        verbose=True
        ):

        # Prepare
        n,k = x.shape[0],self.n_clusters
        
        optimizer_name = optimizer_name.lower()
        optimizer = get_optimizer( optimizer_name,  steps=optimizer_step )
            
        # Training
        if verbose:
            print("\n>>> Start training for {} epochs".format(epoch_size), flush=True)
            print(optimizer.get_config(), flush=True)

        old_loss,to = 1.,time.time()
        
        kmeans = KMeans(n_clusters=self.n_clusters,init='k-means++',n_init=1,max_iter=10)
        old_loss,to = 1.,time.time()
        for epoch in range(1,epoch_size+1):
            batched_input = tf.data.Dataset.from_tensor_slices(x).shuffle( int(n*10) ).batch(batch_size)

            # # Will we print?
            if print_interval == 0:
                will_print = ( epoch%50 == 0 or epoch < 10 or (epoch%10==0 and epoch < 100) ) * (print_interval == 0)
            else:
                will_print  = ( print_interval > 0 and epoch % print_interval == 0 )
            will_print = will_print
                
            # # Optimize
            loss = []
            for batch in batched_input:
                loss += list( self.update_ee(batch, optimizer) )
            
            # # Print and Track
            
            if will_print:
                loss = sum(loss)/n
                p = kmeans.fit_predict( self.encode( x, False).numpy() )
                
                print_line = t2s( time.time()-to )
                print_line += "\tepoch = {},\tloss = {:.3f}, ({:.1f}) ari={:.1f} ({:.2f})".format(
                    epoch, 
                    loss, 
                    np.log10(-(loss-old_loss)/old_loss) ,
                    ari(y,p)*100,
                    np.log10(optimizer._decayed_lr(tf.float32).numpy()),
                )
                print( print_line, flush=True)
                old_loss,to = loss,time.time()

        if verbose:
            print(">>> Done.")

        return epoch
