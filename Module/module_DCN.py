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

class DCN(tf.keras.Model):
    def __init__(self, architecture, n_clusters, true_labels=None, lbd=1.):
        super(DCN, self).__init__()

        tf.keras.backend.clear_session()

        self.input_dim = architecture[0][0][1]
        self.code_dim = architecture[1][0][1]
        
        self.n_clusters = n_clusters
        self.true_labels = true_labels

        self.lbd = lbd
        
        # architecture = [(type of layer, size), ...]
        self.mu = tf.initializers.glorot_uniform()((self.n_clusters,self.code_dim), dtype=tf.float32)
        
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
        
    @tf.function
    def get_s(self, x, training=False, is_code=False):
        if is_code:
            z = x
        else:
            z  = self.encode(x, training)
        
        d = tf.reduce_sum( tf.math.squared_difference( z, self.mu[:,None]) , 2 )
        s = tf.argmin( d ,0 )
        return s
        
    def get_bs(self, x, training=False, is_code=False):
        n,s = x.shape[0],[]
        for i in range(0,n,256):
            s+= self.get_s( x[i:i+256], training, is_code ).numpy().tolist()
        s = np.array(s)
        
        bs = binarizer(s, range(self.n_clusters+1))[:,:-1].astype('float32')
        return bs,s
            
    @tf.function
    def loss(self, x, bs, training=False):
        z  = self.encode(x, training)
        tx = self.decode(z, training)
        
        Lr = tf.reduce_sum( tf.math.squared_difference(x, tx) )
        Lc = .5 * self.lbd * tf.reduce_sum( tf.math.squared_difference( z, tf.matmul(bs, self.mu) ) )

        return tf.stack( (Lr, Lc) )

    @tf.function
    def update(self, batch, batch_bs, optimizer):
        with tf.GradientTape() as tape:
            batched_loss = self.loss(batch, batch_bs, True)
            batched_loss_summed = tf.reduce_sum(batched_loss)
        gradients = tape.gradient( batched_loss_summed, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return batched_loss

    def update_mu(self,x,s):
        n = x.shape[0]
        mu = self.mu.numpy()
        c = np.zeros((self.n_clusters), np.float32)
        for b in range(0,n,256):
            z = self.encode(x[b:b+256], False).numpy()
            for i,kk in enumerate(s[b:b+256]):
                k = int(kk)
                c[k] += 1
                mu[k] = mu[k] - (1 / c[k]) * ( mu[k] - z[i] )
        self.mu = tf.cast( mu , tf.float32)

        
    @tf.function
    def tf_predict(self, x):
        z = self.encode(x, False)
        d = tf.reduce_sum( tf.math.squared_difference( z, self.mu[:,None]) , 2 )
        p = tf.argmin( d ,0 )
        return p
        
    def predict(self,x, y_true=None):
        y = self.tf_predict(x).numpy()
        if y_true is None:
            return y
        else:
            return ari(y_true, y), nmi(y_true, y), acc(y_true, y) 
        
    def predict_km(self,x, y_true=None):
        z = self.encode(x,False).numpy()
        y = KMeans(n_clusters=self.n_clusters).fit_predict(z)
        if y_true is None:
            return y
        else:
            return ari(y_true, y), nmi(y_true, y), acc(y_true, y) 
        
    def cluster_centers_code(self):
        return self.mu.numpy()
        
    def cluster_centers_(self):
        return self.decode( self.mu, False).numpy()
    
    def pre_fit(self, x, 
            wgt=None,
            batch_size=256, 
            epoch_size=100, 
            optimizer_name=None,         
            optimizer_step=1000, 
            print_interval=50, 
            y=None, 
            verbose=True
        ):
        # Prepare
        n,k = x.shape[0],self.n_clusters
        
        if wgt is not None:
            # Init
            if verbose:
                print("\n>>> Init the AE")
                print(optimizer.get_config())
                
            nw = int(len(wgt)/2)
            self.encode.set_weights( wgt[:nw])
            self.decode.set_weights( wgt[nw:])
            
            # Run k-means++
            if verbose:
                print("\n>>> Run k-means")
            kmeans = KMeans(n_clusters=self.n_clusters,n_init=20)
            p = kmeans.fit_predict( self.encode( x, False).numpy() )
            
            if verbose and y is not None:
                print_line = "AE + K-means++\t"
                for name, algo in zip(['ARI','NMI','ACC'], [ari,nmi,acc]):
                    print_line += name+" = {:.1f} ".format( algo(y,p)*100 )
                print( print_line, flush=True)
            
            # Set the MU Weights
            u = kmeans.cluster_centers_
            self.set_weights( [u] + [w.numpy() for w in self.weights[1:]] )

            if verbose and y is not None:
                p = self.predict( x )
                print_line = "AE + MU\t"
                for name, algo in zip(['ARI','NMI','ACC'], [ari,nmi,acc]):
                    print_line += name+" = {:.1f} ".format( algo(y,p)*100 )
                print( print_line, flush=True)
                
        else:
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
                will_print = ( epoch % print_interval == 0 ) * verbose
                    
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
    
    def fit(self, x, 
        y=None, 
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
        
        bs_input = self.get_bs(x)[0]
        for epoch in range(1,epoch_size+1):
            batched_input = tf.data.Dataset.from_tensor_slices( np.arange(n) ).shuffle( int(n*10) ).batch(batch_size)
                
            # # Optimize
            loss = []
            for batch in batched_input:
                batch = batch.numpy()
                loss.append( self.update(x[batch], bs_input[batch], optimizer) )
                
            # Update Assignments
            bs_input,s_input = self.get_bs(x)
            
            # Update Centroids
            self.update_mu( x, s_input )
                    
            # # Will we print?
            if print_interval == 0:
                will_print = ( epoch%50 == 0 or epoch < 10 or (epoch%10==0 and epoch < 100) ) * (print_interval == 0)
            else:
                will_print  = ( print_interval > 0 and epoch % print_interval == 0 )
            will_print = will_print
            
            # # Print
            if will_print * verbose > 0:
                    
                b = len(loss)
                loss = np.sum(loss,0)
                
                print_line = t2s( time.time()-to )
                print_line += "\tepoch = {},\tloss = {:.3f} {:.3f} = {:.3f}, log diff=({:.1f}) lr={:.3f}  ".format(
                    epoch, 
                    loss[0], loss[1], 
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

        if verbose and y is not None:
            p = self.predict( x )
            print_line = "" + t2s( time.time()-to )
            for name, algo in zip(['ARI','NMI','ACC'], [ari,nmi,acc]):
                print_line += " "+name+" = {:.1f}".format( algo(y,p)*100 )
            print( print_line, flush=True)
            print(">>> Done.", flush=True)
    
        return epoch

