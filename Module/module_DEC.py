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

class DEC(tf.keras.Model):
    def __init__(self, architecture, n_clusters, isIDEC, true_labels=None, alpha=1., gamma=.1):
        super(DEC, self).__init__()

        tf.keras.backend.clear_session()

        self.input_dim = architecture[0][0][1]
        self.code_dim = architecture[1][0][1]
        
        self.n_clusters = n_clusters
        self.true_labels = true_labels
        self.isIDEC= isIDEC
        self.gamma = gamma
        self.alpha = alpha
        
        # architecture = [(type of layer, size), ...]
        self.mu = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.n_clusters, )) ,
            tf.keras.layers.Dense(units=self.code_dim, \
                        activation=None, \
                        kernel_initializer=tf.initializers.glorot_uniform(), \
                        use_bias=False, \
                        name="mu" )
            ])
        
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
    def get_p_q(self, x, training=False, is_code=False):
        if is_code:
            z  = x
        else:
            z  = self.encode(x,training)
            
        u  = self.mu( np.eye(self.n_clusters).astype('float32'), False )

        p1 = tf.matmul(
            tf.expand_dims( tf.reduce_sum(tf.square(z), 1), 1),
            tf.ones(shape=(1, self.n_clusters))
        )
        p2 = tf.transpose(tf.matmul(
            tf.reshape( tf.reduce_sum(tf.square(u), 1), shape=[-1, 1]),
            tf.ones(shape=(len(z), 1)),
            transpose_b=True
        ))
        dist = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(z, u, transpose_b=True))
        q = 1.0/(1.0+dist**2/self.alpha)**((self.alpha+1.0)/2.0)
        q = (q/tf.reduce_sum(q, axis=1, keepdims=True))
        
        p = q**2 / tf.reduce_sum(q,axis=0)
        p = p / tf.reduce_sum(p, axis=1, keepdims=True)
        return p,q
            
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
    def loss(self, x, p, training=False):
        z  = self.encode(x, training)
        
        _,q = self.get_p_q( z, training, is_code=True)

        Lc = tf.reduce_mean( tf.losses.kld(p, q) ) 
        if self.isIDEC:
            Lc = Lc * self.gamma
            tx = self.decode(z, training)
            Lr = tf.reduce_mean( tf.losses.mean_squared_error(x,tx) )
        else:
            Lr = 0.
            
        return tf.stack( ( Lc , Lr) )

    @tf.function
    def update(self, batch, p_batch, optimizer):
        with tf.GradientTape() as tape:
            batched_loss = self.loss(batch, p_batch, True)
            batched_loss_summed = tf.reduce_sum(batched_loss)
        if self.isIDEC:
            gradients = tape.gradient( batched_loss_summed, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        else:
            gradients = tape.gradient( batched_loss_summed, self.trainable_variables[:-self.size_dec])
            optimizer.apply_gradients(zip(gradients, self.trainable_variables[:-self.size_dec]))
        return batched_loss
        
    def predict(self,x, y_true=None):
        n,y = len(x),[]
        _,q = self.get_p_q(x, False, False)
        q = tf.argmax(q,1).numpy()
        y+= list(q)
        y = np.array(y)
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
        return self.mu( tf.eye( self.n_clusters ), False).numpy()
        
    def cluster_centers_(self):
        return self.decode( self.mu( tf.eye( self.n_clusters ), False), False).numpy()
        
    
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

        old_loss,to,itr = 1.,time.time(),0
        
        p_input = self.get_p_q(x)[0].numpy()
        for epoch in range(1,epoch_size+1):        
            batched_input = tf.data.Dataset.from_tensor_slices( np.arange(n) ).shuffle( int(n*10) ).batch(batch_size)
                
            # # Optimize
            loss = []
            for batch in batched_input:
                batch = batch.numpy()
                loss.append( self.update(x[batch], p_input[batch], optimizer) )
                itr += 1
                if itr % update_p == 0:
                    p_input = self.get_p_q(x)[0].numpy()

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

