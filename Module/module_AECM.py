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

class AECM(tf.keras.Model):
    def __init__(self, architecture, n_clusters, true_labels=None, beta=1., lbd=1.):
        super(AECM, self).__init__()

        tf.keras.backend.clear_session()

        self.input_dim = architecture[0][0][1]
        self.code_dim = architecture[1][0][1]
        
        self.n_clusters = n_clusters
        self.true_labels = true_labels
        self.lbd = lbd
        self.beta = beta
        
        self.gamma = tf.keras.Sequential( [
                tf.keras.layers.InputLayer(
                    input_shape=(self.code_dim,
                )),
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
                    units=self.code_dim, 
                    activation=None, 
                    kernel_initializer=tf.initializers.glorot_uniform(), 
                    bias_initializer=tf.initializers.glorot_normal(), 
                    name='mu' 
                ),
            ], name="cm_decoder" )
        
        # architecture = [(type of layer, size), ...]
        
        self.encode = tf.keras.Sequential(name='encoder')
        for name, param in architecture[0]:
            self.encode.add( archi2layer(name, param) )
        
        self.decode = tf.keras.Sequential(name='decoder')
        for name, param in architecture[1]:
            self.decode.add( archi2layer(name, param) )
        

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
    def loss_cm(self, z, alpha, training=False):
        x  = self.encode(z, False)
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
        
        loss_op4 = tf.reduce_sum( tf.math.log( tf.reduce_mean(g,0) +1e-10 ) * (1-alpha))

        return tf.stack( (loss_op1 , loss_op2 , loss_op3 , loss_op4) )

    @tf.function
    def update_cm(self, batch, alpha, optimizer):
        with tf.GradientTape() as tape:
            batched_loss = self.loss_cm(batch, alpha, True)
            batched_loss_summed = tf.reduce_sum(batched_loss)
        gradients = tape.gradient( batched_loss_summed, self.trainable_variables[:4])
        optimizer.apply_gradients(zip(gradients, self.trainable_variables[:4]))
        return batched_loss
            
    @tf.function
    def loss(self, x, alpha, training=False):
        c  = self.encode(x, training)
        g  = self.gamma( c, training)
        tc = self.mu(g, training)
        tx = self.decode(c, training)
        
        u  = self.mu( np.eye(self.n_clusters).astype('float32'), False)

        loss_op1 = tf.reduce_sum( tf.math.squared_difference(c, tc) )
        loss_op2 = tf.reduce_sum( tf.reduce_sum( g*(1-g),0 ) )
        
        uu = tf.linalg.matmul( u,u, transpose_b=True )
        loss_op2b = tf.reduce_sum( tf.abs(uu-tf.eye(self.n_clusters)) ) * self.lbd
        
        lmg = tf.math.log( tf.reduce_mean(g,0) +1e-10 )
        
        slmg = tf.sort(lmg)

        loss_op3 = tf.reduce_sum( slmg * (1-alpha) )
#        for k in range(self.n_clusters):
#            loss_op3 += gm[ogm[k]] * (1-alpha[k])
        
        loss_op4 = tf.reduce_sum( tf.math.squared_difference(x, tx) ) * self.beta

        return tf.stack( (loss_op1 , loss_op2 , loss_op2b , loss_op3 , loss_op4) )

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
        n,y = len(x),[]
        for i in range(0,n,256):
            p = self.tf_predict(x[i:i+256])
            y+= list(p)
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
        
    
    def pre_fit(self, x, alpha,
            y=None, 
            wgt=None,
            batch_size=256, 
            epoch_size=100, 
            optimizer_name=None,         
            optimizer_step=1000, 
            print_interval=50, 
            verbose=True
        ):
        # Prepare
        n,k,q = x.shape[0],self.n_clusters,self.code_dim
        
        if hasattr( alpha, '__iter__'):
            alpha = np.array(alpha, dtype='float32')
        else:
            alpha = alpha*np.ones(self.n_clusters,'float32')
        alpha.sort()
        
        if not hasattr( epoch_size, '__iter__'):
            epoch_size = [epoch_size,epoch_size]
            
        if wgt is not None:
            # Init
            if verbose:
                print("\n>>> Init the AE")
                
            nw = int(len(wgt)/2)
            self.encode.set_weights( wgt[:nw])
            self.decode.set_weights( wgt[nw:])
                
        else:
            optimizer_name = optimizer_name.lower()
            optimizer = get_optimizer( optimizer_name,  steps=optimizer_step )
                
            # Training
            if verbose:
                print("\n>>> Start training for {} epochs".format(epoch_size[0]), flush=True)
                print(optimizer.get_config(), flush=True)
            
            old_loss,to = 1.,time.time()
            
            kmeans = KMeans(n_clusters=self.n_clusters,init='k-means++',n_init=1,max_iter=10)
            old_loss,to = 1.,time.time()
            for epoch in range(1,epoch_size[0]+1):
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
        iu = np.linalg.pinv(u)
        
        # Set the CM's Weights
        self.set_weights( 
                        [iu, tf.zeros_initializer()((self.n_clusters,)).numpy(), 
                        u, tf.zeros_initializer()((self.code_dim,)).numpy()] + \
                        [w.numpy() for w in self.weights[4:]] )


        if verbose and y is not None:
            p = self.predict( x )
            print_line = "AE + CM\t"
            for name, algo in zip(['ARI','NMI','ACC'], [ari,nmi,acc]):
                print_line += name+" = {:.1f} ".format( algo(y,p)*100 )
            print( print_line, flush=True)
            
        # Train CM

        optimizer_name = optimizer_name.lower()
        optimizer = get_optimizer( optimizer_name,  steps=optimizer_step )

        # Training
        if verbose:
            print("\n>>> Start pre-training the CM for {} epochs".format(epoch_size[1]))
            print(optimizer.get_config())

        old_loss,to = 1.,time.time()
        for epoch in range(1,epoch_size[1]+1):
            batched_input = tf.data.Dataset.from_tensor_slices(x).shuffle( int(n*10) ).batch(batch_size)

            # # Will we print?
            will_print = ( epoch % print_interval == 0 ) * verbose
                
            # # Optimize
            loss = []
            for batch in batched_input:
                loss.append( self.update_cm(batch, alpha, optimizer) )
            
            # # Print and Track
            if will_print:
                b = len(loss)
                loss = np.array(loss).sum(0)*np.array([1.,b,b,b])
                
                print_line = t2s( time.time()-to )
                print_line += "\tepoch = {},\tloss {:.3f} {:.3f} {:.3f} {:.3f}= {:.3f}, ({:.1f}) ".format(
                    epoch, 
                    loss[0]/n, loss[1]/k, loss[2]/k/k, loss[3], 
                    loss.sum(), 
                    np.log10(-(loss.sum()-old_loss)/old_loss) 
                )
                if y is not None:
                    p = self.predict( x )
                    for name, algo in zip(['ARI','NMI','ACC'], [ari,nmi,acc]):
                        print_line += name+" = {:.1f} ".format( algo(y,p)*100 )

                print( print_line, flush=True)
                old_loss,to = loss.sum(),time.time()
                
        if verbose and y is not None:
            print_line = "AE + CM\t"
            p = self.predict( x )
            for name, algo in zip(['ARI','NMI','ACC'], [ari,nmi,acc]):
                print_line += name+" = {:.1f} ".format( algo(y,p)*100 )
            print( print_line, flush=True)


        if verbose:
            print(">>> Done.")
    
    def fit(self, x, alpha,
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
        
        if hasattr( alpha, '__iter__'):
            alpha = np.array(alpha, dtype='float32')
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
                loss.append( self.update(batch, alpha, optimizer) )

            # # Will we print?
            if print_interval == 0:
                will_print = ( epoch%50 == 0 or epoch < 10 or (epoch%10==0 and epoch < 100) ) * (print_interval == 0)
            else:
                will_print  = ( print_interval > 0 and epoch % print_interval == 0 )
            will_print = will_print
            
            # # Print
            if will_print * verbose > 0:
                loss = np.array(loss).sum(0)*np.array([1.,batch_size,batch_size,batch_size,1.])
                if 'l' in self.tracker:
                    self.tracker_dict[ 'loss' ].append( loss )
                
                print_line = t2s( time.time()-to )
                print_line += "\tepoch = {},\tloss = {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} = {:.3f}, ({:.1f}) {:.3f}  ".format(
                    epoch, 
                    loss[0]/n, loss[1]/k/n, loss[2]/k/n, loss[3]/n,  loss[4]/n, 
                    loss[:2].sum(), 
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

