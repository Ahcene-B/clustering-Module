# Joint Optimization of an Autoencoder for Clustering and Embedding -- Code

This is the code corresponding to the experiments conducted for the work "Unsupervised Scalable Representation Learning for Multivariate Time Series"

## Requirements

Experiments were done with the following package versions for Python 3.8.3:
 - Numpy (`numpy`) v1.18.1;
 - Scikit-learn (`sklearn`) v0.22.1;
 - Tensorflow (`tensorflow`) 2.2.0.

Note that dkm.py requires an earlier version of Tensorflow
 - Tensorflow (`tensorflow`) 1.15.0.
 
## Datasets


The datasets will be generated using `python make_datsets.py`
 
## Usage

To train the clustering module (CM) on a dataset $DATASET with $INITIALIZATION={rand|pre} (random or pretrained) initialization:  
```
python CM.py $DATASET $INITIALIZATION -b BATCHSIZE -e EPOCH -p PRETRAINING_EPOCH -r NB_RUNS -g GPU_ID
```

The same function can be used for `AECM.py`, `DCN.py`, `DEC.py` and `IDEC.py`

The results will be saved for example in `$DATASET/save/save-cm-rand.npz`.

That archive contains 12 files: 
 - `llk` value of the loss,
 - `wgt` weights of the trained network, 
 - `lbl` predicted labels, 
 - `klbl `predicted labels by k-means on the embedding,
 - `ari`,`nmi`,`acc` Adjusted Rand index, Normalized Mutual Information and Accuracy of the predicted label (`lbl`),
 - `kari`,`knmi`,`kacc` Adjusted Rand index, Normalized Mutual Information and Accuracy of the predicted label by k-means (`klbl`),
 - `epc` Duration of the training.

To train DKM, `dkm.py` accordingly to the indications provided in the [[authors' repository]](https://github.com/MaziarMF/deep-k-means).
