import os
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean(y):
    k = 0
    while k < y.max():
        while (y!=k).all():
            y[y>k] = y[y>k]-1
        k += 1
    return y


def make_mnist():
    from tensorflow.keras.datasets import mnist as dataset
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    s = (x_train.shape, x_test.shape)
    np.savez('data/MNIST.npz',x=x,y=y,s=s)
    try:
        os.mkdir('MNIST')
        os.mkdir('MNIST/save')
    except FileExistsError:
        pass

def make_fmnist():
    from tensorflow.keras.datasets import fashion_mnist as dataset
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    s = (x_train.shape, x_test.shape)
    np.savez('data/FMNIST.npz',x=x,y=y,s=s)
    try:
        os.mkdir('FMNIST')
        os.mkdir('FMNIST/save')
    except FileExistsError:
        pass
        

def make_cifar10():
    from tensorflow.keras.datasets import cifar10 as dataset
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).squeeze()
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    s = (x_train.shape, x_test.shape)
    np.savez('data/CIFAR10.npz',x=x,y=y,s=s)
    try:
        os.mkdir('CIFAR10')
        os.mkdir('CIFAR10/save')
    except FileExistsError:
        pass
        

def make_20news():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
    news = fetch_20newsgroups(shuffle=False, subset='all')
    #x = TfidfVectorizer(dtype=np.float64, max_features=2000).fit_transform(news.data).toarray()
    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(news.data)
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x).toarray()
    y = news.target
    np.savez('data/20NEWS.npz',x=x,y=y)
    try:
        os.mkdir('20NEWS')
        os.mkdir('20NEWS/save')
    except FileExistsError:
        pass


def make_usps():
    import os
    if not os.path.exists('./data/usps/usps_train.jf'):
        os.system('unzip ./usps.zip')

    with open('./data/usps/usps_train.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_train, labels_train = data[:, 1:], data[:, 0]

    with open('./data/usps/usps_test.jf') as f:
        data = f.readlines()
    data = data[1:-1]
    data = [list(map(float, line.split())) for line in data]
    data = np.array(data)
    data_test, labels_test = data[:, 1:], data[:, 0]

    x = np.concatenate((data_train, data_test)).astype('float64')
    x = x/x.max()
    y = np.concatenate((labels_train, labels_test))
    s = (data_train.shape, data_test.shape)
    np.savez('./data/USPS.npz',x=x,y=y,s=s)
    try:
        os.mkdir('USPS')
        os.mkdir('USPS/save')
    except FileExistsError:
        pass


def make_reuters():
    import os
    if not os.path.exists('./data/reuters/lyrl2004_tokens_test_pt0.dat'):
        os.system('unzip ./reuters.zip')
        os.system('sh ./data/reuters/get_reuters.sh')
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open('./data/reuters/rcv1-v2.topics.qrels') as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        did_keys = list(did_to_cat.keys())
        for did in did_keys:
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open('./data/reuters/'+ dat) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    x = x[:10000]
    y = y[:10000]
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])

    p = np.random.permutation(x.shape[0])
    x = x[p]
    y = y[p]

    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], int(x.size / x.shape[0])))
    np.savez('data/R10K.npz',x=x,y=y)
    try:
        os.mkdir('R10K')
        os.mkdir('R10K/save')
    except FileExistsError:
        pass

def make_pendigit():
    # https://github.com/sudiptodip15/ClusterGAN/blob/master/pendigit/__init__.py
    finp_tr = './data/pendigit/pendigits.tra.txt'
    finp_tes = './data/pendigit/pendigits.tes.txt'
    data_tr = np.loadtxt(finp_tr, delimiter=',')
    x_train = data_tr[:, 0:16]
    x_train /= 100.0
    y_train = data_tr[:, -1].astype(int)

    data_tes = np.loadtxt(finp_tes, delimiter=',')
    x_test = data_tes[:, 0:16]
    x_test /= 100.0
    y_test = data_tes[:, -1].astype(int)

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).astype(int)
    s = (x_train.shape, x_test.shape)
    np.savez('data/PENDIGIT.npz',x=x,y=y,s=s)
    try:
        os.mkdir('PENDIGIT')
        os.mkdir('PENDIGIT/save')
    except FileExistsError:
        pass


def make_10x73k():
    import os
    from scipy.io import mmread
    # https://github.com/sudiptodip15/ClusterGAN/blob/master/10x_73k/__init__.py
    
    if not os.path.exists('./data/10x_73k/sub_set-720.mtx'):
        os.system('tar -xjvf ./data/10x_73k/10x_73k.tar.bz -C ./data/10x_73k')
    
    total_size = 73233
    train_size = 60000
    test_size = 13233

    def _read_mtx(filename):
        buf = mmread(filename)
        return buf

    def _load_gene_mtx():
        data_path = './data/10x_73k/sub_set-720.mtx'
        data = _read_mtx(data_path)
        data = data.toarray()
        data = np.log2(data + 1)
        scale = np.max(data)
        data = data / scale

        np.random.seed(0)
        indx = np.random.permutation(np.arange(total_size))
        data_train = data[indx[0:train_size], :]
        data_test = data[indx[train_size:], :]

        return data_train, data_test


    def _load_labels():
        data_path = './data/10x_73k/labels.txt'
        labels = np.loadtxt(data_path).astype(int)

        np.random.seed(0)
        indx = np.random.permutation(np.arange(total_size))
        labels_train = labels[indx[0:train_size]]
        labels_test = labels[indx[train_size:]]
        return labels_train, labels_test

    x_train, x_test = _load_gene_mtx()
    y_train, y_test = _load_labels()
    
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).astype(int)
    s = (x_train.shape, x_test.shape)
    
    y = clean(y)
        
    np.savez('data/10X73K.npz',x=x,y=y,s=s)
    try:
        os.mkdir('10X73K')
        os.mkdir('10X73K/save')
    except FileExistsError:
        pass

    
if __name__ == "__main__":
    if not os.path.exists('./data/'):
        os.mkdir('./data/')

    for name,make in [
#        ('MNIST',make_mnist), \
#        ('FMNIST',make_fmnist), \
#        ('USPS',make_usps), \
        ('R10K',make_reuters), \
#        ('20NEWS',make_20news), \
#        ('PENDIGIT',make_pendigit), \
#        ('10X73K',make_10x73k), \
#        ('CIFAR10',make_cifar10), \
    ]:
        print("Make",name)
        try:
            make()
        except:
            print("Failed...")

    print("Done.")
