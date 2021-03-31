import os,argparse
from Tuned_Param import *

###################################################################

parser = argparse.ArgumentParser()
parser.add_argument("dataset", 
                        help="dataset.",
                        type=str,
                    )
parser.add_argument("init", 
                        help="Initialization",
                        choices=['rand','pre'], 
                        default='rand', 
                        type=str, 
                    )

parser.add_argument("-b","--batch", 
                        help="Batch size",
                        default=256,
                        type=int,
                    )

parser.add_argument("-e","--epoch", 
                        help="Number of epochs",
                        default=100,
                        type=int,
                    )

parser.add_argument("-r","--runs", 
                        help="Number of runs",
                        default=20,
                        type=int,
                    )

parser.add_argument("-g","--gpu", 
                        help="Which GPU to use",
                        default="",
                        type=str,
                    )

parser.add_argument("--draft", 
                        help="Is it a test? so we don't save.'",
                        action="store_true",
                    )
                    
args = parser.parse_args()


###################################################################

# Set this before loading the module
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

from Module.module_AE import AE
from Module.utils import *

############################################################################

NAME  = args.dataset.upper()
SAVE  = ~args.draft

LOAD = np.load('data/'+NAME+'.npz',allow_pickle=True)
DATA = LOAD['x'].astype('float32')
TRUE = LOAD['y']
del LOAD

N,D  = DATA.shape
K    = int( TRUE.max()+1 )
OUT = int(AECM_UNIF[NAME]['OUT'])


ARCHI = ([('input',D),
            ('dense', (500, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (500, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (2000, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (OUT, 'glorot_uniform', 'glorot_normal') ),
        ],[('input' , OUT),
            ('dense', (2000, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (500, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (500, 'glorot_uniform', 'glorot_normal') ),
            ('lrelu', .2 ),
            ('dense', (D, 'glorot_uniform', 'glorot_normal') ),
        ])

if SAVE:
    FNAME = NAME+'/save/save-ae.npz'
        
    if not os.path.exists(NAME+'/'):
        os.mkdir(NAME+'/')
    if not os.path.exists(NAME+'/save/'):
        os.mkdir(NAME+'/save/')
    print("*** I will save in ",FNAME)
    if os.path.exists(FNAME):
        print('Already done.')
        sys.exit()
        raise ValueError
    
OPTIM = {
    'MNIST':     'decay_sgd|2|.9',
    'USPS':      'decay_sgd|2|.9',
    'FMNIST':    'decay_sgd|2|0',
    'CIFAR10':   'decay_sgd|3|.9',

    'R10K':      'decay_sgd|2|.9',
    '20NEWS':    'decay_sgd|0|.9',
    '10X73K':    'decay_sgd|2|.9',
    'PENDIGIT':  'decay_sgd|3|.9',
}

LLK = []
LBL,kLBL = [],[]
ARI,NMI,ACC = [],[],[]
kARI,kNMI,kACC = [],[],[]
WGT,EPC = [],[]

for r in range(args.runs):
    print( "\n>>> "+NAME+": AE RUN=",r+1)
    MODEL = AE( 
        architecture=ARCHI, 
        n_clusters=K, 
        true_labels=TRUE, 
    )

    epc = MODEL.fit( 
        x=DATA,
        y=TRUE,
        batch_size=256, 
        epoch_size=args.epoch, 
        optimizer_name=OPTIM[NAME], 
        optimizer_step=int( (33)*(N/args.batch) ),
    )
    
    LLK.append( MODEL.loss_ee(DATA) )
    
    LBL.append( MODEL.predict(DATA) )
    ARI.append( ari( TRUE, LBL[-1] ) )
    NMI.append( nmi( TRUE, LBL[-1] ) )
    ACC.append( acc( TRUE, LBL[-1] ) )
    
    kLBL.append( LBL[-1] )
    kARI.append( ari( TRUE, kLBL[-1] ) )
    kNMI.append( nmi( TRUE, kLBL[-1] ) )
    kACC.append( acc( TRUE, kLBL[-1] ) )
    
    EPC.append( epc )
    
    WGT.append( [w.numpy() for w in MODEL.weights] )

    del MODEL
    
    print( 'ARI: {:.5} NMI: {:.5} ACC: {:.5} EPC: {:.5}'.format(
        np.mean(ARI), 
        np.mean(NMI), 
        np.mean(ACC), 
        np.mean(EPC)
        )
    )
    
    if SAVE:
        np.savez(FNAME,
            llk=LLK,
            wgt=WGT,
            lbl=LBL,klbl=kLBL,
            ari=ARI,nmi=NMI,acc=ACC,
            kari=kARI,knmi=kNMI,kacc=kACC,
            epc=EPC
        )
