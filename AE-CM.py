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
                        default=1000,
                        type=int,
                    )

parser.add_argument("-p","--pre_epoch", 
                        help="Number of pre_epochs",
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

from Module.module_AECM import AECM
from Module.utils import *

############################################################################

NAME  = args.dataset.upper()
INIT  = args.init.lower()
SAVE  = ~args.draft

LOAD = np.load('data/'+NAME+'.npz',allow_pickle=True)
DATA = LOAD['x'].astype('float32')
TRUE = LOAD['y']
del LOAD

N,D  = DATA.shape
K    = int( TRUE.max()+1 )
OUT = int(AECM_UNIF[NAME]['OUT'])
BETA = AECM_UNIF[NAME]['BETA']
ALPHA = AECM_UNIF[NAME]['CC']
BATCH = int(AECM_UNIF[NAME]['BATCH'])
LBD = 1. # CNET_UNIF[NAME]['LBD']

if INIT == 'pre':
    AE = np.load(NAME+'/save/save-ae.npz',allow_pickle=True)['wgt']

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
    FNAME = NAME+'/save/save-aecm-'+ INIT + '.npz'
        
    if not os.path.exists(NAME+'/'):
        os.mkdir(NAME+'/')
    if not os.path.exists(NAME+'/save/'):
        os.mkdir(NAME+'/save/')
    print("*** I will save in ",FNAME)
    if os.path.exists(FNAME):
        print('Already done.')
        sys.exit()
        raise ValueError

LLK = []
LBL,kLBL = [],[]
ARI,NMI,ACC = [],[],[]
kARI,kNMI,kACC = [],[],[]
WGT,EPC = [],[]

for r in range(args.runs):
    print( "\n>>> "+NAME+": AECM+"+INIT+" RUN=",r+1)
    MODEL = AECM( 
        architecture=ARCHI, 
        n_clusters=K, 
        true_labels=TRUE, 
        beta=BETA, 
        lbd=LBD
    )
    
    if INIT == 'pre':
        MODEL.pre_fit( 
            x=DATA, 
            y=TRUE,
            wgt=AE[r],
            alpha=ALPHA,
            batch_size=BATCH, 
            epoch_size=100, 
            optimizer_name='adam|3', 
            optimizer_step=int( 150*(N/BATCH) ),
            print_interval=50, 
            verbose=True,
        )

    epc = MODEL.fit( 
        x=DATA,
        y=TRUE,
        alpha=ALPHA, 
        batch_size=BATCH, 
        epoch_size=args.epoch, 
        optimizer_name='adam|3' if NAME not in ['CIFAR10','10X73K','USPS','PENDIGIT','R10K'] else 'adam_decay|3', 
        optimizer_step=int( 150*(N/BATCH) ),
        print_interval=50, 
        verbose=True
    )
    
    LLK.append( MODEL.loss(DATA,0) )
    
    LBL.append( MODEL.predict(DATA) )
    ARI.append( ari( TRUE, LBL[-1] ) )
    NMI.append( nmi( TRUE, LBL[-1] ) )
    ACC.append( acc( TRUE, LBL[-1] ) )
    
    kLBL.append( MODEL.predict_km(DATA) )
    kARI.append( ari( TRUE, kLBL[-1] ) )
    kNMI.append( nmi( TRUE, kLBL[-1] ) )
    kACC.append( acc( TRUE, kLBL[-1] ) )
    
    EPC.append( epc )
    
    if NAME == 'MNIST':
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
