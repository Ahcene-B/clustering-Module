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

from Module.module_DEC import DEC
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
    FNAME = NAME+'/save/save-idec-'+ INIT + '.npz'
        
    if not os.path.exists(NAME+'/'):
        os.mkdir(NAME+'/')
    if not os.path.exists(NAME+'/save/'):
        os.mkdir(NAME+'/save/')
    print("*** I will save in ",FNAME)
    if os.path.exists(FNAME):
        print('Already done.')
        sys.exit()
        raise ValueError
    
UPDATE_P = {
    'MNIST': 140,
    'FMNIST': 140,
    'USPS': 30,
    'CIFAR10': 140,
    'R10K': 20,
    '20NEWS': 20,
    '10X73K': 20,
    'PENDIGIT': 20,
    }

LLK = []
LBL,kLBL = [],[]
ARI,NMI,ACC = [],[],[]
kARI,kNMI,kACC = [],[],[]
WGT,EPC = [],[]

for r in range(args.runs):
    print( "\n>>> "+NAME+": IDEC+"+INIT+" RUN=",r+1)
    MODEL = DEC( 
        architecture=ARCHI, 
        n_clusters=K, 
        isIDEC=True,
        true_labels=TRUE, 
        alpha=1., 
        gamma=.1
    )
    
    if INIT == 'pre':
        MODEL.pre_fit( 
            x=DATA, 
            y=TRUE,
            wgt=AE[r],
            verbose=True,
        )

    epc = MODEL.fit( 
        x=DATA,
        update_p=UPDATE_P[NAME],
        batch_size=args.batch, 
        epoch_size=args.epoch, 
        optimizer_name='adam_decay|3',
        optimizer_step=int( 150 * (N/args.batch) ),
        y=TRUE,
    )
    
    P = MODEL.get_p_q(DATA)[0]
    LLK.append( MODEL.loss(DATA,P) )
    
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
