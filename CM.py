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
                        default=150,
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

from Module.module_CM import CM
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
ALPHA = CM_UNIF[NAME]['CC']
BATCH = int(CM_UNIF[NAME]['BATCH'])

if SAVE:
    FNAME = NAME+'/save/save-cm-'+ INIT + '.npz'
        
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
LBL = []
ARI,NMI,ACC = [],[],[]
EPC = []

for r in range(args.runs):
    print( "\n>>> "+NAME+": CM+"+INIT+" RUN=",r+1)
    MODEL = CM( 
        input_dim=D, 
        n_clusters=K, 
        true_labels=TRUE, 
    )
    
    if INIT == 'pre':
        MODEL.pre_fit( 
            x=DATA, 
            y=TRUE,
            verbose=True,
        )

    epc = MODEL.fit( 
        x=DATA,
        y=TRUE,
        alpha=ALPHA,
        batch_size=BATCH, 
        epoch_size=args.epoch, 
        optimizer_name='adam|3',
        print_interval=0, 
        verbose=True,
    )
    
    LLK.append( MODEL.loss(DATA,0) )
    
    LBL.append( MODEL.predict(DATA) )
    ARI.append( ari( TRUE, LBL[-1] ) )
    NMI.append( nmi( TRUE, LBL[-1] ) )
    ACC.append( acc( TRUE, LBL[-1] ) )
        
    EPC.append( epc )
    
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
            lbl=LBL,
            ari=ARI,nmi=NMI,acc=ACC,
            epc=EPC
        )
