# %%
# remove these lines if not running on notebooks
#%matplotlib notebook
run_from_notebook = False

# %% [markdown]
# ## Import the required packages
# Insert here all the packages you require, so in case they are not found an error will be shown before any other operation is performed.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,random_split
from torchmetrics.functional import f1_score,matthews_corrcoef
import utils.performance as pf
from datetime import datetime as dt
from h_transformer_1d import HTransformer1D
from IPython.display import clear_output
import torchvision.transforms as transforms
import argparse

# %% [markdown]
# # GENCOVID custom Dataset class

# %%
class GENCOVID(Dataset):
    def __init__(self, experiment="DND", gender="m"):
        # Load data
        self.data    = pd.read_csv("./SienaNoiV2/data/FINAL_3/" + experiment + "_ds_" + gender + ".csv",header=0)
        
        # Set targets as patient grading
        self.targets = pd.read_csv("./SienaNoiV2/data/FINAL_3/" + experiment + "_labels_" + gender + ".csv",header=0)
        self.data.convert_dtypes("float")
        self.cols = self.data.shape[1]

    def __len__(self):
        
        return len(self.targets)

    def __getitem__(self,idx):
    
        return torch.LongTensor(self.data.iloc[idx]), torch.tensor(self.targets['grading'].iloc[idx])

# %% [markdown]
# ## Instance datasets

# %%
experiment = "TNT" #DND -> Dead or not Dead, "SNS" -> Severe or not Severe, Therapy or not Therapy
gender = "f"

# Load Training Set
dataset_train = GENCOVID(experiment=experiment,gender=gender)

# Random split for retrieving Validation set
torch.manual_seed(0)
train_set_size = int(len(dataset_train) * 0.8)
valid_set_size = len(dataset_train) - train_set_size

#Using Validation as Test
dataset_train,dataset_valid = random_split(dataset=dataset_train,lengths=[train_set_size,valid_set_size])

# Some Debug
print("length dataset_train:",len(dataset_train))
print("length dataset_valid:",len(dataset_valid))

# %% [markdown]
# ## Set hyperparameters and options
# Set here your hyperparameters (to be used later in the code), so that you can run and compare different experiments operating on these values. 
# <br>_Note: a better alternative would be to use command-line arguments to set hyperparameters and other options (see argparse Python package)_

# %%
# hyperparameters
batch_size        = 1
learning_rate     = 1e-4
epochs            = 100
momentum          = 0.9
lr_step_size      = 1000   # if < epochs, we are using decaying learning rate
lr_gamma          = 0.01   #0.01
data_augmentation = False   ##########################
activation        = nn.GELU()
input_dim         = dataset_train.dataset.cols
num_tokens        = 256     # number of tokens

causal            = False   # autoregressive or not
max_seq_len       = 128*128 # maximum sequence length


block_size        = 2       # block size
reversible        = False   # use reversibility, to save on memory with increased depth
shift_tokens      = False   # whether to shift half the feature space by one along the sequence dimension, for faster convergence (experimental feature)
n_class           = 2

parser            = argparse.ArgumentParser()
parser.add_argument("--token_size", type = int)
parser.add_argument("--heads",      type = int)
parser.add_argument("--depth",      type = int)
args              = parser.parse_args()

dim               = args.token_size      # dimension
heads             = args.heads           # heads
depth             = args.depth           # depths

dim_head          = 32 #dim // heads      # dimension per head

# Create Hyperparameter Dictionary for Experiment Report
hyperparameters = {'Batch Size'             : batch_size,
                   'Learning Rate'          : learning_rate,
                   'Epochs'                 : epochs,
                   'Momentum'               : momentum,
                   'Activation Fcn'         : activation,
                   'Input Dimension'        : input_dim,
                   'num_tokens'             : num_tokens,
                   'dimension'              : dim,
                   'depth'                  : depth,            
                   'autoregressive or not'  : causal,           
                   'maximum sequence length': max_seq_len,    
                   'heads'                  : heads,
                   'dimension per head'     : dim_head,
                   'block size'             : block_size,
                   'use reversibility'      : reversible,
                   'whether to shift'       : shift_tokens}

# make visible only one GPU at the time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # <-- should be the ID of the GPU you want to use
# options
device          = "cuda:0"  # put here "cuda:0" if you want to run on GPU
monitor_display = True      # whether to display monitored performance plots
display_first_n = 0         # how many samples/batches are displayed
num_workers     = 2         # how many workers (=threads) for fetching data
pretrained      = False     # whether to test a pretrained model (to be loaded) or train a new one
display_errors  = True      # whether to display errors (only in pretrained mode)
save_model_tar  = False     # Save the model for test?

# %% [markdown]
# ## Hierarchical Transformer 
# This is the architecture of the net

# %%
class HIGENTRA(nn.Module):
    def __init__(self,
                 num_tokens   : int,     # number of tokens
                 dim          : int,     # dimension
                 depth        : int,     # depth
                 causal       : bool,    # autoregressive or not
                 max_seq_len  : int,     # maximum sequence length
                 heads        : int,     # heads
                 dim_head     : int,     # dimension per head
                 block_size   : int,     # block size
                 reversible   : bool,    # use reversibility, to save on memory with increased depth
                 shift_tokens : bool,    # whether to shift half the feature space by one along the sequence dimension, for faster convergence (experimental feature)
                 n_class      : int,     # class number for classification
                 activation   = nn.SiLU()
    ):
        super(HIGENTRA,self).__init__()

        """
        ## This is a Hierarchical 1D Transformer
        Architecture:
        -----------------------------------------
        - Encoder --> Encode a tokenized 1D sequence into same size 1D sequence but containing Self-Attention Informations combined
        - CLS Token --> Token added to tokenized sequence random initialized, used for classification purpose (see Vision Transformer)
        - Multi Layer Perceptron --> Once selected the CLF token, MLP gives as output class confident scores
        -----------------------------------------
        Parameters:
        num_tokens   : int,     # number of tokens
        dim          : int,     # dimension
        depth        : int,     # depth
        causal       : bool,    # autoregressive or not
        max_seq_len  : int,     # maximum sequence length
        heads        : int,     # heads
        dim_head     : int,     # dimension per head
        block_size   : int,     # block size
        reversible   : bool,    # use reversibility, to save on memory with increased depth
        shift_tokens : bool,    # whether to shift half the feature space by one along the sequence dimension, for faster convergence (experimental feature)
        n_class      : int,     # class number for classification
        activation   = nn.SiLU()# Activation Function used for MLP Head
        """        
        # Encoder
        self.encoder = HTransformer1D(num_tokens = num_tokens, dim = dim,depth = depth, causal = causal,max_seq_len = max_seq_len, heads = heads, dim_head = dim_head, block_size = block_size, reversible = reversible, shift_tokens = shift_tokens)
        
        # MLP Head Classification
        self.MLP = nn.Linear(dim,n_class)


    def forward(self,x):
        
        x = self.encoder(x)
        
        x = x[:,0,:]
        
        return self.MLP(x)

        #return nn.functional.softmax(self.MLP(x),1)

# %% [markdown]
# ## Create the building blocks for training
# Create an instance of the network, the loss function, the optimizer, and learning rate scheduler.

# %%
net = HIGENTRA(num_tokens   = num_tokens,      # number of tokens
               dim          = dim,             # dimension
               depth        = depth,           # depth
               causal       = causal,          # autoregressive or not
               max_seq_len  = max_seq_len,     # maximum sequence length
               heads        = heads,           # heads
               dim_head     = dim_head,        # dimension per head
               block_size   = block_size,      # block size
               reversible   = reversible,      # use reversibility, to save on memory with increased depth
               shift_tokens = shift_tokens,    # whether to shift half the feature space by one along the sequence dimension, for faster convergence (experimental feature)
               n_class      = n_class,         # Number of classes
               activation   = activation
               )
               
# create loss function
criterion = nn.CrossEntropyLoss()

# create SGD optimizer
optimizer = optim.Adam(params=net.parameters(),lr=learning_rate) # optim.SGD(net.parameters(),lr=learning_rate,momentum=momentum) 

# create learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

# experiment ID
experiment_ID = "%s_%s_%s_bs(%d)lr(%.4f_%d_%.1f)m(%.1f)e(%d)act(%s)" % (type(net).__name__ + gender.upper(), type(criterion).__name__, type(optimizer).__name__,
                batch_size, learning_rate, lr_step_size, lr_gamma, momentum, epochs, type(activation).__name__)

# %% [markdown]
# ## Data Augmentation

# %%
transform_train = transforms.RandomApply(transforms=transforms.RandomHorizontalFlip(p=0.3),p=0.3)

if data_augmentation:
    dataset_train.transform = transform_train
    print(f"After Data Augmentation dataset length: {len(dataset_train)}")


# %% [markdown]
# ## Create data loaders
# Dataloaders are in-built PyTorch objects that serve to sample batches from datasets. 

# %%
# create data loaders
# NOTE 1: shuffle helps training
# NOTE 2: in test mode, batch size can be as high as the GPU can handle (faster, but requires more GPU RAM)
# create dataset and dataloaders
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# %% [markdown]
# ## Define train function
# It is preferable (but not mandatory) to embed training (1 epoch) code into a function, and call that function later during the training phase, at each epoch.

# %%
# define train function (1 epoch)
def train(dataset, dataloader):

    # switch to train mode
    net.train()

    # reset performance measures
    loss_sum = 0.0
    
    # initialize predictions
    predictions    = torch.zeros(len(dataset), dtype=torch.int64)
    groundtruth    = torch.zeros(len(dataset), dtype=torch.int64)
    sample_counter = 0
    target_counter = 0

    # 1 epoch = 1 complete loop over the dataset
    for batch in dataloader:

        # get data from dataloader
        inputs, targets = batch

        # move data to device
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = net(inputs)

        # calculate loss
        loss = criterion(outputs, targets)

        # loss gradient backpropagation
        loss.backward()

        # net parameters update
        optimizer.step()

        # accumulate loss
        loss_sum += loss.item()
        
        # store predictions
        outputs_max = torch.argmax(outputs, dim=1)

        for output in outputs_max:
            predictions[sample_counter] = output
            sample_counter += 1
        
        for target in targets:
            groundtruth[target_counter] = target
            target_counter += 1    

    # step learning rate scheduler
    scheduler.step()

    # return average loss and accuracy
    return loss_sum / len(dataloader), predictions, groundtruth

# %% [markdown]
# ## Define test function
# It is preferable (but not mandatory) to embed the test code into a function, and call that function whenever needed. For instance, during training for validation at each epoch, or after training for testing, or for deploying the model.

# %%
# define test function
# returns predictions
def test(dataset, dataloader):

    # switch to test mode
    net.eval()  

    # initialize predictions
    predictions = torch.zeros(len(dataset), dtype=torch.int64)
    true_pred   = torch.zeros(len(dataset), dtype=torch.int64)
    sample_counter = 0
    target_counter = 0

    # do not accumulate gradients (faster)
    with torch.no_grad():

        # test all batches
        for batch in dataloader:

            # get data from dataloader
            inputs,targets = batch

            # move data to device
            inputs,targets = inputs.to(device, non_blocking=True),targets.to(device,non_blocking=True)

            # forward pass
            outputs = net(inputs)

            # store predictions
            outputs_max = torch.argmax(outputs, dim=1)
            for output in outputs_max:
                predictions[sample_counter] = output
                sample_counter += 1
            
            for target in targets:
                true_pred[target_counter] = target
                target_counter += 1            


    return predictions, true_pred

# %% [markdown]
# ## Train a new model or test a pretrained one
# The code below also includes visual loss/accuracy monitoring during training, both on training and validation sets. 

# %%
# pretrained model not available --> TRAIN a new one and save it
if not pretrained:

    # Create performance folder
    timestamp = (dt.now()).strftime("%Y%m%d_%H%M%S")
    path = experiment + "_" + gender + "_" + timestamp
    os.mkdir(path)

    # reset performance monitors
    losses = []
    train_mccs = []
    valid_mccs = []
    ticks = []

    with open(path + "/" + "experiment_summary.txt",mode='w', encoding="utf-8") as f:
        print(f"***** HYPERPARAMETERS *****",file=f)
        
        for key in hyperparameters:
            print(f"{key} : {hyperparameters[key]}",file=f)
        
        print(f"\n\n ***** MODEL ARCHITECTURE *****\n {net}",file=f)

    f.close()

    # move net to device
    net.to(device)
    # start training
    for epoch in range(1, epochs+1):

        # measure time elapsed
        t0 = time.time()
        
        # train
        avg_loss,train_pred,train_true  = train(dataset_train, dataloader_train)

        acc_train = 100.* (train_pred.eq(train_true).sum().float())/len(dataset_train) 

        # F1-Score train
        f1_train = 100.*f1_score(train_pred,train_true,average='micro')

        # MCC Score Train
        mcc_train = matthews_corrcoef(train_pred,train_true,2)


        # test on validation
        predictions,true_pred = test(dataset_valid, dataloader_valid)
        acc_valid = 100. * predictions.eq(true_pred).sum().float() / len(dataset_valid)
          
        # F1-Score validation 
        f1_valid = 100. * f1_score(predictions,true_pred,average='micro')
        
        # MCC Score Validation
        mcc_valid = matthews_corrcoef(predictions,true_pred,2)

        # update performance history
        losses.append(avg_loss)
        train_mccs.append(mcc_train.cpu())
        valid_mccs.append(mcc_valid.cpu())
        ticks.append(epoch)


        # print or display performance
        if not monitor_display:
            print ("\nEpoch %d\n"
                "...TIME: %.1f seconds\n"
                "...loss: %g (best %g at epoch %d)\n"
                "...training accuracy: %.2f%% (best %.2f%% at epoch %d)\n"
                "...validation accuracy: %.2f%% (best %.2f%% at epoch %d)" % (
                epoch,
                time.time()-t0,
                avg_loss, min(losses), ticks[np.argmin(losses)],
                mcc_train, max(train_mccs), ticks[np.argmax(train_mccs)],
                mcc_valid, max(valid_mccs), ticks[np.argmax(valid_mccs)]))
        else:
            fig, ax1 = plt.subplots(figsize=(12, 8), num=1)
            ax1.set_xticks(np.arange(0, epochs+1, step=epochs/10.0))
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel(type(criterion).__name__, color='blue')
            ax1.set_ylim(0.0001, 1)
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_yscale('log')
            ax1.plot(ticks, losses, 'b-', linewidth=1.0, aa=True, 
                label='Training (best at ep. %d)' % ticks[np.argmin(losses)])
            ax1.legend(loc="lower left")
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('MCC Score', color='red')
            ax2.set_ylim(-1, 1)
            ax2.set_yticks(np.arange(-1, 1, step=1e-1))
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.plot(ticks, train_mccs, 'r-', linewidth=1.0, aa=True, 
                label='Training (%.2f, best %.2f at ep. %d)' % (mcc_train, max(train_mccs), ticks[np.argmax(train_mccs)]))
            ax2.plot(ticks, valid_mccs, 'r--', linewidth=1.0, aa=True, 
                label='Validation (%.2f, best %.2f at ep. %d)' % (mcc_valid, max(valid_mccs), ticks[np.argmax(valid_mccs)]))
            ax2.legend(loc="lower right")
            plt.xlim(0, epochs+1)
            # this works if running from notebooks
            if run_from_notebook:
                fig.show()
                fig.canvas.draw()
            # this works if running from console
            else:
                plt.draw()
                plt.savefig(path + "/" + experiment_ID + ".png", dpi=300)
                plt.show()
                clear_output(wait=True)
            
            fig.clear()

        # save model if validation performance has improved
        if (epoch-1) == np.argmax(valid_mccs):
            # torch.save({
            #     'net': net,
            #     'accuracy': max(valid_accuracies),
            #     'epoch': epoch
            # }, experiment_ID + ".tar")
            # Performance Evaluation Training Set
            pf.eval_performance(path            = path,
                                experimentID    = experiment_ID,
                                training        = True,
                                hyperparameters = hyperparameters,
                                y_pred          = train_pred, 
                                y_true          = train_true)
                                    
            # Performance Summary Validation Set
            pf.eval_performance(path            = path,
                                experimentID    = experiment_ID,
                                training        = False,
                                hyperparameters = hyperparameters,
                                y_pred          = predictions.to("cpu"), 
                                y_true          = true_pred.to("cpu"))
