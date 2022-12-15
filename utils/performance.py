import sklearn.metrics as metr
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.functional import f1_score
from torchmetrics import PrecisionRecallCurve

from pretty_confusion_matrix import pp_matrix_from_data

def eval_performance(path : str, experimentID : str, training : bool, hyperparameters : dict, y_pred, y_true):
    """
    This function displays and save confusion matrix fore chosen flag train or valid (training = False)
    path            : Current experiment path
    experimentID    : Experiment ID name
    training        : Flag for distinguish between TRAIN & TEST
    hyperparameters : Hyperparameters Dictionary contains all parameters
    y_pred          : Index of Net prediction 
    y_true          : Index of True labels tensor
    """

    conf_matrix(path=path,experimentID=experimentID,y_pred=y_pred,y_true=y_true,training=training)

    # eval_roc_curve(path=path,experimentID=experimentID,y_pred=y_pred[y_pred_idx],y_true=y_true[y_true_idx],training=training)
    
    



def conf_matrix(path : str,experimentID : str, training:bool, y_pred, y_true):

    if training:
        img_name = "TRAIN"
    else:
        img_name = "VALID"

    # Build confusion matrix
    pp_matrix_from_data(y_test = y_true,predictions=y_pred,fz=10,cmap="crest",path=(path + "/" + img_name + "_CM.png"),figsize=[12,9])
    
    plt.close()

    

def eval_roc_curve(path : str,experimentID : str, training:bool,scores, targets):
    
    pr_curve = PrecisionRecallCurve(num_classes=3)
    
    plt.figure(figsize=(12,9))
    
    plt.title(experimentID)
    
    #plt.plot(precision,recall,linewidth=1.5)
    
    if training:
        img_name = "TRAIN"
    else:
        img_name = "VALID"

    plt.savefig(path + "/" + img_name + "_PRC.png")
    
    plt.clf()