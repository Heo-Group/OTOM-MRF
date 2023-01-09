
import torch
import torch.nn as nn
import numpy as np
from lib.my_loss_MTC import my_loss

def evaluate(eval_model, data_source,device):
    eval_model.eval() # 평가 모드를 시작합니다.

    loss_test =0
    loss_T1w =0
    loss_Rm =0
    loss_Mm =0
    loss_T2m =0

    with torch.no_grad():
        for i, data in enumerate(data_source):

            X_batch,Y_batch=data
            X_batch,Y_batch = X_batch.to(device),Y_batch.to(device)
        
            x_pred_test = eval_model(X_batch)

            testloss,RMSE_T1w,RMSE_Rm,RMSE_Mm,RMSE_T2m = my_loss(x_pred_test,Y_batch,device) 
            
            loss_test += testloss.item()
            loss_T1w += RMSE_T1w.item()
            loss_Rm += RMSE_Rm.item()
            loss_Mm += RMSE_Mm.item()
            loss_T2m += RMSE_T2m.item()

    return loss_test/(i+1),loss_T1w/(i+1),loss_Rm/(i+1),loss_Mm/(i+1),loss_T2m/(i+1)