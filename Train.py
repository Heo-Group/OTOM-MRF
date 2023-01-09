import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import time
import os
import argparse

from torch.utils.data import TensorDataset, DataLoader
from lib.Model_LSTM import nnModel
from lib.my_loss_MTC import my_loss
from lib.Evaluate import evaluate

parser = argparse.ArgumentParser(description='Setting')
parser.add_argument('--result', type=str, default='result',help='The number of folder for the save of model and loss')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--schedular', type=int, default=5)
parser.add_argument('--portion_valid', type=float, default=0.1,help='Portion of validation dataset from train dataset')
# dataset parameters 
parser.add_argument('--trainset', type=str, default='100K')
parser.add_argument('--testset', type=str, default='10K')
# LSTM structure parameters 
parser.add_argument('--hidden_size', type=int, default=512,help='Hidden size of LSTM')
parser.add_argument('--layer', type=int, default=3,help='Number of layers for each LSTM')
parser.add_argument('--input', type=int, default=5,help='size of the input vector: MTC-MRF signal + four scan parameters')
parser.add_argument('--out_dim', type=int, default=4,help='dimension of output: number of target tissue parameters')
parser.add_argument('--Bi', type=str, default='bi',help='Either bi-directional or uni-directional')
parser.add_argument('--dropout', type=float, default=0)



if __name__ == '__main__':

    args = parser.parse_args()
    os.makedirs(args.result, exist_ok=True)
    with open(args.result+'/arguments.txt','w') as f:
        print(args,file = f)

    # GPU
    GPU_NUM = args.gpu # GPU number
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    ## Dataset load #########################################################################
    dir_data = "data/"

    X_mat = sio.loadmat(dir_data + "Train_2pool_MTC_with_RF_"+args.trainset+"_noise.mat")
    train_X = X_mat['MR_images']

    Y_mat = sio.loadmat(dir_data + "Train_Tissue_norm_MTC_"+args.trainset+".mat")
    train_Y = Y_mat['GT_Train_norm']

    X_mat = sio.loadmat(dir_data + "Test_2pool_MTC_with_RF_"+args.testset+"_noise.mat")
    test_X = X_mat['MR_images']

    Y_mat = sio.loadmat(dir_data + "Test_Tissue_norm_MTC_"+args.testset+".mat")
    test_Y = Y_mat['GT_Test_norm']


    X_train=torch.FloatTensor(train_X)
    Y_train=torch.FloatTensor(train_Y)

    X_test=torch.FloatTensor(test_X)
    Y_test=torch.FloatTensor(test_Y)
    ###Prepare data for batch learning ########################################################
    from torch.utils.data.dataset import random_split

    dataset  = TensorDataset(X_train,Y_train)

    train_len = int(np.shape(train_X)[0]*(1-args.portion_valid))
    valid_len = int(np.shape(train_X)[0]*args.portion_valid)
    train_dataset, val_dataset = random_split(dataset, [train_len,valid_len])

    trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch,shuffle=True)
    validloader   = DataLoader(dataset=val_dataset, batch_size=args.batch,shuffle=True)

    testset = TensorDataset(X_test,Y_test)
    testloader=DataLoader(testset,batch_size=args.batch,shuffle=False)

    ## Model training  ######################################################################
    input_size = args.input ; hidden_size = args.hidden_size; output_size = args.out_dim
    num_layers = args.layer ; dropout = args.dropout 
    if args.Bi == 'bi':
        bidirectional = True
    else:
        bidirectional = False


    rnn = nnModel(input_size, hidden_size, output_size, num_layers, dropout, bidirectional,device)
    rnn = rnn.to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = args.schedular, gamma=args.gamma)

    losses = []
    loss_test = []
    loss_T1w = []
    loss_Rm = []
    loss_Mm = []
    loss_T2m = []

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        start_time=time.time()
        batch_loss = 0 

        rnn.train()
        for i,data in enumerate(trainloader):
            X_batch,Y_batch=data
            X_batch,Y_batch = X_batch.to(device),Y_batch.to(device)
            x_pred = rnn(X_batch)

            loss,*_ = my_loss(x_pred,Y_batch,device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
                

        ## test
        with torch.no_grad(): 
            rnn.eval()
            val_loss, *_ = evaluate(rnn, validloader,device)
            test_loss,RMSE_T1w,RMSE_Rm,RMSE_Mm,RMSE_T2m  = evaluate(rnn, testloader,device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = rnn
                PATH=args.result+'/NN_model.pth'
                torch.save(rnn.state_dict(), PATH)


            print('Epoch:', '%04d' % (epoch + 1),'Time. taken =', '{:.4f}'.format(time.time()-start_time))
            print('Epoch:', '%04d' % (epoch + 1),'Train Loss =', '{:.4f}'.format(batch_loss / (i+1)))
            print('Epoch:', '%04d' % (epoch + 1),'Valid Loss =', '{:.4f}'.format(val_loss))
            print('Epoch:', '%04d' % (epoch + 1),'Test Loss =', '{:.4f}'.format(test_loss))
            print('=========================================================================================')

        scheduler.step()

        losses.append(batch_loss/ (i+1))
        loss_test.append(test_loss)
        loss_T1w.append(RMSE_T1w)
        loss_Rm.append(RMSE_Rm)
        loss_Mm.append(RMSE_Mm)
        loss_T2m.append(RMSE_T2m)

        
        sio.savemat(args.result+'/Trainloss'+'.mat',{'Trainloss': losses})
        sio.savemat(args.result+'/Testloss'+'.mat',{'Testloss': loss_test})
        sio.savemat(args.result+'/RMSE_T1w'+'.mat',{'RMSE_T1w': loss_T1w})
        sio.savemat(args.result+'/RMSE_Rm'+'.mat',{'RMSE_Rm': loss_Rm})
        sio.savemat(args.result+'/RMSE_Mm'+'.mat',{'RMSE_Mm': loss_Mm})
        sio.savemat(args.result+'/RMSE_T2m'+'.mat',{'RMSE_T2m': loss_T2m})    




