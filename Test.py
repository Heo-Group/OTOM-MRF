import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import time
import os
import argparse
import h5py
from torch.utils.data import TensorDataset, DataLoader

from lib.Model_LSTM import nnModel

parser = argparse.ArgumentParser(description='Setting')
parser.add_argument('--result', type=str, default='result')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--out_dim', type=int, default=4)
parser.add_argument('--GAN', type=str, default='')
parser.add_argument('--testset', type=str, default='')
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--input', type=int, default=5)
parser.add_argument('--layer', type=int, default=3)
parser.add_argument('--bi', type=str, default='bi')
parser.add_argument('--dropout', type=float, default=0)

args = parser.parse_args() 

if __name__ == '__main__':

    args = parser.parse_args()
    os.makedirs(args.result, exist_ok=True)
    # GPU
    GPU_NUM = args.gpu # GPU number
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check

    ## Dataset #########################################################################
    dir_data = "data/phantom/"
    dir_data_model = "model/"

    ## Model training
    input_size = args.input ; hidden_size = args.hidden_size; output_size = args.out_dim
    num_layers = args.layer ; dropout = args.dropout
    if args.bi == 'bi':
        bidirectional = True
    else:
        bidirectional = False

    rnn = nnModel(input_size, hidden_size, output_size, num_layers, dropout, bidirectional,device)

    PATH=dir_data_model+'NN_model_MTC_LSTM_bi_new_3layer.pth'
    checkpoint=torch.load(PATH,map_location=device)
    rnn.load_state_dict(checkpoint)
    rnn = rnn.to(device)



    Tisseu_param =['Kex','Mom','T2m','T1w']

    for i in range(4):
        X_mat = sio.loadmat(dir_data + "Test_MTC_RF_Phantom_"+Tisseu_param[i]+"_LOAS40_with_B0_1p2_rB1_0p5_noise.mat")
        X2_mat = sio.loadmat(dir_data + "Test_quant_Phantom_"+Tisseu_param[i]+"_with_B0_1p2_rB1_0p5.mat")

        test_X = X_mat['MR_images']
        test_GT = X2_mat['phantom']
        test_B0 = test_GT[:,5:6]
        test_rB1 = test_GT[:,6:7]
        
        X_test=torch.FloatTensor(test_X)
        B0_test=torch.FloatTensor(test_B0)
        rB1_test=torch.FloatTensor(test_rB1)

        testset = TensorDataset(X_test,B0_test,rB1_test)
        testloader=DataLoader(testset,batch_size=args.batch,shuffle=False)

        data_length=X_test.size(0)
        MR_images_test=torch.zeros([data_length,args.out_dim],device=device)
        rnn.eval()
        for epoch in range(args.epochs):
            start_time=time.time()

                ## testidation
            with torch.no_grad(): # very very very very important!!!
                test_loss = 0.0
                for j, data in enumerate(testloader):
                    [MRF_batch,B0_batch,B1_batch]=data
                    MRF_batch = MRF_batch.to(device)
                    B1_batch = B1_batch.to(device)
                    B0_batch = B0_batch.to(device)
                    
                    MRF_batch[:,:,1]=torch.mul(MRF_batch[:,:,1],B1_batch)
                    MRF_batch[:,:,2]=torch.add(MRF_batch[:,:,2],B0_batch)


                    idxs=list(range(j*args.batch,(j+1)*args.batch))
                    pred = rnn(MRF_batch)
                    MR_images_test[idxs,:] = pred
                
        print('Epoch:', '%04d' % (epoch + 1),'Time. taken =', '{:.4f}'.format(time.time()-start_time))

        quantification_result=MR_images_test        
        quantification_result=quantification_result.cpu()
        quantification_result=quantification_result.numpy()


        sio.savemat(args.result+"/Phantom_"+Tisseu_param[i]+"_rnn_mtc_LOAS40_with_B0_1p2_rB1_0p5_corrected.mat",{'result_nn': quantification_result})
