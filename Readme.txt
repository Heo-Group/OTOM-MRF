OTOM network consists of 3 layer LSTM with 512 hidden states. 
OTOM takes an input vector of MTC signal and the corresponding scan parameters: [Smtc, saturatin power (B1), frequency offset (Omega), saturation time (Ts), delay time (Td)].
The hidden state updated with all input vectors from MRF is used to estimate quantitative water and MTC parameters. 


# Hyperparameters are defiend in the Train.py with parser. 

python Train.py --gpu 0 --batch 256 --epochs 20


# Test code is to quantify tissue parameters with simulated MTC-MRF signals from four digital phantoms.
# Each digital phantom encodes single tissue parameter and B0/B1 inhomogeneities are included.
# Ground truth (GT) B0 and rB1 values are used to correct the scan parameters (MRF schedule).

python Test.py --gpu 0

