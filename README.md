# OTOM-MRF

## Only-Train-Once MR Fingerprinting (OTOM) for Magnetization Transfer Contrast Quantification
OTOM network consists of 3 layer LSTM with 512 hidden states. 
OTOM takes an input vector of MTC signal and the corresponding scan parameters: [Smtc, saturatin power (B1), frequency offset (Omega), saturation time (Ts), delay time (Td)].
The hidden state updated with all input vectors from MRF is used to estimate quantitative water and MTC parameters. 
(publicaiton) <https://doi.org/10.1002/mrm.29629> <https://link.springer.com/chapter/10.1007/978-3-031-16446-0_37>


### Template command

Hyperparameters are defiend in the Train.py with parser.
```
python Train.py --gpu 0 --batch 256 --epochs 20
```

Test code is to quantify tissue parameters with simulated MTC-MRF signals from four digital phantoms. Each digital phantom encodes single tissue parameter and B0/B1 inhomogeneities are included.
Ground truth (GT) B0 and rB1 values are used to correct the scan parameters (MRF schedule).

```
python Test.py --gpu 0
```



![Code_Figure1](https://user-images.githubusercontent.com/122308855/211401238-bb6feb64-5683-43e1-a263-801768b31452.png)

![Code_Figure2](https://user-images.githubusercontent.com/122308855/211401252-52052e00-9a5c-4bb9-a5e0-7961c2dc1f78.PNG)

