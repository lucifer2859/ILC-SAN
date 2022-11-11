# Fully Spiking Actor Network with Intra-layer Connections

This package is the PyTorch implementation of Fully Spiking Actor Network with Intra-layer
Connections (ILC-SAN) that integrates with TD3 algorithms.

## Software Installation ##

* Ubuntu 16.04
* Python 3.6.13
* MuJoCo 2.1
* OpenAI GYM 0.18.0 (with mujoco_py 2.1.2.14)
* PyTorch 1.7.1 (with CUDA 10.2, CUDNN 7.6.5 and tensorboard 2.6.0)

A CUDA enabled GPU is not required but preferred for training.  
The results in the paper are generated from models trained using NVIDIA Tesla V100.

## Example Usage ##
To enter the folder where the code is located, execute the following commands:
```
cd <Dir>/<Project Name>/ilcsan-td3
```

#### 1. Training ILC-SAN ####

To train ILC-SAN using TD3 algorithm, execute the following commands:

```
python hybrid_td3_cuda_norm.py --env HalfCheetah-v3
```

or

```
python hybrid_td3_cuda_norm.py --env HalfCheetah-v3 
							   --encode      [pop-det|pop|layer]
                               --decode      [last-mem|max-mem|mean-mem|fr-mlp]
							   --neurons     [LIF|DN]
                               --connections [intra|no-bias|no-self|no-lateral|bias|self|lateral|none]
```

This will automatically train 1 million steps and save the trained models. All items in brackets are optional, and the first item is the default.

#### 2. Training PopSAN ####

To train PopSAN using TD3 algorithm, execute the following commands:

```
python hybrid_td3_cuda_norm.py --env HalfCheetah-v3 --encode pop-det --decode fr-mlp --connections none
```

#### 3. Training MDC-SAN ####

To train MDC-SAN using TD3 algorithm, execute the following commands:

```
python hybrid_td3_cuda_norm.py --env HalfCheetah-v3 --encode pop --decode fr-mlp --neurons DN --connections none
```

#### 4. Test ILC-SAN/PopSAN/MDC-SAN ####

To test the TD3-trained ILC-SAN/PopSAN/MDC-SAN, execute the following commands:

```
python test_hybrid_td3_cpu.py --env HalfCheetah-v3 
							  --encode      [pop-det|pop|layer]
                              --decode      [last-mem|max-mem|mean-mem|fr-mlp]
							  --neurons     [LIF|DN]
                              --connections [intra|no-bias|no-self|no-lateral|bias|self|lateral|none]
```