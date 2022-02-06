# Overcoming Forgetting in Federated Learning via Importance From Agent (AML-XAI-Project)

## Federated-Learning (PyTorch)
This implementation is highly borrowed from https://github.com/AshwinRJ/Federated-Learning-PyTorch
We conducted this research for the final project of 'Adaptive Machine Learning and Explainable AI' class in Seoul National University by Prof. Taesup Moon.


Experiments are produced on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.
Since the purpose of these experiments are to illustrate the effectiveness of the federated learning paradigm, only simple models such as MLP and CNN are used.

## Requirments
Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the baseline experiment with MNIST on MLP using CPU:
```
python src/baseline_main.py --model=mlp --dataset=mnist --epochs=10
```
* Or to run it on GPU (eg: if gpu:0 is available):
```
python src/baseline_main.py --model=mlp --dataset=mnist --gpu=0 --epochs=10
```
-----

Federated experiment involves training a global model using many local models.

* To run the federated experiment with CIFAR on CNN (IID):
```
python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10
```
-----

(Ours) Improving Federated Learing by considering importance from agent.

* Before running, please move to ewc branch 
```
git checkout -t origin/ewc
```

* To run the federated experiment with CIFAR on CNN (IID):
```
python src/federated_main.py --model=cnn --dataset=cifar --gpu=0 --iid=1 --epochs=10 --fisher_update_type gamma --gamma 0.9
```


You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'fmnist', 'cifar'
* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'
* ```--gpu:```      Default: None (runs on CPU). Can also be set to the specific gpu id.
* ```--epochs:```   Number of rounds of training.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--seed:```     Random Seed. Default set to 1.

#### Federated Parameters
* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.
* ```--num_users:```Number of users. Default is 100.
* ```--frac:```     Fraction of users to be used for federated updates. Default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. Default is 10.
* ```--local_bs:``` Batch size of local updates in each user. Default is 10.
* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.

#### EWC Parameters (Updated)
* ```--global_update:```      Method for aggregating weight. Default is avg
* ```--local_update:```Method for local update. Default is base.
* ```--ewc_lambda:```     Weight for ewc loss. Default is 0.001.
* ```--gamma:``` The weight for remembering the previous fisher matrix. 0: not considering previous fisher info, [0-1].
* ```--fisher_update_type:``` How to aggregate fisher matrix to the weight. Select from 'gamma', 'summation', 'own'
* ```--isher_bs:```  Batch for fisher iterator. Default is 100.


## Experiments results

```Table 1:``` Test accuracy on MNIST:

| Model | MLP | CNN | 
| ----- | -----    | -----    | 
|  FedAvg   |  85.70%  |  93.62%  |  
|  FedProx  |  85.57%  |  93.67%  | 
|  Ours  |  89.19%  |  95.08%  | 

```Table 2:``` Test accuracy on CIFAR-10:

| Model | MLP | CNN | 
| ----- | -----    | -----    | 
|  FedAvg   |  39.64%  |  41.90%  |  
|  FedProx  |  40.78%  |  42.05%  | 
|  Ours  |  39.89%  |  42.80%  | 


