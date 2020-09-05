# Learning Data Manipulation

This repo contains preliminary code of the following paper:

[Learning Data Manipulation for Augmentation and Weighting](http://www.cs.cmu.edu/~zhitingh/data/neurips19_data_manip_preprint.pdf)  
Zhiting Hu*, Bowen Tan*, Ruslan Salakhutdinov, Tom Mitchell, Eric P. Xing  
NeurIPS 2019 (equal contribution)

## Requirements

- `python3.6`
- `pytorch==1.0.1`
- `pytorch_pretrained_bert==0.6.1`
- `torchvision==0.2.2`

## Code
* ```baseline_main.py```: Vanilla BERT Classifier.
* ```ren_main.py```: Described in [(Ren et al.)](https://arxiv.org/pdf/1803.09050.pdf).
* ```weighting_main.py```: Our weighting algorithm.
* ```augmentation_main.py```: Our augmentation algorithm.


## Running
Running scripts for experiments are available in [scripts/](scripts/).

## Results

All the detailed training logs are availble in [results/](results/).

*(Note: The result numbers may be slightly different from those in the paper due to slightly different implementation details and random seeds, while the improvements over comparison methods are consistent.)*

### low data

##### SST-5
|Base Model: BERT|Ren et al.| Weighting  | Augmentation |
|:-:|:-:|:-:|:-:|
| 33.32 ± 4.04 | 36.09 ± 2.26 | 36.51 ± 2.54   | 37.55 ± 2.63 |

##### CIFAR-10
|                  |  Pretrained    | Not Pretrained |
|------------------|----------------|----------------|
|Base Model: ResNet| 34.58 ± 4.13   | 24.68 ± 3.29   |
| Ren et al.       | 23.29 ± 5.95   | 22.26 ± 2.80   |
| Weighting        | 36.75 ± 3.09   | 26.47 ± 1.69   |


### imbalanced data

##### SST-2
|| 20 : 1000 | 50 : 1000　| 100 : 1000
|:-:|:-:|:-:|:-:|
|Base Model: BERT| 54.91 ± 5.98 | 67.73 ± 9.20 | 75.04 ± 4.51 |
|Ren et al.| 74.61 ± 3.54 | 76.89 ± 5.07 | 80.73 ± 2.19 | 
|Weighting| 75.08 ± 4.98 | 79.35 ± 2.59 | 81.82 ± 1.88 | 

##### CIFAR-10
|                  | 20 : 1000    | 50 : 1000    | 100 : 1000   |
|------------------|--------------|--------------|--------------|
|Base Model: ResNet| 70.65 ± 4.98 | 79.52 ± 4.81 | 86.12 ± 3.37 |
| Ren et al.       | 76.68 ± 5.35 | 77.34 ± 7.38 | 78.57 ± 5.61 |
| Weighting        | 79.07 ± 5.02 | 82.65 ± 5.13 | 87.63 ± 3.72 |
