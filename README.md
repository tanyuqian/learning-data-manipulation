# Augmentation-Weighting

## Requirements

- `python3.6`
- `pytorch==1.0.1`
- `pytorch_pretrained_bert==0.6.1`

## Code
* ```baseline_main.py```: Vanilla BERT Classifier.
* ```ren_main.py```: Described in [(Ren et al.)](https://arxiv.org/pdf/1803.09050.pdf).
* ```weighting_main.py```: Our weighting algorithm.
* ```augmentation_main.py```: Our augmentation algorithm.


## Running
Running scripts for experiments are available in [scripts/](scripts/).

## Results

All the detailed training logs are availble in [results/](results/).

### low data (sst-5)

|Base Model: BERT|Ren et al.| Weighting  | Augmentation |
|:-:|:-:|:-:|:-:|
| 33.32 ± 4.04 | 36.09 ± 2.26 | 36.51 ± 2.54   | 37.55 ± 2.63 |

### imbalanced data (sst-2)
|| 20 : 1000 | 50 : 1000　| 100 : 1000
|:-:|:-:|:-:|:-:|
|Base Model: BERT| 54.91 ± 5.98 | 67.73 ± 9.20 | 75.04 ± 4.51 |
|Ren et al.| 74.61 ± 3.54 | 76.89 ± 5.07 | 80.73 ± 2.19 | 
|Weighting| 75.08 ± 4.98 | 79.35 ± 2.59 | 81.82 ± 1.88 | 
