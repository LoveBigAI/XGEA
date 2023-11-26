# The source code for ACM MM 2023 accepted paper Cross-Modal Graph Attention Network for Entity Alignment (XGEA) 

## Dataset
Both the  Bilingual datasets (DBP15K) and cross-KG datasets (FB15K-DB15K/YAGO15K) are come from  the  [MCLEA](https://github.com/lzxlin/MCLEA) repository. You can download them directly in their pages.

## Environment

* Python = 3.8.5
* Keras = 2.3.1
* Tensorflow = 2.2.0
* Pytorh = 1.11.0
* Scipy
* Numpy
* tqdm
* numba
Our XGEA is trained an evaluated both on A100 and A40.

## Training MCLEA
For DBP15K, use the script:
```bash
python DualAmodal.py
```
while for FB15K-DB15K/YAGO15K, there is a slightly different configuration, we will release it soon...
## Acknowledgement

Our codes are  based on [EVA](https://github.com/cambridgeltl/eva), [Dual-AMN](https://github.com/MaoXinn/Dual-AMN), [ContEA](https://github.com/nju-websoft/ContEA), really appreciate their work.
If you have any questions, please feel free to contact me bgxulive@gmail.com.
