# FSAD-Net
This repo contains the Pytorch implementation of our paper: 
> [**Few-Shot Anomaly Detection for Polyp Frames from Colonoscopy**](https://arxiv.org/abs/2006.14811)
>
> [Yu Tian](https://yutianyt.com/), [Gabriel Maicas](https://cs.adelaide.edu.au/~gabriel/), Leonardo Zorron Cheng Tao Pu, Rajvinder Singh, Johan W Verjans, [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/).


- **Accepted at MICCAI 2020.**  



## Training
1. pretrain a [Deep Infomax](https://github.com/rdevon/DIM) encoder using only normal data.  
2. Place your own data into data/  folder and start to train the FSAD network (Classifier) using the pretrained encoder as the feature extractor. 

Run:
```shell
python main.py 
```


## Citation

If you find this repo useful for your research, please consider citing our paper:
```bibtex
@inproceedings{tian2020few,
  title={Few-Shot Anomaly Detection for Polyp Frames from Colonoscopy},
  author={Tian, Yu and Maicas, Gabriel and Pu, Leonardo Zorron Cheng Tao and Singh, Rajvinder and Verjans, Johan W and Carneiro, Gustavo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={274--284},
  year={2020},
  organization={Springer}
}
```







