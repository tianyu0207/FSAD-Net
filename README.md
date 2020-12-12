# FSAD-Net
Few-Shot Anomaly Detection for Polyp Frames from Colonoscopy (MICCAI 2020)
https://arxiv.org/abs/2006.14811


Training step: 
1. train a deep infomax encoder using only normal data. we recommand this repository to train: https://github.com/rdevon/DIM
2. place your own data into data/  folder and start to train the FSAD network (Classifier) using the pretrained encoder from step-1 as the feature extractor. 

You may need to write your own dataloader. 

Please consider to cite our paper. 


@inproceedings{tian2020few,
  title={Few-Shot Anomaly Detection for Polyp Frames from Colonoscopy},
  author={Tian, Yu and Maicas, Gabriel and Pu, Leonardo Zorron Cheng Tao and Singh, Rajvinder and Verjans, Johan W and Carneiro, Gustavo},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={274--284},
  year={2020},
  organization={Springer}
}


