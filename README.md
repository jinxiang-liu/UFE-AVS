# CVPR2024: Audio-Visual Segmentation via Unlabeled Frame Exploitation

Official implementation of  [Audio-Visual Segmentation via Unlabeled Frame Exploitation](https://arxiv.org/abs/2403.11074).

This paper has been accepted by **CVPR 2024**, the project page is [https://jinxiang-liu.github.io/UFE-AVS/](https://jinxiang-liu.github.io/UFE-AVS/).

![](assets/teaser.png)

***********

## Get Started

### Implementation
Our proposed method is versatile for all the existing AVS methods theoretically; in this work we implement the method on top of two representative AVS methods including [TPAVI](https://github.com/OpenNLPLab/AVSBench) and [AVSegFormer](https://github.com/vvvb-github/AVSegFormer). 
Therefore, our implementation is also based on the authors' released codes and we express our thanks to them.
Please check their original repositries for more environment configuration details.

To perform training or inference of our method on top of either of the two baselines, please enter the corresponding directory and run the scripts.



### Data & Model Weights
About the data, inlcuding the flows from AVSBench and extended flows, please check the `PREPARATION.md` to download.

About the model weights, we release all weights based on TPAVI and AVSegFormer on both subsets of AVSBench, please check the `PREPARATION.md` for downloading links.



***********
## Citation
```txt
@InProceedings{Liu_2024_CVPR,
    author    = {Liu, Jinxiang and Liu, Yikun and Zhang, Fei and Ju, Chen and Zhang, Ya and Wang, Yanfeng},
    title     = {Audio-Visual Segmentation via Unlabeled Frame Exploitation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {26328-26339}
}
```
