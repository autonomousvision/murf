<p align="center">
  <h1 align="center">MuRF: Multi-Baseline Radiance Fields</h1>
  <p align="center">
    <a href="https://haofeixu.github.io/">Haofei Xu</a>
    ·
    <a href="https://apchenstu.github.io/">Anpei Chen</a>
    ·
    <a href="https://donydchen.github.io/">Yuedong Chen</a>
    ·
    <a href="https://people.ee.ethz.ch/~csakarid/">Christos Sakaridis</a>
    ·
    <a href="https://yulunzhang.com/">Yulun Zhang</a> <br>
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a>
    ·
    <a href="http://www.cvlibs.net/">Andreas Geiger</a>
    ·
    <a href="https://www.yf.io/">Fisher Yu</a>
  </p>
  <h3 align="center">CVPR 2024</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2312.04565">Paper</a> | <a href="https://haofeixu.github.io/murf/">Project Page</a> </h3>
  <div align="center"></div>
</p>

<div align="center">
  <video src="https://github.com/autonomousvision/murf/assets/19343475/462c9784-aba8-4ceb-bda0-5f9cbcceba6d"/>
</div>

<p align="center">
MuRF supports multiple different baseline settings.
</p>
<p align="center">
  <a href="">
    <img src="https://haofeixu.github.io/murf/assets/dtu_realestate_llff_bar_comparison.png" alt="Logo" width="100%">
  </a>
</p>

<p align="center">
MuRF achieves state-of-the-art performance under various evaluation settings.
</p>

## Installation

Our code is developed based on pytorch 1.10.1, CUDA 11.3 and python 3.8. 

We recommend using [conda](https://www.anaconda.com/distribution/) for installation:

```
conda create -n murf python=3.8
conda activate murf
pip install -r requirements.txt
```



## Model Zoo

The models are hosted on Hugging Face 🤗 : https://huggingface.co/haofeixu/murf

Model details can be found at [MODEL_ZOO.md](MODEL_ZOO.md).



## Datasets

The datasets used to train and evaluate our models are detailed in [DATASETS.md](DATASETS.md)



## Evaluation

The evaluation scripts used to reproduce the numbers in our paper are detailed in [scripts/*_evaluate.sh](scripts).



## Rendering

The rendering scripts are detailed in [scripts/*_render.sh](scripts).


## Training

The training scripts are detailed in [scripts/*_train.sh](scripts).



## Citation

```
@inproceedings{xu2024murf,
      title={MuRF: Multi-Baseline Radiance Fields},
      author={Xu, Haofei and Chen, Anpei and Chen, Yuedong and Sakaridis, Christos and Zhang, Yulun and Pollefeys, Marc and Geiger, Andreas and Yu, Fisher},
      booktitle={CVPR},
      year={2024}
    }
```



## Acknowledgements

This repo is heavily based on [MatchNeRF](https://github.com/donydchen/matchnerf), thanks [Yuedong Chen](https://donydchen.github.io/) for this fantastic work. This project also borrows code from several other repos: [GMFlow](https://github.com/haofeixu/gmflow), [UniMatch](https://github.com/autonomousvision/unimatch), [latent-diffusion](https://github.com/CompVis/latent-diffusion), [MVSNeRF](https://github.com/apchenstu/mvsnerf), [IBRNet](https://github.com/googleinterns/IBRNet), [ENeRF](https://github.com/zju3dv/ENeRF) and [cross_attention_renderer](https://github.com/yilundu/cross_attention_renderer). We thank the original authors for their excellent work.







