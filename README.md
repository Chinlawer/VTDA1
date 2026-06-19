# VTDA
This repo is the official implementation of paper "Visual and Temporal Dual Adaptation for 3D Few-Shot Class-Incremental Learning"

## Requirements

### Installation

PyTorch, PyTorch3d, CLIP, pointnet2_ops, etc., are required. We recommend to create a conda environment and install dependencies in Linux as follows:

```
# create a conda environment
conda create -n vtda python=3.8 -y
conda activate vtda

# install pytorch & pytorch3d
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# install CLIP
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# install pytorch-loss
pip install "git+https://https://github.com/CoinCheung/pytorch-loss.git"
cd pytorch_loss
python setup.py build
python setup.py install

# install pointnet2 & other packages
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install -r requirements.txt
pip install torchnet
pip install torchmetrics
conda install pyparsing
pip install pytorch3D
```

### Data preparation

The overall directory structure should be:

```
│VTDA/
├──data/
│   ├──ModelNet40_Align/
│   ├──ModelNet40_Ply/
│   ├──Rendering/
│   ├──ShapeNet55/
│   ......
├──.......
```

Please refer to [CLIP2Point](https://github.com/tyhuang0428/CLIP2Point) for the dataset download.

## Get start

download the pre-trained checkpoint [best_eval.pth](https://drive.google.com/file/d/1ZAnIANNMqRRRmaVtk8Kp93s_NkGU51zv/view?usp=sharing)  [best_test.pth](https://drive.google.com/file/d/1Jr1yXOu1yKmMs8K7XD8FnttPRHnZOZHx/view?usp=sharing) and  [dgcnn_occo_cls.pth](https://drive.google.com/file/d/1EG7zh8J_IE4rN9aNb_z7ePkAIwD9SwfB/view?usp=drive_link)

```
│VTDA/
├──pre_builts/
│   ├──vit32/
│   |	├──best_eval.pth/
│   |	├──best_test.pth/
│   ├──point/
│   |	├──dgcnn_occo_cls.pth/
```

```
python main.py
```

You can change session_settings.py and args to run in other datasets.

## Acknowledgement
Our codes are built on [FILP-3D](https://github.com/HIT-leaderone/FILP-3D)



