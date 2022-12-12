# Pix2Vox - Tensorflow
This respository contains TensorFlow version of the code for the paper [Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images](https://arxiv.org/abs/1901.11153). The code is based of the original code which is foun [here](https://github.com/hzxie/Pix2Vox). 

## Datasets
We used the [ShapeNet](https://www.shapenet.org/), which is available below:
- [ShapeNet Rendering Images](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)
- [ShapeNet Voxelized Models](http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz)

## Installation
We used [Anaconda](https://www.anaconda.com/products/individual) to set up the environment. To set up the environment, run the following commands:
```
# Clone the repo
git clone https://github.com/xing-angela/dl-final-pix2vox.git
cd mipnerf

# Create the conda environment using tensorflow 2.4.1
conda create --name pix2vox_env tensorflow=2.4.1
conda activate pix2vox_env

# Prepare pip
conda install pip
pip install --upgrade pip

# Install the requirements
pip install -r requirements.txt
```

## Running the Code
To train Pix2Vox, you can run the following command:
```
python runner.py
```

You can add the following flags to the command:
```
--batch-size  # specifies the batch size
--epoch       # specifies the number of epochs
--out         # specifies the output directory
```


