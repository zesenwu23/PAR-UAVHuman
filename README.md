## Dataset
We use [UAVHuman](https://github.com/sutdcv/UAV-Human) (attribute subset) to train and evaluate.


#### Settings:

1. Python version: `3.8`

2. The torch (torchvision) could be downloaded via:
   
   ```
   pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
   ```

3. Other related packages are available at `requirements.txt`


#### Testing:

1. Set the root path to your test dataset. (`(line 30` at `test.py`)
   
   ```
   root = '/dataset/uav-attr/uavhuman' (Please change the path with yours)
   ```

2. Shange the model path, saved in the log path. (`(line 78` at `test.py`)  
   
   ```
   model_path = '/datalog/attribute-log/model_best.pth'
   ```

3. Start testing:
   
   ```
   CUDA_VISIBLE_DEVICES=0 python test.py
   ```
   
   Important: You can also load [our model](https://drive.google.com/drive/folders/1ZkSKgEyIr4Vj-OKNIplC2_CxERNQUbPV?usp=sharing) to get the reported result. 

#### Before training:

We use the Imagenet-pretrained Convnext2 as Backbone.
The corresponding loading code is at `multi_convnextv2.py (line 113)`, which is downloaded from the [official repository](https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt).

#### Training：

1. Set the root path to your own dataset path, and the corresponding code is at `train.py (line 108, line 135)`.
   
   ```
   root = '/dataset/uav-attr/uavhuman' (Please change the path with yours)
   ```

2. Set the log path and model path, and the corresponding code is at `train.py (line 207` 
   
   ```
   log_path = '/datalog/attribute-log/' (log and model are saved in this path)
   ```

3. start training
   
   ```
   CUDA_VISIBLE_DEVICES=0,1 python train.py (Two GPUs are needed)
   ```
   
   Our trained model is at: [Google Drive](https://drive.google.com/drive/folders/1ZkSKgEyIr4Vj-OKNIplC2_CxERNQUbPV?usp=sharing).
