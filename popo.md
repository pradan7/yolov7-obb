# Env

machine 227 (Driver 510.47.03, CUDA 11.6)
docker image: `yolov7obb:v2`. Find the image on dockerhub: 
sha256:8237b3c637c0dc5315a3e7f73c5e3f9738cfb95e741ba7fc77c5721949e1bfe5

Locally, on 227, find it as `v7obb:v2`
Our container is based on `nvcr.io/nvidia/pytorch:22.02-py3` and has god-knows-what-all dependencies added. 

Use the docker-compose file to launch the container. 

## Train
To resume training, add `--resume runs/exp4/weights/best.pt` flag to below commands right after `train.py`. The best part is, that the training resumes from the same epoch where it crashed and everything is logged to the same save directory as before. It is important to give the weight from which training is to be resumed.

```
[DOESN'T WORK] python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 80 --data data/dnca-v2-data_0612.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name yolov7obb --hyp data/hyp.scratch.p5.yaml

[WORKS] python train.py --workers 8 --device 1,2,3,4,5,6,7 --sync-bn --batch-size 42 --data data/dnca-v2-data_0612.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name exp --hyp data/hyp.scratch.p5.yaml

python train.py --workers 8 --device 1,2,3,4,5,6,7 --sync-bn --batch-size 42 --data data/dnca-v2-data_0612.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name exp --hyp data/hyp.scratch.p5.v5obb-exp65-70-hyp.yaml

[scratch board] python3 train.py --workers 8 --device 1,2,3,4,5,6,7 --sync-bn --batch-size 21 --epochs 150 --data data/smol-data.yaml --img 768 768 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name exp --hyp data/hyp.more-copypaste-more-mixup.yaml --multi-scale

[Another try with ddp] python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 72 --data data/dnca-v2-data_0612.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name exp --hyp data/hyp.scratch.p5.yaml
It fails after couple of hours.... Leave it man!

nohup python train.py --workers 8 --device 1,2,3,4,5,6,7 --sync-bn --batch-size 21 --epochs 300 --data data/DOTAv1.yaml --img 1024 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name dotav1 --hyp data/hyp.finetune_dota.yaml > 227.out 2>&1 &
```

## Detect
This is derived from yolov5obb and needs modification.

```
python detect.py --weights 'weights/yolov7_obb_30k_images_60epochs.pt' \
  --source 'data/resize.png' \
  --img-size 640 --device 0 --conf-thres 0.5
```

# Evaluate
The holdout set has been prepared out of batch 6 to 10 in `/algo/users/prashant/GoodsDetector/annotations/dnca_v2_holdout_set`. 

Before running evaluation/training on new data, make sure to follow these steps to avoid `corrupt/image label...` error.
- delete the existing train and val cache  The cache is stored in the directory where the split data is stored. For e.g. `/algo/users/prashant/GoodsDetector/annotations/dnca_v2_holdout_set/split_data/dnca_v2_data/labelTxt/train.cache` i.e the current holdout set directory.
- Run the train.py script (see command above) for this dataset such that it creates new cache files there.
- Now run the val(popo) script for the new dataset. 
Note that for all these steps, the dataset is denoted by its yaml.

The default evaluation script has been modified to run evaluation for different imagesizes (only square allowed for now) and for all weights saved during all training experiments.
```
python3 val.py --weights runs/train/yolov7obb-defparams3/weights/best.pt  --data data/dnca-v2-data_0612.yaml --device 1,2,3,4,5,6,7 --batch-size 21

python3 val.py --weights weights/yolov7_obb_30k_images_60epochs.pt  --data data/dnca-v2-data_0612.yaml --device 1,2,3,4,5,6,7 --batch-size 21

python3 val.py --weights weights/yolov7_obb_30k_images_60epochs.pt  --data data/dnca-v2-holdout.yaml --device 1,2,3,4,5,6,7 --batch-size 21

python3 val.py --weights weights/yolov7_obb_30k_images_60epochs.pt  --data data/smol-data.yaml --device 1,2,3,4,5,6,7 --batch-size 21 --multi-scale
```


## Export
```
python3 export.py --weights runs/train/exp4/weights/best.pt --dynamic --grid --img-size 768
# rename to exp4-768x768-dynamic.onnx etc
```

## Benchmark (DOTA v1)
```
# For mAP50
python val.py --data 'data/DOTAv1.yaml' --img 1024 --batch 1 
```
Note that before val, we will need to train the model on this dataset. I did it via 
```
python train.py --workers 8 --device 1,2,3,4,5,6,7 --sync-bn --batch-size 21 --epochs 300 --data data/DOTAv1.yaml --img 1024 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name dotav1 --hyp data/hyp.finetune_dota.yaml
```

## GPU Capacity
This will help decide the appropriate batch size depending on the number of GPUs to be used.

### 227 (2080Ti) All vals are per GPU
Imgsz 640
Train with DDP: 9
Train without DDP: 6 | Blob = 640*640*3*6
Val without DDP: 7

# Runs
1: default hyp
4: hyp same as exp65
16: exp 4 + more cutpaste, more mixup
19: exp 16 with `--multi-scale`
21: exp 4 (with correct charts)
23: the og v7obb model (`/algo/users/prashant/GoodsDetector/YOLOv7-OBB/weights/yolov7_obb_30k_images_60epochs.torchscript.pt`) used as pretrained model with training cfg same as exp16.
