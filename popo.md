# Env

machine 227 (Driver 510.47.03, CUDA 11.6)
docker image: `yolov7obb:v2`. Find the image on dockerhub: 
sha256:8237b3c637c0dc5315a3e7f73c5e3f9738cfb95e741ba7fc77c5721949e1bfe5

Locally, on 227, find it as `v7obb:v2`
Our container is based on `nvcr.io/nvidia/pytorch:22.02-py3` and has god-knows-what-all dependencies added. 

Use the docker-compose file to launch the container. 

## Train

```
[DOESN'T WORK] python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 80 --data data/dnca-v2-data_0612.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name yolov7obb --hyp data/hyp.scratch.p5.yaml

[WORKS] python train.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 48 --data data/dnca-v2-data_0612.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name yolov7obb-defparams --hyp data/hyp.scratch.p5.yaml

python train.py --workers 8 --device 0 --sync-bn --batch-size 4 --data data/.yaml  --img 640 640 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name yolov7obb-defParams --hyp data/hyp.scratch.p5.yaml
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
```