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

[WORKS] python train.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 48 --data data/dnca-v2-data_0612.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name yolov7obb-defParams --hyp data/hyp.scratch.p5.yaml

python train.py --workers 8 --device 0 --sync-bn --batch-size 4 --data data/.yaml  --img 640 640 --cfg cfg/training/yolov7.yaml --weights weights/yolov7.pt --name yolov7obb-defParams --hyp data/hyp.scratch.p5.yaml
```

## Detect
This is derived from yolov5obb and needs modification.

```
python detect.py --weights 'weights/yolov7_obb_30k_images_60epochs.pt' \
  --source 'data/resize.png' \
  --img-size 640 --device 0 --conf-thres 0.5
```