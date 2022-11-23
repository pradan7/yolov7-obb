pip install wandb

wandb login 38621b8ae96ab63c8b6303586b1edab86d0635df

cd /yolov7

python3 train.py --weights yolov7.pt --cfg cfg/training/yolov7.yaml --data dataset.yaml --epochs 200 --nosave --notest --device 0,1,2,3,4 --save_period 25