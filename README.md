# UNO_Card_Detection

```
source cv3_env/bin/activate
```

```
python3 ./tools/train.py ./configs/yolox/yolox_s_8xb8-300e_coco_UNO.py
```

```
tensorboard --logdir ./
```
```
python cocosplit.py --having-annotations --multi-class -s 0.8 ../UNO_Card_Detection/annotations/UNO_dataset.json train.json test.json
```