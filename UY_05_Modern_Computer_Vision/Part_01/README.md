```bash
$ conda create -n yolov8 python==3.9
$ conda activate yolov8
$ pip install ultralytics==8.0
$ yolo task=detect mode=predict model=yolov8n.pt source=image.jpg
$ yolo task=detect mode=predict model=yolov8n.pt source=video.mp4
$ yolo task=detect mode=predict model=yolov8n.pt source=image.jpg conf=0.95
$ yolo task=detect mode=predict model=yolov8n.pt source=image.jpg conf=0.8 save_txt=True
$ yolo task=detect mode=predict model=yolov8n.pt source=image.jpg conf=0.8 save_txt=True save_crop=True
```
All of this is used to detect. Now we're going to segment.
```bash
$ yolo task=segment mode=predict model=yolov8n-seg.pt source=image.jpg
```