# CS460-Final-Project
## Requirements
* Pycharm, Microsoft Visual Studio Code 
* Pytorch 2.9.0 + CUDA 12.8.0
* numpy, opencv-python, matplotlib, tqdm, ultralytics
## Pretrained models
* YOLOv8
* Faster R-CNN
## Preparation for testing
`` python -m bdd_tl_benchmark.main `  
``
``
  --videos_dir ".\project\training\datasets\bdd100k\videos" `
``
``
  --ann_dir ".\project\training\datasets\bdd100k\video_labels" `
``
``
  --out_dir ".\project\training\exp_out" ` 
``
``
  --device cuda:0 `
``
``
  --max_videos 2000 `
``
``
  --frame_stride 2
``
* Our datasets is Berkeley Deep Drive: http://bdd-data.berkeley.edu/download.html 
