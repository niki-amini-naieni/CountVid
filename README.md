# [AAAI 2026] CountVid: Open-World Object Counting in Videos

Niki Amini-Naieni & Andrew Zisserman

Official PyTorch implementation for CountVid. Details can be found in the paper, [[Paper]](http://arxiv.org/abs/2506.15368) [[Project page]](https://www.robots.ox.ac.uk/~vgg/research/countvid/).

If you find this repository useful, please give it a star ⭐.

<img src=img/teaser.jpg width="100%"/>
<img src=img/countvid-results.gif width="100%"/>

## CountVid

<img src=img/method.jpg width="100%"/>

## Contents
* [Demo](#demo)
* [Dataset Download](#dataset-download)
* [Reproduce Results From Paper](#reproduce-results-from-paper)
* [Training CountGD-Box](#training-countgd-box)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

## Demo

### 1. Clone Repository

```
git clone git@github.com:niki-amini-naieni/CountVid.git
```

### 2. Download Video Frames

* Make the ```demo``` directory inside the ```CountVid``` repository.

  ```
  cd CountVid
  mkdir demo
  ```

* Download the video frames for the demo [here](https://drive.google.com/drive/folders/1v4RNNBHYEQQ82NF8fNiRPhIdQ96-7xCs?usp=sharing), and place them into the ```demo``` directory, so your file tree looks like:

  ```
  CountVid/
    |demo/
      |00001.jpg
      ...
      |00094.jpg
    ...
  ```

### 3.a. Install GCC 

Install GCC. In this project, GCC 11.3 and 11.4 were tested. The following command installs GCC and other development libraries and tools required for compiling software in Ubuntu.

```
sudo apt update
sudo apt install build-essential
sudo apt install gcc-11 g++-11
```

### 3.b. Install CUDA Toolkit:

NOTE: In order to install detectron2 in step 4, you needed to tnstall CUDA Toolkit. Refer to: https://developer.nvidia.com/cuda-downloads

### 4. Set Up Anaconda Environment:

The following commands will create a suitable Anaconda environment for running CountVid and training CountGD-Box. To produce the results in the paper, we used [Anaconda version 2024.02-1](https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh).

```
conda create -n countvid python=3.10
conda activate countvid
conda install -c conda-forge gxx_linux-64 compilers libstdcxx-ng # ensure to install required compilers
cd ..
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd ../CountVid
pip install -r requirements.txt
export CC=/usr/bin/gcc-11 # this ensures that gcc 11 is being used for compilation
cd models/GroundingDINO/ops
python setup.py build install
python test.py # should result in 6 lines of * True
cd ../../../
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### 5. Download Pre-Trained Weights

* Make the ```checkpoints``` directory inside the ```CountVid``` repository.

  ```
  mkdir checkpoints
  ```

* Execute the following command.

  ```
  python download_bert.py
  ```

* Download the pretrained CountGD-Box model available [here](https://drive.google.com/file/d/1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD/view?usp=sharing), and place it in the ```checkpoints``` directory Or use ```gdown``` to download the weight.

  ```
  pip install gdown
  gdown --id 1bw-YIS-Il5efGgUqGVisIZ8ekrhhf_FD -O checkpoints/
  ```

* Download the pretrained  SAM 2.1 weights.

  ```
  wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
  ```

### 6. Run Demo
* Run the following command.
```
python count_in_videos.py --video_dir demo --input_text "penguin" --sam_checkpoint checkpoints/sam2.1_hiera_large.pt --sam_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml --obj_batch_size 30 --img_batch_size 10 --downsample_factor 1 --pretrain_model_path checkpoints/countgd_box.pth --temp_dir ./demo_temp --output_dir ./demo_output --save_final_video --save_countgd_video
```
* Visualize the output.
You should see the following videos saved to the ```demo_output``` folder once the demo has finished running:

  ```final-video.mp4```
  <p align="center">
      <img src="./img/final-video-demo.gif" alt="final output video" width="100%"/>
  </p>

  ```countgd-video.avi```
  <p align="center">
      <img src="./img/countgd-video-demo.gif" alt="timelapse boxes from CountGD-Box" width="100%"/>
  </p>

## Dataset Download

### 1. Download FSCD-147
Download FSCD-147 from [here](https://drive.google.com/file/d/1m_v_hBwXH1NzcuUj_qa-ziKn-LYfUWA6/view?usp=sharing), and update [datasets_fscd147_val.json](config/datasets_fscd147_val.json), and [datasets_fscd147_test.json](config/datasets_fscd147_test.json) to point to the image folder you have downloaded.

### 2. Download VideoCount
Download VideoCount [here](https://drive.google.com/file/d/1elk1-83dstQ0hEPLctrLKUNz4qUkrdDs/view?usp=sharing), and make sure to pass the location of the ```VideoCount``` folder in the input to the ```--data_dir``` command when testing on videos in VideoCount.

The file tree for VideoCount should look like (note: all the files are not visualized for readability purposes):

  ```
  VideoCount/
    |Crystals/
      |anno
        |crystals-count-gt.json
        |crystals-frame-level-counts-gt.json
      |exemplars
        |ma2035_023
        ...
        |ma2035_169
      |frames
        |ma2035_023
        ...
        |ma2035_169
    |MOT20-Count/
      |anno
      |frames
        |MOT20-01
          000001.jpg
          ...
          000429.jpg
        |MOT20-02
        |MOT20-05
    |Penguins/
    |TAO-Count/
      |anno
      |frames
        |val
          |ArgoVerse
          |AVA
          |BDD
          |Charades
          |HACS
          |LaSOT
          |YFCC100M
  ```
Download the TAO validation videos from [here](https://drive.google.com/file/d/1OKj0BgqBeLcdVgvR_5EytYUFraj0PNVI/view?usp=sharing), unzip the folder, and place the unzipped TAO ```val``` folder inside a new ```TAO-Count\frames``` folder in the ```VideoCount``` directory (visualized above). You can also follow the directions at [this](https://motchallenge.net/data/TAO_Challenge/) or [this](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal) link to download the TAO videos if the first link does not work for you.

The files titled ```[benchmark_name]-count-gt.json``` contain the *global count*, the number of unique objects per video for each category. The files titled ```[benchmark_name]-frame-level-counts-gt.json``` contain the *cumulative count at each frame*, the number of unique objects detected so far in the video for each category. The global count for a video is the frame-level count at the last frame.

Note: for the Science-Count (Penguins) dataset, the goal is to count all the seabirds—penguins and shags—that appear in the videos. Because in these videos, the shags look very similar to the penguins and are difficult to distinguish even for human annotators, we used the text prompt "penguin" for all the methods to pick out the birds.

Note: for the Science-Count (Crystals) dataset, ground truth frame-level counts for earlier frames are very accurate. For highly dense later frames, the ground truth frame-level counts can have up to 5% error due to extremely cluttered and overlapping crystals.

## Reproduce Results From Paper

### 1. FSCD-147
* To test the text-only setting, run the following commands.
  
  Validation set:
  ```
  python -u main_fscd147.py --output_dir ./countgd_val -c config/cfg_fscd147_val.py --eval --datasets config/datasets_fscd147_val.json --pretrain_model_path checkpoints/countgd_box.pth --options text_encoder_type=checkpoints/bert-base-uncased --coco_output_file "detections_val_text_only.json" --fscd_gt_file fscd147/instances_val.json --num_exemplars 0
  ```
  ```
  python test_fscd147.py --pred detections_val_text_only.json --gt fscd147/instances_val.json --split "val"
  ```
  Test set:
  ```
  python -u main_fscd147.py --output_dir ./countgd_test -c config/cfg_fscd147_test.py --eval --datasets config/datasets_fscd147_test.json --pretrain_model_path checkpoints/countgd_box.pth --options text_encoder_type=checkpoints/bert-base-uncased --coco_output_file "detections_test_text_only.json" --fscd_gt_file fscd147/instances_test.json --num_exemplars 0
  ```
  ```
  python test_fscd147.py --pred detections_test_text_only.json --gt fscd147/instances_test.json --split "test"
  ```
* To test the exemplar-only setting, run the following commands.
  
  Validation set:
  ```
  python -u main_fscd147.py --output_dir ./countgd_val -c config/cfg_fscd147_val.py --eval --datasets config/datasets_fscd147_val.json --pretrain_model_path checkpoints/countgd_box.pth --options text_encoder_type=checkpoints/bert-base-uncased --coco_output_file "detections_val_exemplars_only.json" --fscd_gt_file fscd147/instances_val.json --no_text
  ```
  ```
  python test_fscd147.py --pred detections_val_exemplars_only.json --gt fscd147/instances_val.json --split "val"
  ```
  Test set:
  ```
  python -u main_fscd147.py --output_dir ./countgd_test -c config/cfg_fscd147_test.py --eval --datasets config/datasets_fscd147_test.json --pretrain_model_path checkpoints/countgd_box.pth --options text_encoder_type=checkpoints/bert-base-uncased --coco_output_file "detections_test_exemplars_only.json" --fscd_gt_file fscd147/instances_test.json --no_text
  ```
  ```
  python test_fscd147.py --pred detections_test_exemplars_only.json --gt fscd147/instances_test.json --split "test"
  ```
  
* To test the multi-modal setting, with both exemplars and text, run the following commands.

  Validation set:
  ```
  python -u main_fscd147.py --output_dir ./countgd_val -c config/cfg_fscd147_val.py --eval --datasets config/datasets_fscd147_val.json --pretrain_model_path checkpoints/countgd_box.pth --options text_encoder_type=checkpoints/bert-base-uncased --coco_output_file "detections_val_text_and_exemplars.json" --fscd_gt_file fscd147/instances_val.json 
  ```
  ```
  python test_fscd147.py --pred detections_val_text_and_exemplars.json --gt fscd147/instances_val.json --split "val"
  ```
  Test set:
  ```
  python -u main_fscd147.py --output_dir ./countgd_test -c config/cfg_fscd147_test.py --eval --datasets config/datasets_fscd147_test.json --pretrain_model_path checkpoints/countgd_box.pth --options text_encoder_type=checkpoints/bert-base-uncased --coco_output_file "detections_test_text_and_exemplars.json" --fscd_gt_file fscd147/instances_test.json --remove_bad_exemplar
  ```
  ```
  python test_fscd147.py --pred detections_test_text_and_exemplars.json --gt fscd147/instances_test.json --split "test"
  ```

### 2. TAO-Count
Run the following commands:
```
nohup python test_tao_count.py --output_file tao-count-text-only-predicted.json --data_dir VideoCount/TAO-Count --checkpoint_dir checkpoints >>./tao-count-text-only-test.log 2>&1 &
```
```
python evaluate_counting_accuracy.py --ground_truth VideoCount/TAO-Count/anno/TAO-count-gt.json --predicted tao-count-text-only-predicted.json --parent_dir VideoCount/TAO-Count/frames
```

### 3. MOT20-Count
Run the following commands:
```
nohup python test_mot20_count.py --output_file mot20-count-text-only-predicted.json --data_dir VideoCount/MOT20-Count --checkpoint_dir checkpoints >>./mot20-count-text-only-test.log 2>&1 &
```
```
python evaluate_counting_accuracy.py --ground_truth VideoCount/MOT20-Count/anno/MOT20-count-gt.json --predicted mot20-count-text-only-predicted.json --parent_dir VideoCount/MOT20-Count/frames
```

### 4. Science-Count (Penguins)
* To test the text-only setting, run the following commands:
```
python test_penguins.py --output_file penguins-count-text-only-predicted.json --data_dir VideoCount/Penguins --checkpoint_dir checkpoints
```
```
python evaluate_counting_accuracy.py --ground_truth VideoCount/Penguins/anno/penguins-count-gt.json --predicted penguins-count-text-only-predicted.json --parent_dir VideoCount/Penguins/frames
```
  
* To test the exemplar-only setting, run the following commands:
```
python test_penguins.py --output_file penguins-count-exemplars-only-predicted.json --data_dir VideoCount/Penguins --checkpoint_dir checkpoints --use_exemplars --no_text
```
```
python evaluate_counting_accuracy.py --ground_truth VideoCount/Penguins/anno/penguins-count-gt.json --predicted penguins-count-exemplars-only-predicted.json --parent_dir VideoCount/Penguins/frames --only_exemplars
```
  
* To test the multi-modal setting, with both exemplars and text, run the following commands:
```
python test_penguins.py --output_file penguins-count-exemplars-and-text-predicted.json --data_dir VideoCount/Penguins --checkpoint_dir checkpoints --use_exemplars
```
```
python evaluate_counting_accuracy.py --ground_truth VideoCount/Penguins/anno/penguins-count-gt.json --predicted penguins-count-exemplars-and-text-predicted.json --parent_dir VideoCount/Penguins/frames
```

### 5. Science-Count (Crystals)
* To test the text-only setting, run the following commands:
```
python test_crystals.py --output_file crystals-count-text-only-predicted.json --data_dir VideoCount/Crystals --checkpoint_dir checkpoints
```
```
python evaluate_counting_accuracy.py --ground_truth VideoCount/Crystals/anno/crystals-count-gt.json --predicted crystals-count-text-only-predicted.json --parent_dir VideoCount/Crystals/frames
```
  
* To test the exemplar-only setting, run the following commands:
```
python test_crystals.py --output_file crystals-count-exemplars-only-predicted.json --data_dir VideoCount/Crystals --checkpoint_dir checkpoints --use_exemplars --no_text
```
```
python evaluate_counting_accuracy.py --ground_truth VideoCount/Crystals/anno/crystals-count-gt.json --predicted crystals-count-exemplars-only-predicted.json --parent_dir VideoCount/Crystals/frames --only_exemplars
```
  
* To test the multi-modal setting, with both exemplars and text, run the following commands:
```
python test_crystals.py --output_file crystals-count-exemplars-and-text-predicted.json --data_dir VideoCount/Crystals --checkpoint_dir checkpoints --use_exemplars
```
```
python evaluate_counting_accuracy.py --ground_truth VideoCount/Crystals/anno/crystals-count-gt.json --predicted crystals-count-exemplars-and-text-predicted.json --parent_dir VideoCount/Crystals/frames
```


## Training CountGD-Box
The training code is downloadable [here](https://drive.google.com/file/d/1jLe9OP4MXr-yVfS-CruXRDF6D9H2Bi7-/view?usp=sharing). Once trained, the model can be used for inference in the main repository. The command used for training is in train.sh (in the zip folder).

## Citation
Please cite our related papers if you build off of our work.
```
@InProceedings{AminiNaieni25,
  title={Open-World Object Counting in Videos},
  author={Amini-Naieni, N. and Zisserman, A.},
  booktitle={Association for Advancement of Artificial Intelligence Conference (AAAI)},
  year={2026}
}

@InProceedings{AminiNaieni24,
  title = {CountGD: Multi-Modal Open-World Counting},
  author = {Amini-Naieni, N. and Han, T. and Zisserman, A.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2024},
}
```
## Acknowledgements
### Code
This repository uses code from the [CountGD repository](https://github.com/niki-amini-naieni/CountGD), the [SAM 2 repository](https://github.com/facebookresearch/sam2), and the [GeCo repository](https://github.com/jerpelhan/GeCo). If you have any questions about our code implementation, please contact us at [niki.amini-naieni@eng.ox.ac.uk](mailto:niki.amini-naieni@eng.ox.ac.uk).
### Data
We use data from the following sources:

**TAO: A Large-Scale Benchmark for Tracking Any Object.**
Achal Dave, Tarasha Khurana, Pavel Tokmakov, Cordelia Schmid and Deva Ramanan, [arXiv:2005.10356](https://arxiv.org/abs/2005.10356)

**MOT20: A benchmark for multi object tracking in crowded scenes.**
Patrick Dendorfer, Hamid Rezatofighi, Anton Milan, Javen Shi, Daniel Cremers, Ian Reid, Stefan Roth, Konrad Schindler, Laura Leal-Taixe [arXiv:2003.09003](http://arxiv.org/abs/2003.09003) 

Dr Tom Hart, [Penguin Watch](https://www.zooniverse.org/projects/penguintom79/penguin-watch), School of Biological and Medical Sciences, Oxford Brookes University (for the Penguins benchmark in Science-Count)

Dr Enzo Liotti and Shun Yang, Department of Materials, University of Oxford (for the Crystals benchmark in Science-Count)

### Funding
We have received funding from the [UKRI Grant VisualAI](https://www.robots.ox.ac.uk/~vgg/projects/visualai/), AWS, a Qualcomm Innovation PhD Fellowship, an Oxford-Reuben Graduate Scholarship, and Darwin Plus.
