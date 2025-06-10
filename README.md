# Introduction

DiffusionMOT: A Diffusion-based Multi-Object Tracker

# Tracking performance

| Dataset  | HOTA| IDF1 | AssA | Results                                                                                                                        
| --- | --- | --- |------|--------------------------------------------------------------------------------------------------------------------------------|
|DanceTrack| 65.3 | 66.6 | 52.4 | [DanceTrack_Results](https://github.com/sad123-yx/DiffusionMOT/releases/download/Tracking_Results/DanceTrack_DiffusionMOT.zip) |
|SportsMOT| 72.9 | 73.3 | 62.0 | [SportsMOT_Results](https://github.com/sad123-yx/DiffusionMOT/releases/download/Tracking_Results/Sportsmot_DiffusionMOT.zip)   |
|MOT20|63.3|77.4| 63.8 | [MOT20_Results](https://github.com/sad123-yx/DiffusionMOT/releases/download/Tracking_Results/MOT20_DiffusionMOT.zip)           |
|MOT17|63.7|78.4| 63.5 | [MOT17_Results](https://github.com/sad123-yx/DiffusionMOT/releases/download/Tracking_Results/MOT17_DiffusionMOT.zip)           |

# Install requirements

```
pip3 install -r requirements.txt
```

# Data preparation

The file structure should look like:

```
datasets
  └——dancetrack
  |        └———annotations
  |        └———train
  |        └———val
  |        └———train_val
  |        └———test
  └——sportsmot
  |        └———annotations
  |        └———train
  |        └———val
  |        └———train_val
  |        └———test
  └——mot
  |        └———annotations
  |        └———train
  |        └———test
  └——mot20
           └———annotations
           └———train
           └———test
```

# Training

* Train DanceTrack model (use only train set)

```
cd <DiffusionMOT>
python3 tools/DiffusionMOT_train.py -f configs/exps/yolox_x_det_dancetrack_trian.py -d 4 -b 4 -o -c models/dancetrack_train.pth.tar
python3 tools/DiffusionMOT_train.py -f configs/exps/yolox_x_track_dancetrack_trian.py -d 4 -b 4 -o -c models/dancetrack_train.pth.tar
```

* Train DanceTrack model (use train & val sets)

```
cd <DiffusionMOT>
python3 tools/DiffusionMOT_train.py -f configs/exps/yolox_x_det_dancetrack_trianval.py -d 4 -b 4 -o -c models/dancetrack_trainval.pth.tar
python3 tools/DiffusionMOT_train.py -f configs/exps/yolox_x_track_dancetrack_trianval.py -d 4 -b 4 -o -c models/dancetrack_trainval.pth.tar
```
* Train SportsMOT model (use only train set)

```
cd <DiffusionMOT>
python3 tools/DiffusionMOT_train.py -f configs/exps/yolox_x_det_sportsmot_trian.py -d 4 -b 4 -o -c models/sportsmot_train.pth.tar
python3 tools/DiffusionMOT_train.py -f configs/exps/yolox_x_track_sportsmot_trian.py -d 4 -b 4 -o -c models/sportsmot_train.pth.tar
```

* Train SportsMOT model (use train & val sets)

```
cd <DiffusionMOT>
python3 tools/DiffusionMOT_train.py -f configs/exps/yolox_x_det_sportsmot_trianval.py -d 4 -b 4 -o -c models/sportsmot_trainval.pth.tar
python3 tools/DiffusionMOT_train.py -f configs/exps/yolox_x_track_sportsmot_trianval.py -d 4 -b 4 -o -c models/sportsmot_trainval.pth.tar
```

MOT17 & MOT20 model's training strategy can refer from this [repo](https://github.com/RainBowLuoCS/DiffusionTrack), and the motion model training strategy can refer from this [repo](https://github.com/Kroery/DiffMOT).


# Tracking

* Test on DanceTrack

```
cd <DiffusionMOT>
python3 tools/DiffusionMOT_train.py -evaldata "dancetrack" -f configs/exps/yolox_x_track_dancetrack_train_val.py   -d 1 -b 1 -c models/dancetrack_trainval.pth.tar
```

* Test on SportMOT

```
cd <DiffusionMOT>
python3 tools/DiffusionMOT_train.py -evaldata "sportsmot" -f configs/exps/yolox_x_track_sportsmot_train_val.py  -d 1 -b 1 -c models/sportsmot_trainval.pth.tar
```

* Test on MOT20

```
cd <DiffusionMOT>
python3 tools/DiffusionMOT_train.py -evaldata "mot20" -f configs/exps/yolox_x_track_mot20.py  -d 1 -b 1 -c models/mot20.pth.tar
```

* Test on MOT17

```
cd <DiffusionMOT>
python3 tools/DiffusionMOT_train.py -evaldata "mot17" -f configs/exps/yolox_x_track_mot17.py  -d 1 -b 1 -c models/mot17.pth.tar
```

# Acknowledgement

A large part of the code is borrowed from [DiffusionTrack](https://github.com/RainBowLuoCS/DiffusionTrack) and [DiffMOT](https://github.com/Kroery/DiffMOT). Thanks for their wonderful works!

