# Master thesis: Radar Image Reconstruction with Raw ADC data

## Introduction

This repository contains code and models for radar image reconstruction using raw ADC data. The models developed for this thesis are based on the T-FFTRadNet architecture. We explore three variations of this model:

- T-FFTRadNet: The original model from the paper used as a baseline.  
- Extended T-FFTRadNet: An extension that includes the prediction of Doppler values.  
- ADAT-FFTRadNet: A variation where the encoder structure is modified, integrating the backbone of TransRadar for better performance.  

Each model is trained and evaluated with different configurations of input data and hyperparameters.

|          Model         |                description                   |      Train     |  Evaluation |
|------------------------|----------------------------------------------|---------------|-----------------|
|      T-FFTRadNet       |   The original model from the paper          | 6-Train_optuna.py         |    3-Evaluation |
|   Extended T-FFTRadNet | Additional prediction on doppler values      | 6-Train_optuna_RARD.py         |    3-Evaluation_RARD.py        |
|   ADAT-FFTRadNet       | Alter the encoder structure with the backbone of TransRadar     |  6-Train_optuna_RARD_ADA.py         |      3-Evaluation_RARD_ADA.py      |

# Training

The models are trained using Optuna for hyperparameter tuning, which helps find the best model configuration.

## Hyperparameter tuning with optuna
To start hyperparameter tuning with Optuna, run the following command:
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/6-Train_optuna.py --config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/RD_matlab_server_config.json --trials 2
```
--config: path/to/the/config/file
--trials: indicate the number of trials you want to test 

For the initial training, since the config files provided include normalization constants for each of the differing input types. 
To obtain the normalization constants for each input type, use the following command within the dataset folder:
`$ python print_dataset_statistics.py`


To do hyperparameter tuning and finetuning from checkpoint
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/6-Train_optuna.py --config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/RD_matlab_server_ft_config.json --resume /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/RADIal_SwinTransformer_RD.pth --trials 1
```

### Training command for each model

1. T-FFTRadNet:
(1)RD input
```
python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/6-Train_optuna.py --config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/RD_matlab_server_config.json --trials 30
```
(2)ADC input
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/6- Train_optuna.py --config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/ADC_matlab_server_config.json --trials 30
```

2. Extended T-FFTRadNet
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/6-Train_optuna_RARD.py --config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/ADC_matlab_server_RARD_config.json --trials 60
```

3. ADAT-FFTRadNet
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/6-Train_optuna_RARD_ADA.py -- config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/ADC_matlab_server_RARD_ADAblock_config.json --trials 60

```

# Evaluation
To do the evaluation and export the evaluation score file
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/3-Evaluation_RARD.py --config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/ADC_matlab_server_doppler_config.json -- checkpoint /imec/other/dl4ms/chu06/public/model_checkpoint/TFFTRadNet/TFFTRadNet_ADC_RARD/SwinTra nsformer_RD___Sep-23-2024___15:55:43/SwinTransformer_RD_epoch79_loss_291.2625_AP_0.0000 _AR_0.0000_trialnumber_00_batch04.pth
```

1. T-FFTRadNet:
(1)RD input
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/3-Evaluation.py --config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/RD_matlab_server_config.json -- checkpoint /imec/other/dl4ms/chu06/public/model_checkpoint/TFFTRadNet/TFFTRadNet-optuna-10000/SwinTransformer_RD___Sep -01-2024___01:37:22/SwinTransformer_RD_epoch80_loss_11.9749_AP_0.8219_AR_0.9463_trialnumber_01_batch04.pth
```
(2)ADC input
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/3-Evaluation.py --config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/ADC_matlab_server_config.json -- checkpoint /imec/other/dl4ms/chu06/public/model_checkpoint/TFFTRadNet/TFFTRadNet_ADC/SwinTransformer_RD___Aug-31-2024___18:23:32/SwinTransformer_RD_epoch80_loss_16.8008_AP_0.7929_AR_0.8462_trialnumber_00_batch04.pth
```

2. Extended T-FFTRadNet
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/3-Evaluation_RARD.py --config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/ADC_matlab_server_doppler_config.json -- checkpoint /imec/other/dl4ms/chu06/public/model_checkpoint/TFFTRadNet/TFFTRadNet_ADC_RARD/SwinTra nsformer_RD___Sep-23-2024___15:55:43/SwinTransformer_RD_epoch79_loss_291.2625_AP_0.0000 _AR_0.0000_trialnumber_00_batch04.pth --plot --eval
```
--plot: plot the detection and output prabaility map
--eval: calculate the evaluation score 

3. ADAT-FFTRadNet
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/3-Evaluation_RARD_ADA.py--config /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/config/ADC_matlab_server_RARD_ADA_config.json --checkpoint /imec/other/dl4ms/chu06/public/model_checkpoint/TFFTRadNet/TFFTRadNet_ADC_RARD_AD A/SwinTransformer_RD___Oct-24-2024___15:58:14/SwinTransformer_RD_epoch60_loss_ 706.7412_AP_0.5153_AR_0.8890_trialnumber_00_batch08.pth --plot --eval
```
--plot: plot the detection and output prabaility map
--eval: calculate the evaluation score 

## Evaluation plot with IoU threshold 

To generate plots with different IoU thresholds for testing and training datasets:
1. Testing Dataset (T-FFTRadNet with RD and ADC input, Extended T-FFTRadNet with ADC input):
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/7-Evaluation_plot.py /imec/other/dl4ms/chu06/TFFTRadNet_detection_score_RD.txt /imec/other/dl4ms/chu06/TFFTRadNet_detection_score_ADC.txt /imec/other/dl4ms/chu06/TFFTRadNet_detection_score_ADC_RARD_lr1-3e.txt
```

2. Training Dataset (T-FFTRadNet with ADC input, Extended T-FFTRadNet with ADC input, ADAT-FFTRadNet with ADC input):
```
$ python /imec/other/dl4ms/chu06/T_FFTRadNet/RadIal/7-Evaluation_plot.py /imec/other/dl4ms/chu06/TFFTRadNet_detection_score_ADC.txt /imec/other/dl4ms/chu06/TFFTRadNet_detection_score_ADC_RARD_lr1-3e.txt /imec/other/dl4ms/chu06/TFFTRadNet_detection_score_ADC_RARD_ADAblock.txt
```

# Backup code

These scripts provide alternative or parallel implementations:

- For parallel computing implementation: 6-Train_optuna_lightning.py

- For different weights loss in RD and RA prediction maps: 6-Train_optuna_RA3RD.py and 6-Train_optuna_RA3RD_ADA.py


# Below are ReadME content from the original paper


# T-FFTRadNet:  Object Detection with Swin Vision Transformers from Raw ADC Radar Signals
Accepted into the Bravo Workshop at ICCV 2023.

This repository contains all code needed to reproduce experiments and is based on the implementation from [ValeoAI](https://github.com/valeoai/RADIal).

HD Radar (RadIal dataset) and LD Radar (RADDet dataset) models are split into separate folders. Each folder follows a similar structure.

# Abstract 
Object detection utilizing Frequency Modulated Continuous Wave radar is becoming increasingly popular in the field of autonomous systems. Radar does not possess the same drawbacks seen by other emission-based sensors such as LiDAR, primarily the degradation or loss of return signals due to weather conditions such as rain or snow. However, radar does possess traits that make it unsuitable for standard emission-based deep learning representations such as point clouds. Radar point clouds tend to be sparse and therefore information extraction is not efficient. To overcome this, more traditional digital signal processing pipelines were adapted to form inputs residing directly in the frequency domain via Fast Fourier Transforms. Commonly, three transformations were used to form Range-Azimuth-Doppler cubes in which deep learning algorithms could perform object detection. This too has drawbacks, namely the pre-processing costs associated with performing multiple Fourier Transforms and normalization. We explore the possibility of operating on raw radar inputs from analog to digital converters via the utilization of complex transformation layers. Moreover, we introduce hierarchical Swin Vision transformers to the field of radar object detection and show their capability to operate on inputs varying in pre-processing, along with different radar configurations, i.e., relatively low and high numbers of transmitters and receivers, while obtaining on par or better results than the state-of-the-art.
# Contents
- [Requirements](#Section-1)
- [Dataset Access](#Section-2)
- [T-FFTRadNet](#Section-3)
- [Usage](#Section-4)
    

# Requirements

## Model Development

System:     Windows 11  
Python:     3.9.12  
Pytorch:    1.12.1  
CUDA:       11.3  
Conda:      4.12.0  

For package requirements run:

`$ conda create --name <env> --file requirements.txt`

# Dataset Access

We utilize two datasets, [RadIal](https://github.com/valeoai/RADIal) and [RADDet](https://github.com/ZhangAoCanada/RADDet#DatasetLink). Each of these datasets can be downloaded from their linked repositories.

For RADDet, we utilize the raw ADC format and perform all pre-processing in the dataloading pipeline. The downloading of the full RAD cubes is not needed.

For RadIal, we utilize both raw ADC and their pre-processed Range-Doppler matrices provided in the link above. We have provided additional code within the RadIal folder to store raw ADC matrices as numpy arrays possessing the correct numbering scheme with respect to the annotation .csv file.

Modify the data_config.json file with the correct paths to the CalibrationTable.npy and labels.csv files. The raw binary files should be located under the Data_dir field.


```
	'Calibration': 'path/to/CalibrationTable.npy',

	'Method': 'ADC',

	'label_path': '/path/to/labels.csv',

	'Data_Dir': 'path/to/raw_binary',

	'Output_Folder': '/path/to/output_destination'
    
```    
You can then execute the following command within the ADCProcessing folder to generate the raw ADC dataset:

`$ python Make_ADC_Data.py --config /path/to/data_config.json`
 


# T-FFTRadNet 

T-FFTRadNet builds off prior work from [FFTRadNet](https://github.com/valeoai/RADIal/tree/main/FFTRadNet), utilizing heirachical Swin Vision transformers as the feature extraction head on Range-Doppler inputs.

![alt text](Figures/Diagram_v3.png)

The model is capable of utilizing raw ADC inputs via complex-valued linear layers from [CubeLearn](https://github.com/zhaoymn/cubelearn) that mimic the action of a 2D Fourier Transform. These layers provide the removal of all pre-processing/normalization and provide increased mAP in LD radar settings. In HD radar, performance is approximately the same as standard Fourier Transform. Inference time is greatly reduced via the utilization of raw ADC and the complex-valued layers.

![alt text](Figures/Example_FFT_Layers.png)

# RADDet Usage

Download the raw ADC files along with the gt_box and gt_box_test files and structure them as follows:
```
└── RADDet_Data/  
    ├── ADC/   
    ├── gt_box/  
    └── gt_box_test/
```

In the Train.py and Evaluation.py folders, two datasets will be created with flags 'Training' and 'Testing'. All testing indices will be subtracted from the training dataset.

We provide three config files, 'RAD_config.json, RD_config.json' and 'ADC_config.json'. The data_mode field will control inputs to the model and model creation. The name field will correspond to the experiment name and will be included in the newly created folder where the model is saved at each epoch during training, the config file used is copied and other log files. Please modify the root_dir field to point to where the RADDet_Data folder is stored, and modify the output directory field to where you want created models to be stored.

A model utilizing ADC inputs can then be trained using the following command:

`$ python Train.py --config config/ADC_config.json`

To evaluate models run the following command:

`$ python Evaluate.py --config /path/to/my_experiment/config.json --checkpoint /path/to/my_experiment/model.pth`

The config files provided include normalization constants for each of the differing input types. To obtain your own normalization constants you can run the following command within the dataset folder:

`$ python print_dataset_statistics.py`

# RadIal Usage

After producing the raw ADC data, downloading the Range-Doppler matrices and other data files structure them as below:
```
└── RadIal_Data/
    ├── ADC_Data/ 
    ├── camera/
    ├── laser_PCL/
    ├── radar_FFT/
    ├── radar_Freespace/
    ├── radar_PCL/
    └── labels.csv
```

Note: It is possible to utilize raw ADC and perform all Fourier Transforms in the dataloading pipeline. However, this is slow and we reccomend creating new files for these inputs and loading them in.

We provide three config files, 'RD_config.json, RD_Shift_config.json' and 'ADC_config.json'. The data_mode field will control inputs to the model and model creation, for example a Range-Doppler matrix with centered zero Doppler frequency. The name field will correspond to the experiment name and will be included in the newly created folder where the model is saved at each epoch during training, the config file used is copied and other log files. Please modify the root_dir field to point to where the RadIal_Data folder is stored, and modify the output directory field to where you want created models to be stored.


A model utilizing ADC inputs can then be trained using the following command:

`$ python 1-Train.py --config config/ADC_config.json`

To evaluate models run the following command:

`$ python 3-Evaluate.py --config /path/to/my_experiment/config.json --difficult --checkpoint /path/to/my_experiment/model.pth`

The --difficult flag includes both 'easy' and 'difficult' scenes for evaluation. Removal of this implies only 'easy' scenes.

The config files provided include normalization constants for each of the differing input types. To obtain your own normalization constants you can run the following command within the dataset folder:

`$ python print_dataset_statistics.py`

Visualization of the model's output can be obtained via the following command:

`$ python 2-Test.py --config /path/to/config.json --difficult --checkpoint /path/to/model.pth`

In each case, training can be resumed via the command:

`$ python Train.py --config /path/to/config.json --resume /path/to/previous_model.pth`

Note: You will need to modify the experiment name in the config file to resume training. We reccomend 'previous_experiment_name_resume'.



# Pre-trained Models

Pre-trained models can be found at the following [link](https://drive.google.com/drive/folders/1xihOyEDL_hHrkTi4rdgJ3LvIJO_z_jIk?usp=sharing). These models can be utilized with the provided config files.
