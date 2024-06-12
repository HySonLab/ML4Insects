# EPGS (EPG Segmentation) - A library for EPG signal analysis of pierce-sucking insects

## Overview
Electrical penetration graph (EPG) is a technique used to study the feeding behavior of sucking insects such as aphids. Specifically, the experimental insect and host plant are made part of an electrical circuit, which is closed when aphid mouthparts penetrate plant tissue. When the aphid stylet is inserted intercellularly, the voltage is positive and when inserted intracellularly, the voltage is negative. Waveforms in EPG have been correlated to specific aphid feeding behaviors by stylectomy followed by microscopy of the plant tissue to determine the approximate location of the stylet as well as observing aphid head movement, posture, and muscle dynamics. EPG is well established and has been widely used to study the mechanisms of plant virus transmission by aphids, the effect of resistant and susceptible lines on aphid feeding behaviors, and to better our understanding of the mechanisms that aphids use to continuously drink from the phloem.   

![ML4Insects](/figures/workflow.png "Workflow of the segmentation approach used in the package.")

The EPGS (abbreviated for EPG-Segmentation) package is an open-source Python package, designed to be compatible with the popular Stylet+ EPG System by W. F. Tjallingii [1]. EPGS can provide many functions including data visualization, accurate automatic segmentation, and calculations of various EPG parameters, which facilitate the data analysis stage in the study of EPG signals. The package was used as a helpful support tool for our study in characterizing aphid's behavior based on this data. 

With simple syntaxes, the package provides several machine learning algorithms for EPG signal segmentation, among which the 1D convolutional neural network (CNN1D) and extreme gradient boosting classifier (XGB) is recommended for initial uses with EPGS. The segmentation procedure follows a sliding-window technique where the entire signal is broken into non-overlapping segments, then the each of them is labeled independently before concatenating the predictions to form a unified segmentation. Although the approach and the algorithms are simple, we observe great performance in terms of 1) the segment classification results and 2) the overlap rate between the prediction and the ground-truth aggregated segmentation. We believe that there are much room for improvement thanks to the existence of numerous segmentations/detections deep learning algorithms.

## Usage 
To use EPGS, first download the package (.zip) and put it inside your working directory at the same level with your `data` folder whose subfolders are your dataset. Your dataset folder should contains all the dataset containing recordings with the ASCII format obtained from [Stylet+ application](https://www.epgsystems.eu/). Each complete recording comprises of recording file with `.A0x` extension. Their names should be formated into `dataset.recording_name.A0x`. The analysis file containing the ground truth segmentation for a recording should be put in a folder named `dataset_ANA` at the same level with your `dataset` folder. 

For example 
```
working directory
├── data
│   ├── dataset
|   |   └── dataset.name0.A01
|   |   └── dataset.name1.A01
|   |   └── ...
|   |   └── dataset.name0.A08
|   |   └── dataset.name1.A08
|   ├── dataset_ANA
|   |   └── dataset.name0_ANA
|   |   └── dataset.name1_ANA
|   |   └── ...
|   └── ...
└── EPGS
└── ...
```

## Functions
For training a model, predicting segmenation or making visualization, please refer to the example notebooks. 
### Train ML models for characterizing EPG waveforms
EPGS provides two trainer objects for training Deep Learning (CNN1D, ResNet and CNN2D) and Traditional Machine Learning (XGB, Random Forest, Logistic Regression) for characterizing EPG waveforms. After a model is trained, it can be used for EPG waveform segmentation. The trainer objects also provide post-prediction utilities such as saving checkpoints (for DL models) and plot the segmentation result/ metrics.
### Visualization
A nice feature of EPGS is the ability to create color plot of an input recording,  in both _static_ and _interactive_ states. The data visualization functions are based on well-known visualization libraries such as matplotlib and plotly. To help with visualizing a huge numbers of data points, we use `plotly-resampler` [2]. 

![ML4Insects](/figures/prediction.png "Example of a prediction segmentation vs the ground-truth version. The overlap rate is 95%.")

## Please cite our paper with

```bibtex
@article {Dinh2024.06.10.598170,
	author = {Dinh, Quang-Dung and Kunk, Daniel and Hy, Truong Son and Nalam, Vamsi J and Dao, Phuong},
	title = {Machine learning for characterizing plant-insect interactions through electrical penetration graphic signal},
	elocation-id = {2024.06.10.598170},
	year = {2024},
	doi = {10.1101/2024.06.10.598170},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {The electrical penetration graph (EPG) is a well-known technique that provides insights into the feeding behavior of insects with piercing-sucking mouthparts, mostly hemipterans. Since its inception in the 1960s, EPG has become indispensable in studying plant-insect interactions, revealing critical information about host plant selection, plant resistance, virus transmission, and responses to environmental factors. By integrating the plant and insect into an electrical circuit, EPG allows researchers to identify specific feeding behaviors based on distinct waveform patterns associated with activities within plant tissues. However, the traditional manual analysis of EPG waveform data is time-consuming and labor-intensive, limiting research throughput. This study presents a novel machine-learning approach to automate the segmentation and classification of EPG signals. We rigorously evaluated six diverse machine learning models, including neural networks, tree-based models, and logistic regressions, using an extensive dataset from aphid feeding experiments. Our results demonstrate that a Residual Network (ResNet) architecture achieved the highest overall waveform classification accuracy of 96.8\% and highest segmentation overlap rate of 84.4\%, highlighting the potential of machine learning for accurate and efficient EPG analysis. This automated approach promises to accelerate research in this field significantly and has the potential to be generalized to other insect species and experimental settings. Our findings underscore the value of applying advanced computational techniques to complex biological datasets, paving the way for a more comprehensive understanding of insect-plant interactions and their broader ecological implications. The source code for all experiments conducted within this study is publicly available at https://github.com/HySonLab/ML4InsectsCompeting Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/06/11/2024.06.10.598170},
	eprint = {https://www.biorxiv.org/content/early/2024/06/11/2024.06.10.598170.full.pdf},
	journal = {bioRxiv}
}
```

## Contributors
* Quang-Dung Dinh
* Truong Son Hy (PI)
* Phuong Dao (PI)

## References
[1.](https://onlinelibrary.wiley.com/doi/10.1111/j.1570-7458.1978.tb02836.x) Tjallingii WF. Electronic Recording of Penetration Behaviour by Aphids Entomologia Experimentalis et Applicata. 1978; 24(3): 721–730.

[2.](https://ieeexplore.ieee.org/document/9973221) J. Van Der Donckt, J. Van der Donckt, E. Deprost and S. Van Hoecke, "Plotly-Resampler: Effective Visual Analytics for Large Time Series," 2022 IEEE Visualization and Visual Analytics (VIS), Oklahoma City, OK, USA, 2022, pp. 21-25, doi: 10.1109/VIS54862.2022.00013 . [GitHub](https://github.com/predict-idlab/plotly-resampler) 

[3.](https://link.springer.com/article/10.1007/s11263-019-01194-0) Runia, T.F.H., Snoek, C.G.M. & Smeulders, A.W.M. Repetition Estimation. Int J Comput Vis 127, 1361–1383 (2019). https://doi.org/10.1007/s11263-019-01194-0. [GitHub](https://github.com/tomrunia/PyTorchWavelets) \
