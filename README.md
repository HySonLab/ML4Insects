# DiscoEPG (Discovery EPG) - A library for EPG signal analysis of pierce-sucking insects 🐞🍃⚡💻

## 🌎 Overview
Electrical penetration graph (EPG) is a technique used to study the feeding behavior of sucking insects such as aphids. Specifically, the experimental insect and host plant are made part of an electrical circuit, which is closed when aphid mouthparts penetrate plant tissue. When the aphid stylet is inserted intercellularly, the voltage is positive and when inserted intracellularly, the voltage is negative. Waveforms in EPG have been correlated to specific aphid feeding behaviors by stylectomy followed by microscopy of the plant tissue to determine the approximate location of the stylet as well as observing aphid head movement, posture, and muscle dynamics. EPG is well established and has been widely used to study the mechanisms of plant virus transmission by aphids, the effect of resistant and susceptible lines on aphid feeding behaviors, and to better our understanding of the mechanisms that aphids use to continuously drink from the phloem.   

![ML4Insects](/figures/workflow.png "Workflow of the segmentation approach used in the package.")

DiscoEPG (abbreviated for Discovery-EPG) package is an open-source Python package, designed to be compatible with the popular Stylet+ EPG System by W. F. Tjallingii [1]. DiscoEPG provides many utilities including data visualization, accurate automatic segmentation, and calculations of various EPG parameters, which facilitate the data analysis stage in the study of EPG signals. The package was used as a helpful support tool for our study in characterizing aphid's behavior based on this data. 

The novelty of DiscoEPG lies in the automatic segmentation procedure, which follows a sliding-window technique where the entire signal is broken into non-overlapping segments, then the each of them is labeled independently before concatenating the predictions to form a unified segmentation. Despite being simple, we observe great performance in terms of 1) the segment classification results and 2) the overlap rate between the prediction and the ground-truth aggregated segmentation. 

## 📁 Novel features of DiscoEPG

### ML for characterizing EPG waveforms
DiscoEPG provides two trainer objects `EPGSegment` and `EPGSegmentML` which respectively support training Deep Learning (CNN1D, ResNet and CNN2D) and Traditional Machine Learning (XGB, Random Forest, Logistic Regression) for automatically detect EPG waveforms. For Deep Learning models, it is possible to save the trained model for future use, while only `XGB` from the other group provide a similar function. The prediction results can be plot to visually assess or make post-prediction refinement, as alignment errors is unavoidable. To make this step easier, `EPGSegment` allows saving the prediction result in a `*.ANA` file which can be processed by Stylet+. 

### Visualization
DiscoEPG allows users to create color plots,  in both _static_ and _interactive_ states of EPG recordings. The data visualization functions are based on well-known visualization libraries such as matplotlib and plotly. To help with visualizing a huge numbers of data points, `plotly-resampler` [2] was incorporated into our package. The figure below shows an example of a plot between the predicted segmentation and the ground-truth version. The overlap rate is 95%, where the errors were mostly caused by minor waveforms such as pd. 

![ML4Insects](/figures/prediction.png)

### EPG parameters calculation 
DiscoEPG can calculate various EPG parameters proposed for aphids, adopted from [4].

### 
## 📓 Example of usage

### Installation
To install DiscoEPG, simply run 

``` 
pip install DiscoEPG
```

For DiscoEPG to run properly, you only need to prepare a dataset folder which contains all the dataset containing recordings with the ASCII format obtained from [Stylet+ application](https://www.DiscoEPGystems.eu/). Corresponding to each dataset `<dataset_name>`, there should be one subfolder called `<dataset_name>` containing the recording data (with `.A0x` extension) and another one called `<dataset_name>_ANA` containing the waveform position (with `.ANA` extension). Each complete recording comprises of multiple recording files. In case the prefix of these data files are different, DiscoEPG will automatically rename them from `<file_name>.A0x` into `<folder_name>_<file_name>.A0x`, as well as the analysis file.

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
├── config
|   └── config_file_name.json
|	└── ...
└── Your Python script
└── ...
```

### For loading EPG data and doing EPG parameters calculation
```
from DiscoEPG import EPGDataset
root_dir = <your_working_directory>
dataset = EPGDataset(data_path = root_dir, dataset_name = <a_dataset_name>)
```

### For training/making inference with ML models
```
from DiscoEPG import EPGSegment # Importing trainer objects
from DiscoEPG.utils import process_config
config_file = <the_path_to_your_config_file> # Define the path to your config file
config = process_config(config_file)
epgs = EPGSegment(config) # Call the trainer
```
**NOTE.** Please refer to the tutorial notebooks for explicit detail on how to work with DiscoEPG. 

## 💡 Acknowledgement
We hugely thanks the authors of the cited work for providing us with necessary tools which were the building blocks for DiscoEPG. 

## ✅ If you find our work helpful, please cite it with

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

## 🧑‍🔬 Contributors
* Quang-Dung DINH, Institut Galilée, Universite Sorbonne Paris Nord, Villetaneuse 93430, Paris, France
* Truong Son HY (PI), Department of Mathematics and Computer Science, Indiana State University, Terre Haute, IN 47809, United States
* Phuong DAO (PI), Department of Agricultural Biology, Colorado State University, Fort Collins, CO 80523, United States

## 📖 References
[1.](https://onlinelibrary.wiley.com/doi/10.1111/j.1570-7458.1978.tb02836.x) _Aphids' EPG waveforms_. Tjallingii WF. Electronic Recording of Penetration Behaviour by Aphids Entomologia Experimentalis et Applicata. 1978; 24(3): 721–730.

[2.](https://ieeexplore.ieee.org/document/9973221) _Package for effective EPG visualization_. J. Van Der Donckt, J. Van der Donckt, E. Deprost and S. Van Hoecke, "Plotly-Resampler: Effective Visual Analytics for Large Time Series," 2022 IEEE Visualization and Visual Analytics (VIS), Oklahoma City, OK, USA, 2022, pp. 21-25. [GitHub](https://github.com/predict-idlab/plotly-resampler) 

[3.](https://link.springer.com/article/10.1007/s11263-019-01194-0) _The Pytorch implementation of wavelet transform_. Runia, T.F.H., Snoek, C.G.M. & Smeulders, A.W.M. Repetition Estimation. Int J Comput Vis 127, 1361–1383 (2019). [GitHub](https://github.com/tomrunia/PyTorchWavelets) 

[4.](https://academic.oup.com/jinsectscience/article/24/3/28/7701043) _The EPG parameters which we adopt_. Elisa Garzo, Antonio Jesús Álvarez, Aránzazu Moreno, Gregory P Walker, W Fred Tjallingii, Alberto Fereres, Novel program for automatic calculation of EPG variables, Journal of Insect Science, Volume 24, Issue 3, May 2024, 28.
