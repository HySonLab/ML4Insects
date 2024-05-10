# EPGS (EPG Segmentation) - A library for EPG signal analysis of pierce-sucking insects

Electrical penetration graph (EPG) is a technique used to study the feeding behavior of sucking insects such as aphids. Specifically, the experimental insect and host plant are made part of an electrical circuit, which is closed when aphid mouthparts penetrate plant tissue. When the aphid stylet is inserted intercellularly, the voltage is positive and when inserted intracellularly, the voltage is negative. Waveforms in EPG have been correlated to specific aphid feeding behaviors by stylectomy followed by microscopy of the plant tissue to determine the approximate location of the stylet as well as observing aphid head movement, posture, and muscle dynamics. EPG is well established and has been widely used to study the mechanisms of plant virus transmission by aphids, the effect of resistant and susceptible lines on aphid feeding behaviors, and to better our understanding of the mechanisms that aphids use to continuously drink from the phloem.   

The EPGS (abbreviated for EPG-Segmentation) package is an open-source Python package, designed to be compatible with the popular Stylet+ EPG System by **cite EPG system**. EPGS can provide many functions including data visualization, accurate automatic segmentation, and calculations of various EPG parameters, which facilitate the data analysis stage in the study of EPG signals. The package was used as a support tool for our study.

The data visualization functions are based on well-known visualization libraries such as matplotlib and plotly. To help with visualizing a huge numbers of data points, the code includes **cite plotly resampler**. 
With EPGS, users can visualize the entire signal in both _static_ and _interactive_ states.

With simple syntax, the package provides several machine learning algorithms for EPG signal segmentation, among which the fully convolutional network (FCN) and extreme gradient boosting classifier (XGB) is recommended. The segmentation algorithms follows a sliding-window technique where the entire signal is broken into non-overlapping segments, then the each of them is labeled independently before concatenating the predictions to form a unified segmentation. Although the approach and the algorithms are simple, we observe great performance in terms of 1) the ability of ML models to recognize a pattern from a segment and 2) the overall overlapping rate between the prediction and the ground-truth. Therefore, we believe that there are much room for improvement thanks to the existence of numerous segmentations algorithms in image segmentation, which is a similar problem. 

Finally, the calculation of EPG parameters is a significant part of EPGS data analysis, so we include this function into EPGS in order to make it a well-rounded package, providing necessary tools all-in-one. 

Contributors:
* Quang Dung Dinh
* Truong Son Hy
* Phuong Dao
