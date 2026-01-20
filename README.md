# FFreGNN
Learning to Understand Financial Risk Contagion from A Frequency-Domain Graph Learning Framework

## Abstract

Understanding and predicting financial risk contagion is paramount for effective risk management and investment decision-making. Although the effectiveness of deep graph learning in this area is widely recognised, existing methods are confined to time-domain frameworks and struggle to capture contagion pathways beyond the most direct or synchronous ones. This makes it difficult to resolve the risk propagation anomalies that are inherent in financial markets. To overcome this limitation, we propose a Finance Frequency-Domain Graph Neural Network, an end-to-end graph learning framework based on the frequency domain. Specifically, (1) a knowledge-guided paradigm alignment module performs spectral calibration via learnable market frequency prototypes to project stock features into a unified semantic space; (2) a dynamic relation inference mechanism distills risk contagion pathways by quantifying spectral coherence and phase differentials between dominant frequency bands; and (3) a state-aware risk propagation module achieves adaptive modeling of risk evolution by re-routing the flow of risk information based on each node's unique state. Extensive experiments on real-world datasets from the Chinese and US stock markets demonstrate that our model achieves the best performance in terms of both prediction accuracy and investment metrics, highlighting the effectiveness of frequency-domain modeling.

## Dataset
https://pan.baidu.com/s/1rHzwyQYyvG_NU-aQH1aSGQ?pwd=g9ie (This link contains the three Chinese market datasets used.)

## Procedure
![Image text](https://github.com/Xsc-Swufe/FFreGNN/blob/main/FFreGNN/procdure.png)

## Requirements

Python == 3.11  

torch == 2.0.1 

torchvision == 0.15.2  

numpy == 1.24.4  

## How to train the model
Run main.py python main.py --device=$your_device_id(default=0) --hidden=$hidden(default=128) --length=$length(default=60) --scale_num=$scale_num(default=3) --path_num=$path_num(default=6)


## Contact
Xiaosc123456@gmail.com
