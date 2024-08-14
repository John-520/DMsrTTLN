# [IEEE TII 2024] DMsrTTLN code

This is the source code for "<b>Deep Multilayer Sparse Regularization Time-Varying Transfer Learning Networks With Dynamic Kullback–Leibler Divergence Weights for Mechanical Fault Diagnosis</b>". You can refer to the following steps to reproduce the cross device fault diagnosis experiment under time-varying speed.

## Abstract
Rotating machinery is widely used in industrial production, and its reliable operation is crucial for ensuring production safety and efficiency. Mechanical equipment often faces the challenge of variable speeds. However, existing research pays little attention to domain-adaptive and cross-device diagnostic tasks under time-varying conditions. To fill this research gap and address the serious domain shift problem in cross-device fault diagnosis tasks under time-varying speeds, this paper proposes a deep multilayer sparse regularization time-varying transfer learning network (DMsrTTLN) with dynamic Kullback–Leibler divergence weights (DKLDW). The main contributions and innovations of DMsrTTLN are as follows: 1) a multilayer sparse regularization module to effectively reduce speed fluctuations; 2) an amplitude activation function to enhance the differentiation of data with different labels; 3) the kurtosis maximum mean discrepancy (KMMD), where the Gaussian kernel function adaptively adjusts according to the kurtosis values of the data to enhance domain adaptation capability; and 4) the DKLDW mechanism dynamically balances distance and adversarial metrics to improve model convergence and stability. The DMsrTTLN model with DKLDW exhibits strong generalization performance in cross-device domain shift scenarios. Experimental validation in the same-device and cross-device scenarios is performed on three mechanical machines under time-varying speeds, and the results are compared with those of six state-of-the-art approaches. The results showed that the DMsrTTLN has a better convergence effect and greater diagnostic accuracy.

## Proposed Network

![image](https://github.com/user-attachments/assets/ea5ef1d2-45ed-4252-b6e9-a7b3b453d7bc)



## Dataset Preparation

**You can find the dataset download link in references [24-26], and the paper can be downloaded from my personal homepage [here](https://john-520.github.io/).**


### Cross-Machine Condition

For example:

```python
mian.py
---dataloaders  #dataset

```

## Contact

If you have any questions, please feel free to contact me:

- **Name:** Feiyu Lu
- **Email:** 21117039@bjtu.edu.cn

## Early Access Citation

If you find this paper and repository useful, please cite our paper 😊.

```
F. Lu, Q. Tong, X. Jiang, Z. Feng, J. Xu and J. Huo, "Deep Multilayer Sparse Regularization Time-Varying Transfer Learning Networks With Dynamic Kullback–Leibler Divergence Weights for Mechanical Fault Diagnosis," in IEEE Transactions on Industrial Informatics, doi: 10.1109/TII.2024.3438229.
keywords: {Fault diagnosis;Task analysis;Feature extraction;Adaptation models;Vibrations;Nonhomogeneous media;Transfer learning;Cross-device;fault diagnosis;maximum mean diversity (MMD);time-varying;transfer learning},


```
