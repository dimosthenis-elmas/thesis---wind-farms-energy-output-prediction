![Image: Model predictions for a 18h context window and 14 hours of predictions for all 10 wind farms](<readme image 1.png>)

<sub><h4>Model predictions for a 18h context window and 14 hours of predictions for all 10 wind farms<h4></sub>

<div align="center"> <h1>Forecasting renewable energy generation using deep learning<h1></div>
<div align="center"> <h2>Dimosthenis Elmas<h2></div>
<div align="center"> <h3>A thesis submitted to the Faculty of Sciences of the School of Informatics of the Aristotle University of Thessaloniki<h3> </div>
<div align="center"> <h3>in partial fulfillment of the requirements for the<h3></div>
<br>
<div align="center"> <h2>Master’s degree in Artificial Intelligence<h2> </div>


This repository contains part of the experimentations for my master's degree thesis: Forecasting renewable energy generation using deep learning. 

In this thesis we studied, among other things, the problem of forecasting time series concerning the production of electricity of a cluster of 10 wind farms. The multivariate time series we used concern a set of wind farms which, due to their geographical proximity, exhibit correlations in terms of the electricity production. Such time series partly depend on future interpretive variables or covariates, (that is, variables that refer to future moments such as weather forecasts, etc.), and partly on the so-called spatio-temporal correlations. 

We employed 2 strategies: 

- One based on the Informer architecture (Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, Wancai Zhang) and the use of what we call 'sliding windows' (a two stage (fine tuning) training strategy in an RNN fashion using expanded gradient tapes (in the first stage: a standard training cycle. In the second stage: later predictions are based on the previous network predictions and this is enforced inside the training loop by expanding the gradient tapes, much like in an RNN network).   

- And a second one based on probabilistic diffusion models  (CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation
Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon).

- The benchmark we used for our comparisons was based on the TFT architecture: (Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting Authors: Bryan Lim, Sercan Arik, Nicolas Loeff and Tomas Pfister).




