## Tensor Tracking, Streaming/Online/Adaptive Tensor Decomposition, Dynamic Tensor Analysis
[On going] A list of up-to-date papers on streaming tensor decomposition, tensor tracking, and dynamic tensor decomposition.

I will continue to update this list with the newest resources. If you find any missed resources (paper/code) or errors, please feel free to open an issue, make a pull request, or contact me via thanhle88.tbt@gmail.com.


## Survey Paper
[*A Contemporary and Comprehensive Survey on Streaming Tensor Decomposition*](https://ieeexplore.ieee.org/document/9994046) [PDF](https://thanhtbt.github.io/files/2022_TKDE_A%20Contemporary%20and%20Comprehensive%20Survey%20on%20Streaming%20Tensor%20Decomposition.pdf)

Authors: Thanh Trung Le, Karim Abed-Meraim, Nguyen Linh Trung and Adel Hafiane

## Tracking Under CP/PARAFAC Format
#### Subspace-based Methods 

* PARAFAC-SDT/-RLS: “Adaptive algorithms to track the PARAFAC decomposition of a third-order tensor,” in **IEEE Trans. Signal Process.**, 2009, [Paper](https://ieeexplore.ieee.org/document/4799120), [Code](http://dimitri.nion.free.fr/)

* 3D-OPAST: ``Fast adaptive PARAFAC decomposition algorithm with linear complexity", in **IEEE ICASSP**, 2016, [Paper](https://ieeexplore.ieee.org/document/7472876)

* CP-PETRELS: ``Adaptive PARAFAC decomposition for third-order tensor completion", in **IEEE ICCE**, 2016, [Paper](https://ieeexplore.ieee.org/document/7562652)

* SOAP: "Second-order optimization based adaptive PARAFAC decomposition of three-way tensors", in **Elsevier DSP**, 2017, [Paper](https://www.sciencedirect.com/science/article/pii/S105120041730009X), [Code](https://drive.google.com/drive/folders/1x6PdEsr-1xDccm7titi5dQPLwgcKuOii)

#### Block-Coordinate Descent

* TeCPSGD: "Subspace Learning and Imputation for Streaming Big Data Matrices and Tensors", in **IEEE TSP**, 2015, [Paper](https://ieeexplore.ieee.org/document/7072498), [Code](https://github.com/hiroyuki-kasai/OLSTEC/tree/master/benchmark/TeCPSGD)

* OLCP: "Accelerating Online CP Decompositions for Higher Order Tensors", in **ACM SIGKDD**, 2016, [Paper](https://dl.acm.org/doi/abs/10.1145/2939672.2939763), [Code](https://shuozhou.github.io/)

* OLSTEC: "Online low-rank tensor subspace tracking from incomplete data by CP decomposition using recursive least squares", in **IEEE ICASSP**, 2016
[Paper](https://ieeexplore.ieee.org/abstract/document/7472131), [Code](https://github.com/hiroyuki-kasai/OLSTEC)

* CP-stream: "Streaming tensor factorization for infinite data sources", in **SDM**, 2018, [Paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611975321.10), [Code](https://github.com/ShadenSmith/splatt)

* SPADE: "SPADE: Streaming PARAFAC2 decomposition for large datasets", in **SDM**, 2020, [Paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611976236.65), [Code](http://www.cs.ucr.edu/~egujr001/ucr/madlab/src/SPADE.zip)


* SOFIA: "Robust Factorization of Real-world Tensor Streams with Patterns, Missing Values, and Outliers", in **IEEE ICDE**, 2020, [Paper](https://ieeexplore.ieee.org/abstract/document/9458640), [Code](https://github.com/wooner49/sofia)

* iCP-AM: "Incremental CP tensor decomposition by alternating minimization method", in **SIAM J. Matrix Anal. Appl**, 2020, [Paper](https://epubs.siam.org/doi/abs/10.1137/20M1319097) 


* ROLCP: "A Fast Randomized Adaptive CP Decomposition for Streaming Tensors", in **IEEE ICASSP**, 2021,  [Paper](https://ieeexplore.ieee.org/abstract/document/9413554), [Code](https://github.com/thanhtbt/ROLCP)
  
* RACP: "Robust Tensor Tracking with Missing Data and Outliers: Novel Adaptive CP Decomposition and Convergence Analysis" in **IEEE Trans. Signal Process.**, 2022, [Paper](https://ieeexplore.ieee.org/document/9866940) [Code](https://github.com/thanhtbt/tensor_tracking)
  
* ACP: "Tracking online low-rank approximations of higher-order incomplete streaming tensors", in **Cell Patterns**, 2023,  [Paper](https://www.sciencedirect.com/science/article/pii/S2666389923001046), [Code](https://github.com/thanhtbt/tensor_tracking)

#### Bayesian Inference

* POST: "Probabilistic streaming tensor decomposition", in **IEEE ICDM**, 2018,  [Paper](https://ieeexplore.ieee.org/abstract/document/8594834), [Code](https://github.com/yishuaidu/POST)

* BRST: "Variational Bayesian inference for robust streaming tensor factorization and completion", in **IEEE ICDM**, 2018,  [Paper](https://ieeexplore.ieee.org/abstract/document/8594834), [Code](https://github.com/colehawkins/Robust-Streaming-Tensor-Factorization)

## Reference
If you find this repository helpful for your work, please cite

[1] L.T. Thanh, K. Abed-Meraim, N. L. Trung and A. Hafiane. “[*A Contemporary and Comprehensive Survey on Streaming Tensor Decomposition*](https://ieeexplore.ieee.org/document/9994046)”. **IEEE Trans. Knowl. Data Eng., 2023**. [PDF](https://thanhtbt.github.io/files/2022_TKDE_A%20Contemporary%20and%20Comprehensive%20Survey%20on%20Streaming%20Tensor%20Decomposition.pdf).

