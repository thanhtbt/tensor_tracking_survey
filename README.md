## Tensor Tracking, Streaming/Online/Adaptive/Incremental Tensor Decomposition, Dynamic Tensor Analysis
[On going] A list of up-to-date papers on streaming tensor decomposition, tensor tracking, and dynamic tensor decomposition.

I will continue to update this list with the newest resources. If you find any missed resources (paper/code) or errors, please feel free to open an issue, make a pull request, or contact me via thanhle88.tbt@gmail.com.


## Survey Paper
[*A Contemporary and Comprehensive Survey on Streaming Tensor Decomposition*](https://ieeexplore.ieee.org/document/9994046) [PDF](https://thanhtbt.github.io/files/2022_TKDE_A%20Contemporary%20and%20Comprehensive%20Survey%20on%20Streaming%20Tensor%20Decomposition.pdf)

Authors: Thanh Trung Le, Karim Abed-Meraim, Nguyen Linh Trung and Adel Hafiane

## Dynamic Tensor Analysis Under CP/PARAFAC Format
### Subspace-based Methods 

* PARAFAC-SDT/-RLS: “Adaptive algorithms to track the PARAFAC decomposition of a third-order tensor,” in **IEEE Trans. Signal Process.**, 2009, [Paper](https://ieeexplore.ieee.org/document/4799120), [Code](http://dimitri.nion.free.fr/)

* 3D-OPAST: ``Fast adaptive PARAFAC decomposition algorithm with linear complexity", in **IEEE ICASSP**, 2016, [Paper](https://ieeexplore.ieee.org/document/7472876)

* CP-PETRELS: ``Adaptive PARAFAC decomposition for third-order tensor completion", in **IEEE ICCE**, 2016, [Paper](https://ieeexplore.ieee.org/document/7562652)

* SOAP: "Second-order optimization based adaptive PARAFAC decomposition of three-way tensors", in **Elsevier DSP**, 2017, [Paper](https://www.sciencedirect.com/science/article/pii/S105120041730009X), [Code](https://drive.google.com/drive/folders/1x6PdEsr-1xDccm7titi5dQPLwgcKuOii)

### Block-Coordinate Descent

* TeCPSGD: "Subspace Learning and Imputation for Streaming Big Data Matrices and Tensors", in **IEEE Trans. Signal Process.**, 2015, [Paper](https://ieeexplore.ieee.org/document/7072498), [Code](https://github.com/hiroyuki-kasai/OLSTEC/tree/master/benchmark/TeCPSGD)

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

### Bayesian Inference

* POST: "Probabilistic streaming tensor decomposition", in **IEEE ICDM**, 2018,  [Paper](https://ieeexplore.ieee.org/abstract/document/8594834), [Code](https://github.com/yishuaidu/POST)

* BRST: "Variational Bayesian inference for robust streaming tensor factorization and completion", in **IEEE ICDM**, 2018,  [Paper](https://ieeexplore.ieee.org/abstract/document/8594834), [Code](https://github.com/colehawkins/Robust-Streaming-Tensor-Factorization)

* SBDT: "Streaming Bayesian deep tensor factorization", in **ICML**, 2021, [Paper](https://proceedings.mlr.press/v139/fang21d/fang21d.pdf), [Code](https://github.com/xuangu-fang/Streaming-Bayesian-Deep-Tensor)

### Multi-aspect Streaming CP Decomposition

* MAST: "Multi-aspect streaming tensor completion", in **ACM SIGKDD**, 2017, [Paper](https://dl.acm.org/doi/pdf/10.1145/3097983.3098007), [Code](https://github.com/xuangu-fang/Streaming-Bayesian-Deep-Tensor) 

* OR-MSTC: "Outlier-Robust Multi-Aspect Streaming Tensor Completion and Factorization", in **IJCAI**, 2019, [Paper](https://www.ijcai.org/proceedings/2019/442) 

* InParTen2:  "Multi-aspect incremental tensor decomposition based on distributed in-memory big data systems", in **J.DataInf. Sci.**, 2020, [Paper](https://www.ijcai.org/proceedings/2019/442)
  
* DisMASTD: "Dismastd: An efficient distributed multi-aspect streaming tensor decomposition", in **IEEE ICDE**, 2021, [Paper](https://ieeexplore.ieee.org/document/9458848)


## Dynamic Tensor Analysis under Tucker/HOSVD format

### Online Tensor Dictionary Learning
#### Incremental Subspace Learning on Tensor Unfolding Matrices

* DTA and STA: "Beyond streams and graphs: dynamic tensor analysis", in **ACM SIGKDD**, 2007, [Paper](https://dl.acm.org/doi/abs/10.1145/1150402.1150445), [Code](https://www.sunlab.org/software)

* IRTSA: "Robust visual tracking based on incremental tensor subspace learning", in **IEEE ICCV**, 2007, [Paper](https://ieeexplore.ieee.org/abstract/document/4408950), [M-Code](https://www.cs.toronto.edu/~dross/ivt/), [P-Code](https://github.com/matkovst/IncrementalVisualTracker-python)

* RTSL: "A tensor-based approach for big data representation and dimensionality reduction", in **Int. J. Mach. Learn. Cybern**, 2011, [Paper](https://link.springer.com/article/10.1007/s13042-011-0017-0)
  
* ITF: "An incremental tensor factorization approach for web service recommendation", in **IEEE ICDM Works**, 2014,  [Paper](https://ieeexplore.ieee.org/abstract/document/7022617)

* IHOSVD: "A tensor-based approach for big data representation and dimensionality reduction", in **IEEE Trans. Emerg. Topics Comput.**, 2014,  [Paper](https://ieeexplore.ieee.org/abstract/document/6832490)

* Ho-RLSL: "Recursive tensor subspace tracking for dynamic brain network analysis", in **IEEE Trans. Signal Inf. Process. Netw.**, 2017, [Paper](https://ieeexplore.ieee.org/ielaam/6884276/8098622/7852497-aam.pdf)

* DHOSVD: "A distributed HOSVD method with its incremental computation for big data in cyber-physical-social systems", in **IEEE Trans. Comput. Social Syst.**, 2018, [Paper](https://ieeexplore.ieee.org/abstract/document/8357923)

* MDHOSVD: "  A multi-order distributed HOSVD with its incremental computing for big services in cyber-physical-social systems", in **IEEE Trans. Big Data**, 2018, [Paper](https://ieeexplore.ieee.org/abstract/document/8333789)

* IMDHOSVD: "Improved multi-order distributed HOSVD with its incremental computing for smart city services", in **IEEE Trans. Sustainable Comput.**, 2018, [Paper](https://ieeexplore.ieee.org/abstract/document/8536482)



#### Online Multimodal Dictionary Learning


* OTDL: "Online multimodal dictionary learning", in **NeuroComputing**, 2019, [Paper](https://www.sciencedirect.com/science/article/pii/S0925231219311919)

* ODL: "Learning separable dictionaries for sparse tensor representation: An online approach", in **IEEE Trans.  Circuits Syst. II**, 2019, [Paper](https://ieeexplore.ieee.org/abstract/document/8424902)

* ORLTM: "Online robust low-rank tensor modeling for streaming data analysis", in **IEEE Trans. Neural Netw. Learn. Syst.**, 2019, [Paper](https://ieeexplore.ieee.org/abstract/document/8440682)

* OLRTR: "Streaming data preprocessing via online tensor recovery for large environmental sensor networks", in **ACM Trans. Knowl. Disc. Data**, 2022, [Paper](https://dl.acm.org/doi/abs/10.1145/3532189), [Code](https://github.com/yuehu9/Online_Robust_Tensor_Recovery)

* D-L1-Tucker: "Dynamic L1-norm Tucker tensor decomposition", in **IEEE J. Sel. Topics Signal Process.**, 2021, [Paper](https://ieeexplore.ieee.org/abstract/document/9358012), [Code](https://github.com/dgchachlakis/L1-norm-Tucker-Tensor-Decomposition)

* ROLTD: "Robust Online Tucker Dictionary Learning from Multidimensional Data Streams", in **APSIPA-ASC**, 2022, [Paper](https://ieeexplore.ieee.org/abstract/document/9980029), [Code](https://github.com/thanhtbt/ROTDL)


### Tensor Subspace Tracking 

* LRUT: "Accelerated low-rank updates to tensor decompositions", in **IEEE HPEC**, 2016, [Paper](https://ieeexplore.ieee.org/abstract/document/7761607)

* Riemannian-based method: "Low-rank tensor completion: a Riemannian manifold preconditioning approach", in **ICML**, 2016, [Paper](https://proceedings.mlr.press/v48/kasai16.html), [Code](https://bamdevmishra.in/codes/tensorcompletion/)

* BASS-Tucker: "Bayesian streaming sparse Tucker decomposition", in **UAI**, 2021, [Paper](https://proceedings.mlr.press/v161/fang21b.html), [Code](https://github.com/xuangu-fang/Bayesian-streaming-sparse-tucker)

* RT-NTD and BK-NTD: " Incremental nonnegative tucker decomposition with block-coordinate descent and recursive approaches", in **Symmetry**, 2022, [Paper](https://www.mdpi.com/2073-8994/14/1/113), [Code](https://github.com/RafalZdunek/Incremental-NTD)
 
* ATD: "Tracking online low-rank approximations of higher-order incomplete streaming tensors", in **Cell Patterns**, 2023,  [Paper](https://www.sciencedirect.com/science/article/pii/S2666389923001046), [Code](https://github.com/thanhtbt/tensor_tracking)

### Multi-aspect Streaming Tucker Decomposition

* SIITA: "Inductive Framework for Multi-Aspect Streaming Tensor Completion with Side Information", in **ACM CIKM**, 2018, [Paper](https://dl.acm.org/doi/abs/10.1145/3269206.3271713), [Code](https://github.com/madhavcsa/SIITA)

* eOTD: "eOTD: An efficient online tucker decomposition for higher order tensors", in **IEEE ICDM**, 2018, [Paper](https://ieeexplore.ieee.org/abstract/document/8594989)
  

## Dynamic Tensor Analysis Under Tensor-Train Format

### Dynamic Decomposition of Time-series Tensors with Fix-Size (Non-Streaming)

* DATT: "Dynamical approximation by hierarchical Tucker and tensor-train tensors", in **SIAM J. Matrix Anal. Appl.**, 2013, [Paper](https://epubs.siam.org/doi/abs/10.1137/120885723)

* DATT: "Time integration of tensor trains", in **SIAM J. Numer. Anal.**, 2015, [Paper](https://epubs.siam.org/doi/abs/10.1137/140976546)


### Incremental Decomposition of Tensors in Stationary Environments (i.e., TT-Cores are Fixed Over Time)

* ITTD: "An incremental tensor-train decomposition for cyber-physical-social big data", in **IEEE Trans. Big Data**, 2018,  [Paper](https://ieeexplore.ieee.org/abstract/document/8449102)

* DTT: "DTT: A highly efficient distributed tensortrain decomposition method for IIoT big data", in **IEEE Trans. Ind. Inf**, 2021, [Paper](https://ieeexplore.ieee.org/document/8963751)

 ### Streaming Decomposition of Tensors in Non-Stationary Environments (i.e., TT-Cores Can Change Over Time)

* TT-FOA: "Adaptive Algorithms for Tracking Tensor-Train Decomposition of Streaming Tensors", in **EUSIPCO**, 2020,  [Paper](https://ieeexplore.ieee.org/document/9287780), [Code](https://github.com/thanhtbt/ATT)

* ROBOT: "Robust Tensor Tracking With Missing Data Under Tensor-Train Format", in **EUSIPCO**, 2022,  [Paper](https://ieeexplore.ieee.org/document/9287780), [Code](https://github.com/thanhtbt/ROBOT)

* TT-ICE: "An Incremental Tensor Train Decomposition Algorithm", in ArXiv, 2022. [Paper](https://arxiv.org/pdf/2211.12487.pdf), [Code](https://github.com/dorukaks/TT-ICE)
  
* ATT: "A Novel Recursive Least-Squares Adaptive Method For Streaming Tensor-Train Decomposition With Incomplete Observations", in **Elsevier Signal Process.**, 2023, [Paper](https://www.sciencedirect.com/science/article/pii/S0165168423003717), [Code](https://github.com/thanhtbt/ATT-miss)

* STTA: "Streaming tensor train approximation", in **SIAM J. Sci. Comput.**, 2023, [Paper](https://epubs.siam.org/doi/abs/10.1137/22M1515045), [Code](https://github.com/RikVoorhaar/tt-sketch)


## Dynamic Tensor Analysis Under Block-Term Decomposition Format


* OnlineBTD: "OnlineBTD: Streaming algorithms to track the block term decomposition of large tensors", in **DSAA**, 2020, [Paper](https://ieeexplore.ieee.org/abstract/document/9260061), [Code](http://www.cs.ucr.edu/~egujr001/ucr/madlab/src/OnlineBTD.zip)
  
* O-BTD-RLS: "Online rank-revealing block-term tensor decomposition", in **Elsevier Signal Process.**, 2023, [Paper](https://www.sciencedirect.com/science/article/pii/S0165168423002001)

* SBTD: "A Novel Tensor Tracking Algorithm For Block-Term Decomposition of Streaming Tensors", in **IEEE SSP**, 2023, [Paper](https://ieeexplore.ieee.org/document/10208007)


## Dynamic Tensor Analysis Under T-SVD Format

* TO-RPCA: "An online tensor robust PCA algorithm for sequential 2D data", in **IEEE ICASSP**, 2016, [Paper](https://ieeexplore.ieee.org/document/7472114)
  
* TOUCAN: "Grassmannian optimization for online tensor completion and tracking with the t-svd", in **IEEE Trans. Signal Process.**, 2022, [Paper](https://ieeexplore.ieee.org/abstract/document/9756209), [Code](https://web.eecs.umich.edu/~girasole/?p=676)

## Dynamic Tensor Analysis Under Tensor-Ring/Tensor Network Format

* STR:  "Tracking Tensor Ring Decompositions of Streaming Tensors", in ArXiv, 2023, [Paper](https://arxiv.org/pdf/2307.00719.pdf)

## Related Sources
#### Good Surveys of (Batch) Tensor  Decomposition and Analysis

* "Unsupervised multiway data analysis: A literature survey", in **IEEE TKDE**, 2008, [Paper](https://ieeexplore.ieee.org/abstract/document/4538221)
* "Tensor decompositions and applications", in **SIAM Rev.**, 2009, [Paper](https://epubs.siam.org/doi/10.1137/07070111X)
* "Breaking the curse of dimensionality using decompositions of incomplete tensors: Tensor-based scientific computing in
big data analysis", in **IEEE Signal Process. Mag.**, 2014, [Paper](https://ieeexplore.ieee.org/document/6879619)
* "Tensor networks for dimensionality reduction and large-scale optimization: Part 1 low-rank tensor decompositions", in **Found. Trends Mach. Learn.**, 2016, [Paper](https://www.nowpublishers.com/article/Details/MAL-059)
* "Tensor decompositions for signal processing applications: From two-way to multiway component analysis", in **IEEE Signal Process. Mag.**, 2009, [Paper](https://ieeexplore.ieee.org/abstract/document/7038247)
* "Tensor Decomposition for Signal Processing and Machine Learning", in **IEEE Trans. Signal Process.**, 2009, [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7891546)


 
## Citation
If you find this repository helpful for your work, please cite

[1] L.T. Thanh, K. Abed-Meraim, N. L. Trung and A. Hafiane. “[*A Contemporary and Comprehensive Survey on Streaming Tensor Decomposition*](https://ieeexplore.ieee.org/document/9994046)”. **IEEE Trans. Knowl. Data Eng., 2023**. [PDF](https://thanhtbt.github.io/files/2022_TKDE_A%20Contemporary%20and%20Comprehensive%20Survey%20on%20Streaming%20Tensor%20Decomposition.pdf).

