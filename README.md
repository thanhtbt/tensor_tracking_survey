## Tensor Tracking, Streaming/Online/Adaptive/Incremental Tensor Decomposition and Dynamic Tensor Analysis

A list of up-to-date research papers on streaming tensor decomposition, tensor tracking, and dynamic tensor analysis.

*Contributions to improve the completeness of this list are greatly appreciated. Please feel free to open an issue, make a pull request, contribute, and revise (or contact [me](https://thanhtbt.github.io/)) if you find any missed references or errors*.

## Table of Content

 - [Survey and Overview Paper](#survey-paper)
- [Dynamic Tensor Analysis Under CP/PARAFAC Format](#dynamic-tensor-analysis-under-cp-parafac-format)
  * [Subspace-based Methods](#subspace-based-methods)
  * [Block-Coordinate Descent (BCD)](#block-coordinate-descent--bcd-)
    + [BCD + Adaptive Least-Squares Filters](#bcd---adaptive-least-squares-filters)
    + [BCD + Stochastic Gradient Solvers](#bcd---stochastic-gradient-solvers)
    + [BCD + ADMM](#bcd---admm)
  * [Bayesian Inference](#bayesian-inference)
  * [Multi-aspect Streaming CP Decomposition](#multi-aspect-streaming-cp-decomposition)
  * [Streaming PARAFAC2 Decomposition](#streaming-parafac2-decomposition)
- [Dynamic Tensor Analysis Under Tucker/HOSVD Format](#dynamic-tensor-analysis-under-tucker-hosvd-format)
  * [Online Tensor Dictionary Learning](#online-tensor-dictionary-learning)
    + [Incremental Subspace Learning on Tensor Unfolding Matrices](#incremental-subspace-learning-on-tensor-unfolding-matrices)
    + [Online Multimodal Dictionary Learning](#online-multimodal-dictionary-learning)
  * [Tensor Subspace Tracking](#tensor-subspace-tracking)
  * [Bayesian Inference](#bayesian-inference)
  * [Multi-aspect Streaming Tucker Decomposition](#multi-aspect-streaming-tucker-decomposition)
- [Dynamic Tensor Analysis Under Tensor-Train Format](#dynamic-tensor-analysis-under-tensor-train-format)
  * [Dynamic Decomposition of Time-series Tensors with Fix-Size](#dynamic-decomposition-of-time-series-tensors-with-fix-size--non-streaming-)
  * [Incremental Decomposition of Tensors in Stationary Environments](#incremental-decomposition-of-tensors-in-stationary-environments--ie--tt-cores-are-fixed-over-time-)
  * [Streaming Decomposition of Tensors in Non-Stationary Environments](#streaming-decomposition-of-tensors-in-non-stationary-environments--ie--tt-cores-can-change-over-time-)
- [Dynamic Tensor Analysis Under Block-Term Decomposition Format](#dynamic-tensor-analysis-under-block-term-decomposition-format)
- [Dynamic Tensor Analysis Under T-SVD Format](#dynamic-tensor-analysis-under-t-svd-format)
- [Dynamic Tensor Analysis Under Tensor-Ring/Tensor Network Format](#dynamic-tensor-analysis-under-tensor-ring-tensor-network-format)
- [Dynamic/Adaptive Coupled Tensor Decomposition](#dynamic-coupled-tensor-decomposition)

- [Research Challenges, Open Problems, and Future Directions](#challenges)
- [Related Sources](#related-sources)
  * [Good Surveys of (Batch) Tensor  Decomposition and Analysis](#good-surveys-of--batch--tensor--decomposition-and-analysis)
  * [Tensor Toolbox and Software](#tensor-toolbox-and-software)
    + [MATLAB](#matlab)
    + [Python](#python)
    + [R](#r)
    + [Julia](#julia)
   * [Some Applications and Datasets](#applications-datasets)
- [Citation](#citation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'></a></i></small>


## [Survey and Overview Paper](#survey-paper) 
[*A Contemporary and Comprehensive Survey on Streaming Tensor Decomposition*](https://ieeexplore.ieee.org/document/9994046), in **IEEE TKDE**, 2023, [PDF](https://thanhtbt.github.io/files/2022_TKDE_A%20Contemporary%20and%20Comprehensive%20Survey%20on%20Streaming%20Tensor%20Decomposition.pdf)

Authors: [Thanh Trung Le](https://thanhtbt.github.io/), Karim Abed-Meraim, Nguyen Linh Trung and Adel Hafiane

![tensor_tracking](https://github.com/thanhtbt/tensor_tracking_survey/assets/26319211/6219b4d6-ce85-47ce-aa51-748661560a99)


## [Dynamic Tensor Analysis Under CP/PARAFAC Format](#dynamic-tensor-analysis-under-cp-parafac-format)
### [Subspace-based Methods](#content) 

* **PARAFAC-SDT/-RLS**: “Adaptive algorithms to track the PARAFAC decomposition of a third-order tensor,” in **IEEE Trans. Signal Process.**, 2009, [Paper](https://ieeexplore.ieee.org/document/4799120), [PDF](https://www.ece.umn.edu/~nikos/AdaptivePARAFAC.pdf), [Code](https://github.com/thanhtbt/ROLCP/tree/main/Algorithms/PAFARAC(2009))

* **3D-OPAST**: ``Fast adaptive PARAFAC decomposition algorithm with linear complexity", in **IEEE ICASSP**, 2016, [Paper](https://ieeexplore.ieee.org/document/7472876), [PDF](https://inria.hal.science/hal-01306461/file/3DOPASTLinear.pdf)

* **CP-PETRELS**: ``Adaptive PARAFAC decomposition for third-order tensor completion", in **IEEE ICCE**, 2016, [Paper](https://ieeexplore.ieee.org/document/7562652), [PDF](http://eprints.uet.vnu.edu.vn/eprints/id/document/560)

* **SOAP**: "Second-order optimization based adaptive PARAFAC decomposition of three-way tensors", in **Digital Signal Process.**, 2017, [Paper](https://www.sciencedirect.com/science/article/pii/S105120041730009X), [Code](https://drive.google.com/drive/folders/1x6PdEsr-1xDccm7titi5dQPLwgcKuOii)

### [Block-Coordinate Descent (BCD) Methods](#content) 
#### [BCD + Adaptive Least-Squares Filters](#content) 

* **OLSTEC**: "Fast online low-rank tensor subspace tracking by CP decomposition using recursive least squares from incomplete observations", in **NeuroComput.**, 2017,  [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231218313584), [PDF](https://arxiv.org/pdf/1709.10276.pdf), [Code](https://github.com/hiroyuki-kasai/OLSTEC)
   * Conference version: "Online low-rank tensor subspace tracking from incomplete data by CP decomposition using recursive least squares", in **IEEE ICASSP**, 2016 [Paper](https://ieeexplore.ieee.org/abstract/document/7472131), [PDF](https://arxiv.org/pdf/1602.07067.pdf), [Code](https://github.com/hiroyuki-kasai/OLSTEC)

* **CP-NLS**: "Nonlinear least squares updating of the canonical polyadic decomposition", in **EUSIPCO**, 2017, [Paper](https://ieeexplore.ieee.org/document/8081290), [PDF](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2017/papers/1570347577.pdf), [Code](https://www.tensorlabplus.net/papers/vandecappelle2017cpdu.html)

* **CP-stream**: "Streaming tensor factorization for infinite data sources", in **SDM**, 2018, [Paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611975321.10), [PDF](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975321.10), [Code in Splatt Toolbox](https://github.com/ShadenSmith/splatt)

* **InParTen**: "Incremental PARAFAC decomposition for three-dimensional tensors using Apache Spark", in **ICWE**, 2019, [Paper](https://link.springer.com/chapter/10.1007/978-3-030-19274-7_5)

* **TenNOODL**: "Provable online CP/PARAFAC decomposition of a structured tensor via dictionary learning", in **NeurISP**, 2021, [Paper](https://dl.acm.org/doi/10.5555/3495724.3496695), [PDF](https://proceedings.neurips.cc/paper/2020/file/85b42dd8aae56e01379be5736db5b496-Paper.pdf), [Code](https://github.com/srambhatla/TensorNOODL)

* **SliceNStitch**: "Slicenstitch: Continuous CP decomposition of sparse tensor streams", in **IEEE ICDE**, 2021, [Paper](https://ieeexplore.ieee.org/abstract/document/9458693), [PDF](https://arxiv.org/pdf/2102.11517.pdf), [Code](https://github.com/DMLab-Tensor/SliceNStitch)

* **STF**: "Accurate online tensor factorization for temporal tensor streams with missing value", in **ACM CIKM**, 2021, [Paper](https://dl.acm.org/doi/abs/10.1145/3459637.3482048), [PDF](https://datalab.snu.ac.kr/~ukang/papers/stfCIKM21.pdf), [Code](https://github.com/snudatalab/STF)
  
* **ROLCP**: "A Fast Randomized Adaptive CP Decomposition for Streaming Tensors", in **IEEE ICASSP**, 2021,  [Paper](https://ieeexplore.ieee.org/abstract/document/9413554), [PDF](https://thanhtbt.github.io/files/2021_ICASSP%20-%20Randomized%20Adaptive%20CP%20Algorithm.pdf), [Code](https://github.com/thanhtbt/ROLCP)

* **OnlineCPDL**: "Online nonnegative CP-dictionary learning for Markovian data" in **J. Mach. Learn. Res.**, 2022,  [PDF](https://www.jmlr.org/papers/volume23/21-0419/21-0419.pdf), [Code](https://github.com/HanbaekLyu/OnlineCPDL)
   
* **ACP**: "Tracking online low-rank approximations of higher-order incomplete streaming tensors", in **Cell Patterns**, 2023,  [Paper](https://www.sciencedirect.com/science/article/pii/S2666389923001046), [PDF](https://thanhtbt.github.io/files/2023_Patterns_Tensor_Tracking_Draw.pdf), [Code](https://github.com/thanhtbt/tensor_tracking)

* **ALTO**: "Dynamic Tensor Linearization and Time Slicing for Efficient Factorization of Infinite Data Streams", in **IEEE IPDPS**, 2023,  [Paper](https://ieeexplore.ieee.org/abstract/document/10177430), [Code](https://github.com/jeewhanchoi/ALTO-stream)

* **OnlineGCP**: "Streaming Generalized Canonical Polyadic Tensor Decompositions", in **PASC**, 2023,  [Paper](https://dl.acm.org/doi/abs/10.1145/3592979.3593405), [PDF](https://arxiv.org/pdf/2110.14514.pdf), [Code](https://gitlab.com/tensors/genten)


* **Online CP**: "A Tensor Based Unified Framework for Streaming Signal Online Analysis", in **IEEE SPL**, 2024,  [Paper](https://ieeexplore.ieee.org/abstract/document/10555149)
  



#### [BCD + Stochastic Gradient Solvers](#content) 

* **TeCPSGD**: "Subspace Learning and Imputation for Streaming Big Data Matrices and Tensors", in **IEEE Trans. Signal Process.**, 2015, [Paper](https://ieeexplore.ieee.org/document/7072498), [PDF](https://arxiv.org/pdf/1404.4667.pdf), [Code](https://github.com/hiroyuki-kasai/OLSTEC/tree/master/benchmark/TeCPSGD)

* **OLCP**: "Accelerating Online CP Decompositions for Higher Order Tensors", in **ACM SIGKDD**, 2016, [Paper](https://dl.acm.org/doi/abs/10.1145/2939672.2939763), [PDF](https://www.kdd.org/kdd2016/papers/files/rfp0403-zhouA.pdf), [Code](https://shuozhou.github.io/)

* **OnlineSCP**: "Online CP Decomposition for Sparse Tensors", in **IEEE ICDM**, 2018, [Paper](https://ieeexplore.ieee.org/abstract/document/8595011), [PDF](https://shuozhou.github.io/papers/shuo18icdm_short.pdf), [Code](https://shuozhou.github.io/)

* **SOFIA**: "Robust Factorization of Real-world Tensor Streams with Patterns, Missing Values, and Outliers", in **IEEE ICDE**, 2020, [Paper](https://ieeexplore.ieee.org/abstract/document/9458640), [PDF](https://arxiv.org/pdf/2102.08466.pdf?trk=public_post_comment-text), [Code](https://github.com/wooner49/sofia)

* **iCP-AM**: "Incremental CP tensor decomposition by alternating minimization method", in **SIAM J. Matrix Anal. Appl**, 2020, [Paper](https://epubs.siam.org/doi/abs/10.1137/20M1319097) 

* **DAO-CP**: "DAO-CP: Data Adaptive Online CP Decomposition", in **Plus One**, 2021, [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9009670/pdf/pone.0267091.pdf), [PDF](https://datalab.snu.ac.kr/dao-cp/resources/paper.pdf), [Code](https://github.com/lucetre-snu/dao-cp)

* **CIDTD**: "Scale-variant structural feature construction of EEG stream via component-increased Dynamic Tensor Decomposition", in **Knowledge-Based Systems**, 2024, [Paper](https://www.sciencedirect.com/science/article/pii/S0950705124003824#bib1)

#### [BCD + ADMM](#content) 

* **spCP-stream**:  "High-Performance Streaming Tensor Decomposition", in **IEEE IPDPS**, 2021, [Paper](https://ieeexplore.ieee.org/document/9460519), [PDF](https://www.cs.uoregon.edu/Reports/DRP-202106-Soh.pdf), [Code](https://github.com/jeewhanchoi/row-sparse-cpstream)
  
* **RACP**: "Robust Tensor Tracking with Missing Data and Outliers: Novel Adaptive CP Decomposition and Convergence Analysis" in **IEEE Trans. Signal Process.**, 2022, [Paper](https://ieeexplore.ieee.org/document/9866940), [PDF](https://thanhtbt.github.io/files/2022_TSP_RACP%20(Raw).pdf)
  
* **T-MUST**: "Robust online tensor completion for IoT streaming data recovery", in **IEEE Trans. Neural Netw. Learn. Syst.**, 2022, [Paper](https://ieeexplore.ieee.org/abstract/document/9758937)

### [Bayesian Inference Methods](#content) 

* **POST**: "Probabilistic streaming tensor decomposition", in **IEEE ICDM**, 2018,  [Paper](https://ieeexplore.ieee.org/abstract/document/8594834), [PDF](https://users.cs.utah.edu/~zhe/pdf/POST.pdf), [Code](https://github.com/yishuaidu/POST)

* **BRST**: "Variational Bayesian inference for robust streaming tensor factorization and completion", in **IEEE ICDM**, 2018,  [Paper](https://ieeexplore.ieee.org/abstract/document/8594834), [PDF](https://arxiv.org/pdf/1809.02153.pdf), [Code](https://github.com/colehawkins/Robust-Streaming-Tensor-Factorization)

* **SBDT**: "Streaming Bayesian deep tensor factorization", in **ICML**, 2021, [Paper](https://proceedings.mlr.press/v139/fang21d.html), [PDF](https://proceedings.mlr.press/v139/fang21d/fang21d.pdf), [Code](https://github.com/xuangu-fang/Streaming-Bayesian-Deep-Tensor)

* **SFTL**: "Streaming Factor Trajectory Learning for Temporal Tensor Decomposition", in **NeurIPS**, 2023, [Paper](https://neurips.cc/virtual/2023/poster/71689), [PDF](https://proceedings.mlr.press/v139/fang21d/fang21d.pdf), [PDF](https://arxiv.org/pdf/2310.17021.pdf), [Code](https://github.com/xuangu-fang/Streaming-Factor-Trajectory-Learning)

### [Multi-aspect Streaming CP Decomposition Methods](#content) 

* **MASTA**: "Multi-aspect-streaming tensor analysis", in **Knowl.-Based Syst.**, 2015, [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0950705115002671), [PDF](https://repositorio.inesctec.pt/server/api/core/bitstreams/ca4786ca-278c-417a-8bbc-7f00b511ecfd/content), [Code](https://github.com/fanaee/MASTA/blob/main/MASTA.zip)

* **MAST**: "Multi-aspect streaming tensor completion", in **ACM SIGKDD**, 2017, [Paper](https://dl.acm.org/doi/pdf/10.1145/3097983.3098007), [PDF](http://library.usc.edu.ph/ACM/KKD%202017/pdfs/p435.pdf), [Code](https://github.com/xuangu-fang/Streaming-Bayesian-Deep-Tensor) 

* **OR-MSTC**: "Outlier-Robust Multi-Aspect Streaming Tensor Completion and Factorization", in **IJCAI**, 2019, [Paper](https://www.ijcai.org/proceedings/2019/442), [PDF](https://www.researchgate.net/profile/Lifang-He/publication/334843950_Outlier-Robust_Multi-Aspect_Streaming_Tensor_Completion_and_Factorization/links/5e1ff98b458515ba208a85c8/Outlier-Robust-Multi-Aspect-Streaming-Tensor-Completion-and-Factorization.pdf)

* **InParTen2**:  "Multi-aspect incremental tensor decomposition based on distributed in-memory big data systems", in **J. Data Inf. Sci.**, 2020, [Paper](https://www.ijcai.org/proceedings/2019/442)
  
* **DisMASTD**: "Dismastd: An efficient distributed multi-aspect streaming tensor decomposition", in **IEEE ICDE**, 2021, [Paper](https://ieeexplore.ieee.org/document/9458848), [PDF](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=7127&context=sis_research)


* **GOCPT**: "GOCPT: Generalized Online Canonical Polyadic Tensor Factorization and Completion", in **IJCAI**, 2022, [Paper](https://www.ijcai.org/proceedings/2022/0326.pdf), [PDF](https://arxiv.org/pdf/2205.03749.pdf), [Code](https://github.com/ycq091044/GOCPT)

### [Streaming PARAFAC2 Decomposition Methods](#content) 

* **SPADE**: "SPADE: Streaming PARAFAC2 decomposition for large datasets", in **SDM**, 2020, [Paper](https://epubs.siam.org/doi/abs/10.1137/1.9781611976236.65), [PDF](https://madlab.cs.ucr.edu/papers/2020_SDM_SPADE.pdf), [Code](http://www.cs.ucr.edu/~egujr001/ucr/madlab/src/SPADE.zip)

* **Dpar2**: "Dpar2: Fast and scalable parafac2 decomposition for irregular dense tensors", in **IEEE ICDE**, 2022, [Paper](https://ieeexplore.ieee.org/document/9835294), [PDF](https://arxiv.org/pdf/2203.12798.pdf), [Code](https://datalab.snu.ac.kr/dpar2/)

* **ATOM**: "Accurate PARAFAC2 Decomposition for Temporal Irregular Tensors with Missing Values", in **IEEE BigData**, 2022, [Paper](https://ieeexplore.ieee.org/document/10020667), [PDF](https://jungijang.github.io/resources/2022/BigData/atom.pdf), [Code](https://datalab.snu.ac.kr/atom/)
  
* **DASH**: "Fast and Accurate Dual-Way Streaming PARAFAC2 for Irregular Tensors--Algorithm and Application", in **ACM SIGKDD**, 2023, [Paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599342), [PDF](https://arxiv.org/pdf/2305.18376.pdf), [Code](https://github.com/snudatalab/Dash)

* **tPARAFAC2**: "A Time-aware tensor decomposition for tracking evolving patterns", in **Proc. MLSP**, 2023, [Paper](https://arxiv.org/pdf/2308.07126.pdf), [Code](https://github.com/cchatzis/tPARAFAC2)

### [Others](#content) 

* **DEMOTE**: "Dynamic Tensor Decomposition via Neural Diffusion-Reaction Processes", in **NeurIPS**, 2023, [Paper](https://neurips.cc/virtual/2023/poster/71967), [PDF](https://arxiv.org/pdf/2310.19666.pdf), [Code](https://github.com/wzhut/Dynamic-Tensor-Decomposition-via-Neural-Diffusion-Reaction-Processes)


## [Dynamic Tensor Analysis Under Tucker/HOSVD Format](#content) 

### [Online Tensor Dictionary Learning Methods](#content) 
#### [Incremental Subspace Learning on Tensor Unfolding Matrices](#content) 

* **DTA/STA**: "Beyond streams and graphs: dynamic tensor analysis", in **ACM SIGKDD**, 2007, [Paper](https://dl.acm.org/doi/abs/10.1145/1150402.1150445), [PDF](https://www.cs.cmu.edu/~christos/PUBLICATIONS/kdd06DTA.pdf), [TKDD](https://doi.org/10.1145/1409620.1409621), [Code](https://www.sunlab.org/software)

* **IRTSA**: "Robust visual tracking based on incremental tensor subspace learning", in **IEEE ICCV**, 2007, [Paper](https://ieeexplore.ieee.org/abstract/document/4408950), [PDF](https://web.archive.org/web/20170811190623id_/http://www.isee.zju.edu.cn/dsec/pdf/iccv07_final.pdf), [IJCV](https://link.springer.com/content/pdf/10.1007/s11263-010-0399-6.pdf), [M-Code](https://www.cs.toronto.edu/~dross/ivt/), [P-Code](https://github.com/matkovst/IncrementalVisualTracker-python)

* **RTSL**: "Robust tensor subspace learning for anomaly detection", in **Int. J. Mach. Learn. Cybern**, 2011, [Paper](https://link.springer.com/article/10.1007/s13042-011-0017-0), [PDF](https://d1wqtxts1xzle7.cloudfront.net/44758963/Robust_tensor_subspace_learning_for_anom20160415-2942-1k05299-libre.pdf?1460719842=&response-content-disposition=inline%3B+filename%3DRobust_tensor_subspace_learning_for_anom.pdf&Expires=1699913109&Signature=GuYaEuc17BoLAtxz8fQHYrYCEurs8yyvb68Hq2jeqCI9xiIA45AwJ8Zpuv~L-QDBRdHFR7OB9eaeB93kRL2Kyk9gbUIBvkle4BeFKZkHT-451vXu5wgVVdBdCNXS5iLQaU2moCFPZ6RH57y1C9R3cGMhG8Gnw4yuFq~maDBmAThTKxbSiXK-yzBZ47TtI4KPKcylq50TfrsqzNxmHMXM7kCY8qXk0KYVhmbPM9NcwSlr-iLAfNQJUBBFtXknsrTQbxqckmar4OZQ-wmzW4UO5C31bx84u39Fi2qd2DhvHEeG6Bbz5UB310gDrbptV0gWvm48tTQdb1Yh0ziusuvMqw__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
  
* **ITF**: "An incremental tensor factorization approach for web service recommendation", in **IEEE ICDM Works**, 2014,  [Paper](https://ieeexplore.ieee.org/abstract/document/7022617)

* **Online-LRTL**: "Accelerated online low rank tensor learning for multivariate spatiotemporal streams", in **ICML**, 2015, [Paper](https://proceedings.mlr.press/v37/yua15.html), [PDF](https://proceedings.mlr.press/v37/yua15.pdf), [Matlab](https://roseyu.com/code.html), [Python](https://nbviewer.org/github/xinychen/tensor-learning/blob/master/baselines/Online-LRTL.ipynb)

* **IHOSVD**: "A tensor-based approach for big data representation and dimensionality reduction", in **IEEE Trans. Emerg. Topics Comput.**, 2014,  [Paper](https://ieeexplore.ieee.org/abstract/document/6832490), [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6832490)

* **Ho-RLSL**: "Recursive tensor subspace tracking for dynamic brain network analysis", in **IEEE Trans. Signal Inf. Process. Netw.**, 2017, [Paper](https://ieeexplore.ieee.org/abstract/document/7852497), [PDF](https://ieeexplore.ieee.org/ielaam/6884276/8098622/7852497-aam.pdf)

* **DHOSVD**: "A distributed HOSVD method with its incremental computation for big data in cyber-physical-social systems", in **IEEE Trans. Comput. Social Syst.**, 2018, [Paper](https://ieeexplore.ieee.org/abstract/document/8357923), [PDF](https://www.researchgate.net/profile/Liu-Huazhong/publication/327298675_An_Incremental_Tensor-Train_Decomposition_for_Cyber-Physical-Social_Big_Data/links/5cc8e492a6fdcc1d49bbfe40/An-Incremental-Tensor-Train-Decomposition-for-Cyber-Physical-Social-Big-Data.pdf)

* **MDHOSVD**: "A multi-order distributed HOSVD with its incremental computing for big services in cyber-physical-social systems", in **IEEE Trans. Big Data**, 2018, [Paper](https://ieeexplore.ieee.org/abstract/document/8333789)

* **IMDHOSVD**: "Improved multi-order distributed HOSVD with its incremental computing for smart city services", in **IEEE Trans. Sustain. Comput.**, 2018, [Paper](https://ieeexplore.ieee.org/abstract/document/8536482)

* **SMLN**: "A sequential multilinear Nystrom" om algorithm for streaming low-rank approximation of tensors in Tucker format",  in **Applied Mathematics Letters**, 2024 [Paper](https://www.sciencedirect.com/science/article/abs/pii/S089396592400291X?via%3Dihub)


#### [Online Multimodal Dictionary Learning](#content) 

* **Singleshot**: "Singleshot: A scalable Tucker tensor decomposition", in **NeurIPS**, 2019, [Paper](https://proceedings.neurips.cc/paper/2019/hash/384babc3e7faa44cf1ca671b74499c3b-Abstract.html), [PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/384babc3e7faa44cf1ca671b74499c3b-Paper.pdf)

* **OTDL**: "Online multimodal dictionary learning", in **NeuroComput.**, 2019, [Paper](https://www.sciencedirect.com/science/article/pii/S0925231219311919), [PDF](https://www.researchgate.net/profile/Traore-Abraham/publication/335445803_Online_Multimodal_Dictionary_Learning/links/5ddca6c7458515dc2f4dd091/Online-Multimodal-Dictionary-Learning.pdf)

* **ODL**: "Learning separable dictionaries for sparse tensor representation: An online approach", in **IEEE Trans.  Circuits Syst. II**, 2019, [Paper](https://ieeexplore.ieee.org/abstract/document/8424902)

* **ORLTM**: "Online robust low-rank tensor modeling for streaming data analysis", in **IEEE Trans. Neural Netw. Learn. Syst.**, 2019, [Paper](https://ieeexplore.ieee.org/abstract/document/8440682)

* **TTMTS**: "Low-rank Tucker approximation of a tensor from streaming data", in **SIAM J. Math. Data Sci.**, 2020, [Paper](https://epubs.siam.org/doi/abs/10.1137/19M1257718), [PDF](https://arxiv.org/abs/1904.10951), [Code](https://github.com/udellgroup/tensorsketch)

* **Zoom-Tucker**: "Fast and Memory-Efficient Tucker Decomposition for Answering Diverse Time Range Queries" in **ACM SIGKDD**, 2021, [Paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467290), [PDF](https://jungijang.github.io/resources/2021/KDD/zoomtucker.pdf), [Code](https://datalab.snu.ac.kr/zoomtucker/)


* **OLRTR**: "Streaming data preprocessing via online tensor recovery for large environmental sensor networks", in **ACM Trans. Knowl. Disc. Data**, 2022, [Paper](https://dl.acm.org/doi/abs/10.1145/3532189), [PDF](https://lab-work.github.io/download/HuStreaming2021.pdf), [Code](https://github.com/yuehu9/Online_Robust_Tensor_Recovery)

* **D-L1-Tucker**: "Dynamic L1-norm Tucker tensor decomposition", in **IEEE J. Sel. Topics Signal Process.**, 2021, [Paper](https://ieeexplore.ieee.org/abstract/document/9358012), [PDF](https://ieeexplore.ieee.org/ielaam/4200690/9393360/9358012-aam.pdf), [Code](https://github.com/dgchachlakis/L1-norm-Tucker-Tensor-Decomposition)

* **ROLTD**: "Robust Online Tucker Dictionary Learning from Multidimensional Data Streams", in **APSIPA-ASC**, 2022, [Paper](https://ieeexplore.ieee.org/abstract/document/9980029), [PDF](https://thanhtbt.github.io/files/2022_APSIPA_Robust%20Online%20Tucker%20Dictionary%20Learning%20from%20Multidimensional%20Data%20Streams.pdf), [Code](https://github.com/thanhtbt/ROTDL)


### [Tensor Subspace Tracking Methods](#content) 

* **LRUT**: "Accelerated low-rank updates to tensor decompositions", in **IEEE HPEC**, 2016, [Paper](https://ieeexplore.ieee.org/abstract/document/7761607), [PDF](http://www.harperlangston.com/papers/lowrank_hpec.pdf)

* **Riemannian-based method**: "Low-rank tensor completion: a Riemannian manifold preconditioning approach", in **ICML**, 2016, [Paper](https://proceedings.mlr.press/v48/kasai16.html), [PDF](https://proceedings.mlr.press/v48/kasai16.pdf), [Code](https://bamdevmishra.in/codes/tensorcompletion/)

* **RT-NTD and BK-NTD**: " Incremental nonnegative Tucker decomposition with block-coordinate descent and recursive approaches", in **Symmetry**, 2022, [Paper](https://www.mdpi.com/2073-8994/14/1/113), [Code](https://github.com/RafalZdunek/Incremental-NTD)

* **D-TuckerO**: "Static and Streaming Tucker Decomposition for Dense Tensors", in **ACM Trans. Knowl. Disc. Data**, 2022, [Paper](https://dl.acm.org/doi/10.1145/3568682), [PDF](https://datalab.snu.ac.kr/~ukang/papers/dtuckerTKDD23.pdf), [Code](https://datalab.snu.ac.kr/dtucker/)

* **ATD**: "Tracking online low-rank approximations of higher-order incomplete streaming tensors", in **Cell Patterns**, 2023,  [Paper](https://www.sciencedirect.com/science/article/pii/S2666389923001046), [PDF](https://thanhtbt.github.io/files/2023_Patterns_Tensor_Tracking_Draw.pdf), [Code](https://github.com/thanhtbt/tensor_tracking)

* **Online Tucker**: "A Tensor Based Unified Framework for Streaming Signal Online Analysis", in **IEEE SPL**, 2024,  [Paper](https://ieeexplore.ieee.org/abstract/document/10555149)
  

### [Bayesian Inference Methods](#content) 

* **SNBTD**: "Streaming nonlinear Bayesian tensor decomposition", in  **UAI**, 2020, [Paper](https://proceedings.mlr.press/v124/pan20a.html), [PDF](https://proceedings.mlr.press/v124/pan20a/pan20a.pdf), [Code](https://github.com/USTCEarthDefense/SNBTD)
  
* **BASS-Tucker**: "Bayesian streaming sparse Tucker decomposition", in **UAI**, 2021, [Paper](https://proceedings.mlr.press/v161/fang21b.html), [PDF](https://proceedings.mlr.press/v161/fang21b/fang21b.pdf), [Code](https://github.com/xuangu-fang/Bayesian-streaming-sparse-tucker)
  
* **BCTT**: "Bayesian Continuous-Time Tucker Decomposition", in **ICML**, 2022, [Paper](https://proceedings.mlr.press/v162/fang22b.html), [PDF](https://proceedings.mlr.press/v162/fang22b/fang22b.pdf), [Code](https://github.com/xuangu-fang/Bayesian-Continuous-Time-Tucker-Decomposition)

  
### [Multi-aspect Streaming Tucker Decomposition Methods](#content) 

* **SIITA**: "Inductive Framework for Multi-Aspect Streaming Tensor Completion with Side Information", in **ACM CIKM**, 2018, [Paper](https://dl.acm.org/doi/abs/10.1145/3269206.3271713), [PDF](https://pdfs.semanticscholar.org/99c8/b8d79ec419209513ee49274b4b7b537d3d73.pdf), [Code](https://github.com/madhavcsa/SIITA)

* **eOTD**: "eOTD: An efficient online tucker decomposition for higher order tensors", in **IEEE ICDM**, 2018, [Paper](https://ieeexplore.ieee.org/abstract/document/8594989), [PDF](https://par.nsf.gov/servlets/purl/10098518)
  

## [Dynamic Tensor Analysis Under Tensor-Train Format](#content) 

### [Dynamic Decomposition of Time-series Tensors with Fix-Size (Non-Streaming)](#content) 

* **DATT**: "Dynamical approximation by hierarchical Tucker and tensor-train tensors", in **SIAM J. Matrix Anal. Appl.**, 2013, [Paper](https://epubs.siam.org/doi/abs/10.1137/120885723), [PDF](https://www.researchgate.net/profile/Reinhold-Schneider/publication/264273502_Dynamical_Approximation_By_Hierarchical_Tucker_And_Tensor-Train_Tensors/links/54d261b40cf25017917dea17/Dynamical-Approximation-By-Hierarchical-Tucker-And-Tensor-Train-Tensors.pdf)

* **DATT**: "Time integration of tensor trains", in **SIAM J. Numer. Anal.**, 2015, [Paper](https://epubs.siam.org/doi/abs/10.1137/140976546), [PDF](https://arxiv.org/pdf/1407.2042.pdf)


### [Incremental Decomposition of Tensors in Stationary Environments (i.e., TT-Cores are Fixed Over Time)](#content) 

* **ITTD**: "An incremental tensor-train decomposition for cyber-physical-social big data", in **IEEE Trans. Big Data**, 2018,  [Paper](https://ieeexplore.ieee.org/abstract/document/8449102), [PDF](https://www.researchgate.net/profile/Liu-Huazhong/publication/327298675_An_Incremental_Tensor-Train_Decomposition_for_Cyber-Physical-Social_Big_Data/links/5cc8e492a6fdcc1d49bbfe40/An-Incremental-Tensor-Train-Decomposition-for-Cyber-Physical-Social-Big-Data.pdf)

* **DTT**: "DTT: A highly efficient distributed tensor train decomposition method for IIoT big data", in **IEEE Trans. Ind. Inf**, 2021, [Paper](https://ieeexplore.ieee.org/document/8963751)

### [Streaming Decomposition of Tensors in Non-Stationary Environments (i.e., TT-Cores Can Change Over Time)](#content) 

* **TT-FOA**: "Adaptive Algorithms for Tracking Tensor-Train Decomposition of Streaming Tensors", in **EUSIPCO**, 2020,  [Paper](https://ieeexplore.ieee.org/document/9287780), [PDF](https://hal.univ-lille.fr/hal-02865257v1/file/EUSIPCO%281%29.pdf), [Code](https://github.com/thanhtbt/ATT)

* **ROBOT**: "Robust Tensor Tracking With Missing Data Under Tensor-Train Format", in **EUSIPCO**, 2022,  [Paper](https://ieeexplore.ieee.org/document/9287780), [PDF](https://thanhtbt.github.io/files/2022_EUSIPCO-Robust%20Tensor%20Tracking%20with%20Missing%20Data%20under%20Tensor-Train%20Format.pdf), [Code](https://github.com/thanhtbt/ROBOT)
  
* **ATT**: "A Novel Recursive Least-Squares Adaptive Method For Streaming Tensor-Train Decomposition With Incomplete Observations", in **Signal Process.**, 2024, [Paper](https://www.sciencedirect.com/science/article/pii/S0165168423003717), [PDF](https://thanhtbt.github.io/files/2023_SP_ATT.pdf), [Code](https://github.com/thanhtbt/ATT-miss)

* **STTA**: "Streaming tensor train approximation", in **SIAM J. Sci. Comput.**, 2023, [Paper](https://epubs.siam.org/doi/abs/10.1137/22M1515045), [PDF](https://arxiv.org/abs/2208.02600), [Code](https://github.com/RikVoorhaar/tt-sketch)

* **SPTT**: "Streaming probabilistic tensor train decomposition", in ArXiv, 2023, [Paper](https://arxiv.org/pdf/2302.12148.pdf)

* **TT-ICE**: "An Incremental Tensor Train Decomposition Algorithm", in **SIAM J. Scient. Comput.**, 2024. [Paper](https://arxiv.org/pdf/2211.12487.pdf), [Code](https://github.com/dorukaks/TT-ICE)

* **Online TT**: "A Tensor Based Unified Framework for Streaming Signal Online Analysis", in **IEEE SPL**, 2024,  [Paper](https://ieeexplore.ieee.org/abstract/document/10555149)

  
## [Dynamic Tensor Analysis Under Block-Term Decomposition Format](#content) 


* **OnlineBTD**: "OnlineBTD: Streaming algorithms to track the block term decomposition of large tensors", in **DSAA**, 2020, [Paper](https://ieeexplore.ieee.org/abstract/document/9260061), [PDF](https://www.cs.ucr.edu/~epapalex/papers/2020_DSAA_onlineBTD.pdf), [Code](http://www.cs.ucr.edu/~egujr001/ucr/madlab/src/OnlineBTD.zip)
  
* **O-BTD-RLS**: "Online rank-revealing block-term tensor decomposition", in **Signal Process.**, 2023, [Paper](https://www.sciencedirect.com/science/article/pii/S0165168423002001), [PDF](https://arxiv.org/pdf/2106.10755.pdf)

* **SBTD**: "A Novel Tensor Tracking Algorithm For Block-Term Decomposition of Streaming Tensors", in **IEEE SSP**, 2023, [Paper](https://ieeexplore.ieee.org/document/10208007), [PDF](https://thanhtbt.github.io/files/2023_SSP%20-%20A%20novel%20tensor%20tracking%20algorithm%20for%20block-term%20decomposition%20of%20streaming%20tensors.pdf)


## [Dynamic Tensor Analysis Under T-SVD Format](#content) 

* **TO-RPCA**: "An online tensor robust PCA algorithm for sequential 2D data", in **IEEE ICASSP**, 2016, [Paper](https://ieeexplore.ieee.org/document/7472114), [PDF](https://merl.com/publications/docs/TR2016-004.pdf)
  
* **TOUCAN**: "Grassmannian optimization for online tensor completion and tracking with the T-SVD", in **IEEE Trans. Signal Process.**, 2022, [Paper](https://ieeexplore.ieee.org/abstract/document/9756209), [PDF](https://arxiv.org/abs/2001.11419), [Code](https://web.eecs.umich.edu/~girasole/?p=676)
  * Conference version: "Online Tensor Completion and Free Submodule Tracking With The T-SVD", in **IEEE ICASSP**, 2020, [Paper](https://ieeexplore.ieee.org/abstract/document/9053199), [PDF](https://par.nsf.gov/servlets/purl/10205313), [Code](https://par.nsf.gov/servlets/purl/10205313)

* "Effective streaming low-tubal-rank tensor approximation via frequent directions", in **IEEE Trans. Neural Netw. Learn. Syst.**, 2022, [Paper](https://ieeexplore.ieee.org/document/9796147), [PDF](https://arxiv.org/pdf/2108.10129.pdf)

## [Dynamic Tensor Analysis Under Tensor-Ring/Tensor Network Format](#content) 

* "Multi-Aspect Streaming Tensor Ring Completion for Dynamic Incremental Data", in **IEEE Signal Process. Lett.**, 2022, [Paper](https://ieeexplore.ieee.org/abstract/document/9996547)
* **OLSTR-SGD/RLS**: "Online subspace learning and imputation by Tensor-Ring decomposition", in **Neural Netw.**, 2022, [Paper](https://www.sciencedirect.com/science/article/pii/S0893608022001976)
* **STRC & TRSSD**: "Patch tracking-based streaming tensor ring completion for visual data recovery", in **IEEE Trans. Circuits Syst. Video Techn.**, 2022, [Paper](https://ieeexplore.ieee.org/document/9828504), [PDF](https://arxiv.org/pdf/2105.14620.pdf), [Code](https://github.com/he1c/PTSTRC)
* **STR**:  "Tracking Tensor Ring Decompositions of Streaming Tensors", in ArXiv, 2023, [Paper](https://arxiv.org/pdf/2307.00719.pdf)


## [Dynamic/Adaptive Coupled Tensor Decomposition](#content) 
* **C-CPD-T**: "An Adaptive Algorithm for Tracking Third-Order Coupled Canonical Polyadic Decomposition", in **IEEE ICASSP**, 2024, [Paper](https://ieeexplore.ieee.org/abstract/document/10447395)


## [Related Sources](#content) 
### [Good Surveys of (Batch) Tensor  Decomposition, Analysis, and Applications](#content) 

* Evrim Acar et al. *"[Unsupervised multiway data analysis: A literature survey](https://ieeexplore.ieee.org/abstract/document/4538221)"*, in **IEEE TKDE**, 2008, [PDF](https://web.archive.org/web/20130328222403id_/http://www.cs.rpi.edu/research/pdf/07-06.pdf)
  
* Tamara G. Kolda et al. *"[Tensor decompositions and applications](https://epubs.siam.org/doi/10.1137/07070111X)"*, in **SIAM Rev.**, 2009, [PDF](https://www.cs.umd.edu/class/fall2018/cmsc498V/slides/TensorBasics.pdf)
  
* Lieven De Lathauwer et al. *"[Breaking the curse of dimensionality using decompositions of incomplete tensors: Tensor-based scientific computing in
big data analysis](https://ieeexplore.ieee.org/document/6879619)"*, in **IEEE Signal Process. Mag.**, 2014, [PDF](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=925f8bc9ea8c12c398bb2473e737a67cc15cc8ad)

* Andrzej Cichocki & Lieven De Lathauwer et al. *"[Tensor decompositions for signal processing applications: From two-way to multiway component analysis](https://ieeexplore.ieee.org/abstract/document/7038247)"*, in **IEEE Signal Process. Mag.**, 2015, [PDF](https://arxiv.org/pdf/1403.4462.pdf)
  
* Andrzej Cichocki & Ivan Oseledets et al. *"[Tensor networks for dimensionality reduction and large-scale optimization: Part 1 low-rank tensor decompositions](https://www.nowpublishers.com/article/Details/MAL-059)"*, in **Found. Trends Mach. Learn.**, 2016, [PDF](https://arxiv.org/abs/1609.00893)
  
* Andrzej Cichocki & Ivan Oseledets et al. *"[Tensor networks for dimensionality reduction and large-scale optimization: Part 2 Applications and future perspectives](https://www.nowpublishers.com/article/Details/MAL-067)"*, in **Found. Trends Mach. Learn.**, 2017, [PDF](https://arxiv.org/abs/1708.09165)
  
* Nicholas D. Sidiropoulos & Lieven De Lathauwer  et al. *"[Tensor Decomposition for Signal Processing and Machine Learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7891546)"*, in **IEEE Trans. Signal Process.**, 2017, [PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7891546)
  
* Nicholas D. Sidiropoulos et al. *"[Tensors for data mining and data fusion: Models, applications, and scalable algorithms](https://dl.acm.org/doi/10.1145/2915921)"*, in **ACM Trans. Intell. Syst. Technol**, 2017, [PDF](https://web.archive.org/web/20170712024543id_/http://people.ece.umn.edu/~nikos/a16-papalexakis.pdf)
  
* Yannis Panagakis et al. *"[Tensor methods in computer vision and deep learning](https://ieeexplore.ieee.org/document/9420085)"*, in **Proc. IEEE**, 2021, [PDF](https://arxiv.org/pdf/2107.03436.pdf)


### [Tensor Toolbox and Software](#content) 
#### [MATLAB](#content) 
* Nway-Toolbox, [Link](https://fr.mathworks.com/matlabcentral/fileexchange/1088-the-n-way-toolbox)
* TensorToolbox, [Link](https://www.tensortoolbox.org/)
* TensorLab, [Link](https://www.tensorlab.net/)
* TensorBox, [Link](https://github.com/phananhhuy/TensorBox)
* Tensor-Tensor Product Toolbox, [Link](https://github.com/canyilu/Tensor-tensor-product-toolbox/)
* Splatt, [Link](https://github.com/ShadenSmith/splatt)
* TT-Toolbox, [Link](https://github.com/oseledets/TT-Toolbox)
* Hierarchical Tucker Toolbox, [Link](https://www.epfl.ch/labs/anchp/index-html/software/htucker/)
  
#### [Python](#content) 
* TT-Toolbox, [Link](https://github.com/oseledets/ttpy)
* Pyttb Toolbox, [Link](https://github.com/sandialabs/pyttb)
* TensorLy, [Link](https://tensorly.org/stable/index.html)  
* Hottbox, [Link](https://github.com/hottbox/hottbox)
* Tensorfac, [Link](https://etiennecmb.github.io/tensorpac/)
* TensorD, [Link](https://tensord-v02.readthedocs.io/en/latest/introduction.html)

#### [R](#content) 
* Tensor, [Link](https://cran.r-project.org/web/packages/tensor/index.html)
* rTensor, [Link](https://cran.r-project.org/web/packages/rTensor/index.html)
* nnTensor, [Link](https://cran.r-project.org/web/packages/nnTensor/index.html)
* TensorBF, [Link](https://cran.r-project.org/web/packages/tensorBF/index.html)

#### [Julia](#content) 
* TensorDecomp, [Link](https://github.com/yunjhongwu/TensorDecompositions.jl)
* Tensortoolbox, [Link](https://github.com/lanaperisa/TensorToolbox.jl)
* iTensor, [Link](https://scipost.org/10.21468/SciPostPhysCodeb.4)

### [Some Applications and Time-Series Datasets](#content) 

#### [Computer Vision](#\content)

* Video/visual tracking: [ICCV 2007](https://ieeexplore.ieee.org/document/4408950), [ICCV 2015](https://www.cv-foundation.org/openaccess/content_iccv_2015_workshops/w24/html/Sobral_Online_Stochastic_Tensor_ICCV_2015_paper.html), [TPAMI 2016](https://ieeexplore.ieee.org/document/7429797), [TSP 2022](https://ieeexplore.ieee.org/document/9866940), [TNNLS 2022](https://ieeexplore.ieee.org/abstract/document/9772051)
  
* Online video denoising: [ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wen_Joint_Adaptive_Sparsity_ICCV_2017_paper.pdf),  [TIP 2018](https://ieeexplore.ieee.org/abstract/document/8438535)

* Segmentation and Classification: [TPAMI 2008](https://ieeexplore.ieee.org/abstract/document/4378389), [SIAM JMDS 2020](https://epubs.siam.org/doi/abs/10.1137/19M1257718)

* Video datasets: [Link-1](https://github.com/xiaobai1217/Awesome-Video-Datasets), [Link-2](https://paperswithcode.com/datasets?mod=videos), [Link-3](http://jacarini.dinf.usherbrooke.ca/)

#### [NeuroScience](#\content)

* Dynamic functional connectivity networks analysis: [TBE 2016](https://ieeexplore.ieee.org/abstract/document/7452353), [TSIPN 2017](https://ieeexplore.ieee.org/abstract/document/7852497), [Front. Neurosci. 2022](https://www.frontiersin.org/articles/10.3389/fnins.2022.861402/full)

* Online EEG Completion and Analysis: [ISMICT 2018](https://ieeexplore.ieee.org/document/8573711), [TSP 2022](https://ieeexplore.ieee.org/document/9866940), [Patterns 2023](https://www.sciencedirect.com/science/article/pii/S2666389923001046)

* Datasets: [Link-1](https://github.com/meagmohit/EEG-Datasets), [Link-2](https://github.com/openlists/ElectrophysiologyData), [Link-3](https://www.erpwav), [Link-4](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html)

#### [Anomaly Detection](#\content)

* Sensor Networks: [KIS 2014](https://link.springer.com/article/10.1007/s10115-014-0733-3), [TNSM 2016](https://ieeexplore.ieee.org/abstract/document/7536642), [DMKD 2018](https://link.springer.com/article/10.1007/s10618-018-0560-3)

* Visual anomaly detection: [TVCG 2017](https://ieeexplore.ieee.org/document/8022952)

* Others: See [Survey](https://www.sciencedirect.com/science/article/pii/S0950705116000472)


## Citation
If you find this repository helpful for your work, please cite

[1] L.T. Thanh, K. Abed-Meraim, N. L. Trung and A. Hafiane. “[*A Contemporary and Comprehensive Survey on Streaming Tensor Decomposition*](https://ieeexplore.ieee.org/document/9994046)”. **IEEE Trans. Knowl. Data Eng.**, 2023 [PDF](https://thanhtbt.github.io/files/2022_TKDE_A%20Contemporary%20and%20Comprehensive%20Survey%20on%20Streaming%20Tensor%20Decomposition.pdf).
