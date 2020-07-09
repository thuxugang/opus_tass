# OPUS-TASS

Motivation: Predictions of protein backbone torsion angles (ϕ and ψ) and secondary structure from sequence are crucial subproblems in protein structure prediction. With the development of deep learning approaches, their accuracies have been significantly improved. To capture the long-range interactions, most of studies integrate bidirectional recurrent neural networks into their models. In this study, we introduce and modify a recently proposed architecture named Transformer to capture the interactions between the two residues theoretically with arbitrary distance. Moreover, we also take advantage of multi-task learning to improve the generalization of neural network by introducing related tasks into the training process. Similar to many previous studies, OPUS-TASS uses an ensemble of models and achieves better results. 

Results: OPUS-TASS uses the same training and validation sets as SPOT-1D. We compare the performance of OPUS-TASS and SPOT-1D on TEST2016 (1213 proteins) and TEST2018 (250 proteins) proposed in SPOT-1D paper, CASP12 (55 proteins), CASP13 (32 proteins) and CASP-FM (56 proteins) proposed in SANIT paper, and a recently released PDB structures collected from CAMEO (93 proteins) named as CAMEO93. On these 6 test sets, OPUS-TASS achieves consistent improvements in both backbone torsion angles prediction and secondary structure prediction. On CAMEO93, for SPOT-1D, the mean absolute errors for ϕ and ψ predictions are 16.89 and 23.02, respectively, and the accuracies for 3 and 8-state secondary structure predictions are 87.72% and 77.15%, respectively. In comparison, OPUS-TASS achieves 16.56 and 22.56 for ϕ and ψ predictions, and 89.06% and 78.87% for 3 and 8-state secondary structure predictions, respectively. In particular, after using our torsion angles refinement method OPUS-Refine as the post-processing procedure for OPUS-TASS, the mean absolute errors for final ϕ and ψ predictions are further decreased to 16.28 and 21.98.

## Usage

### Dependency

```
Python 3.7
TensorFlow v2.0
```

The standalone version of OPUS-TASS (including training & inference codes) is hosted on [Baidu Drive](https://pan.baidu.com/s/1Gx4iewX8_Khp4J87N_Fw5w) with password `ia8c`. Also, it can be downloaded directly from [Here](http://ma-lab.rice.edu/MaLab/dist/opus_tass.tar).

## Datasets

All the datasets we used as well as their corresponding OPUS-TASS prediction results are hosted on [Baidu Drive](https://pan.baidu.com/s/1L6w_qBIKvleO2uFr1Ekevw) with password `zmc1`. Also, they can be downloaded directly from [Here](http://ma-lab.rice.edu/MaLab/dist/opus_tass_datasets.zip).

Train & Val:
1. .angles (seq_id phi psi x1 x2 x3 x4): 181: Missing atoms for calculation 182: X doesn't exist
2. .csf: CSF3 feature
3. .inputs (20pssm + 30hhm + 7pc + 19psp): Inputs file for training
4. .labels (8ss(one-hot) + 3csf(double) + [2*(phi+psi) + 2*(x1+x2+x3+x4)](sin & cos) + asa + [phi+psi+x1+x2+x3+x4)](real value)): Labels file for training
5. .labels_mask (1: Will not be involved in loss calculation): Labels mask file for training

TEST2016 & TEST2018:
1. .dssp: (from Zhou's Website) : Prot.dssp contains the sequence, 8-state secondary structure, phi, psi, and ASA of the protein sequence. We extract the sequence from the structured segments of the PDB file. We did not observe a big change in performance when using the full length sequence bashed on PDB SEQRES records. Any angles of 360 (for theta/tau) are ignored in training and result analysis. Any X SS elements are ignored (rare). Most unclassified elements from DSSP are classified as coils, however.
2. .opus: OPUS-TASS result

CAMEO93 & CASP:
1. .angles (seq_id phi psi)
2. .ss: 8-state secondary structure

If you use the training or TEST2016 dataset, please cite following paper as well:
Jack Hanson, Kuldip Paliwal, Thomas Litfin, Yuedong Yang, Yaoqi Zhou, Accurate prediction of protein contact maps by coupling residual two-dimensional bidirectional long short-term memory with convolutional neural networks, Bioinformatics, Volume 34, Issue 23, 01 December 2018, Pages 4039–4045, https://doi.org/10.1093/bioinformatics/bty481

If you use the training or TEST2018, please cite following paper as well:
Jack Hanson, Kuldip Paliwal, Thomas Litfin, Yuedong Yang, Yaoqi Zhou, Improving prediction of protein secondary structure, backbone angles, solvent accessibility and contact numbers by using predicted contact maps and an ensemble of recurrent and residual convolutional neural networks, Bioinformatics, In Press, bty1006, https://doi.org/10.1093/bioinformatics/bty1006

If you use the CASP, please cite following paper as well:
Mostofa Rafid Uddin, Sazan Mahbub, M Saifur Rahman, Md Shamsuzzoha Bayzid, SAINT: Self-Attention Augmented Inception-Inside-Inception Network Improves Protein Secondary Structure Prediction, Bioinformatics, btaa531


## Useful Tools

### OPUS-CSF

To make CSF feature, please use the scripts in `mkcsf` folder. More information about OPUS-CSF can be found [here](https://github.com/thuxugang/opus_csf).

### OPUS-Refine

The information of OPUS-Refine can be found [here](https://github.com/thuxugang/opus_refine).

## Performance

### Terms
```
TA: backbone torsion angles
SS3: 3-state secondary structure
SS8: 8-state secondary structure
ASA: solvent accessible surface area
SDA: side-chain dihedral angles
```

### Performance of Different Types of Model

In OPUS-TASS, we train 6 different types of model with the same neural network architecture, but with different outputs: PP (TA), SS (SS3/SS8), C2 (SS3/SS8/TA), C3 (SS3/SS8/TA/CSF3), C4 (SS3/SS8/TA/CSF3/ASA) and C5 (SS3/SS8/TA/CSF3/ASA/SDA). 

#### Validation Set

|Models|SS3|SS8|MAE(ϕ)|MAE(ψ)|
|:----:|:----:|:----:|:----:|:----:|
|PP|-|-|**16.34±0.02**|23.07±0.07|
|SS|87.01±0.08|	76.54±0.10|	-|	-|
|C2	|87.22±0.09	|77.07±0.07|	16.67±0.10|	23.50±0.16|
|C3|	87.39±0.02|	77.35±0.07|	16.54±0.04|	23.12±0.02|
|C4|**87.43±0.03**|**77.39±0.04**|	16.50±0.06|**23.04±0.08**|
|C5	|87.35±0.08|	77.33±0.04|	16.54±0.08	|23.10±0.08|

#### TEST2016

|Models|SS3|SS8|MAE(ϕ)|MAE(ψ)|
|:----:|:----:|:----:|:----:|:----:|
|PP|-|-|**16.33±0.02**|23.59±0.05|
|SS|	86.59±0.04	|76.02±0.04|	-|	-|
|C2|	86.86±0.13	|76.53±0.16|	16.69±0.11|	23.99±0.20|
|C3|	87.05±0.05	|76.77±0.03	|16.55±0.02	|23.60±0.07|
|C4|	**87.08±0.07**|	**76.84±0.10**	|16.50±0.05|	**23.53±0.06**|
|C5|	86.94±0.04	|76.74±0.05|	16.56±0.07|	23.64±0.09|

### Performance of different predictors on TEST2016 and TEST2018

#### TEST2016

|Predictors|SS3|SS8|MAE(ϕ)|MAE(ψ)|
|:----:|:----:|:----:|:----:|:----:|
|SPOT-1D |	87.16 |	77.10 |	16.27 |	23.26 |
|OPUS-TASS	|87.79 |	78.01 	|15.78 |	22.46 |


#### TEST2018

|Predictors|SS3|SS8|MAE(ϕ)|MAE(ψ)|
|:----:|:----:|:----:|:----:|:----:|
|MUFOLD| 	84.78 |	73.66 |	17.78 |	27.24 |
|NetSurfP-2.0 |	85.31 	|73.81 |	17.90 |	26.63 |
|SPOT-1D |	86.18 |	75.41 |	16.89 |	24.87 |
|OPUS-TASS	|86.84 |	76.59 	|16.40 |	24.06 |

### Performance of different predictors on CASP

#### CASP12

|Predictors|SS3|P(ss3)|SS8|P(ss8)|MAE(ϕ)|P(mae(ϕ))|MAE(ψ)|P(mae(ψ))|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|SPOT-1D	|84.82	|-	|73.99|	-|	18.44|	-	|26.90|	-|
|OPUS-TASS	|85.47|	1.40E-02|	75.68	|4.84E-08	|18.08|	1.05E-02|	25.98|	7.97E-04|

#### CASP13

|Predictors|SS3|P(ss3)|SS8|P(ss8)|MAE(ϕ)|P(mae(ϕ))|MAE(ψ)|P(mae(ψ))|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|SPOT-1D	|86.53|	-	|74.32|	-	|18.48|	-	|26.73|	-|
|OPUS-TASS	|87.62|	3.55E-03|	76.30|	1.49E-05|	17.89|	1.24E-02|	25.93|	7.02E-02|

#### CASPFM

|Predictors|SS3|P(ss3)|SS8|P(ss8)|MAE(ϕ)|P(mae(ϕ))|MAE(ψ)|P(mae(ψ))|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|SPOT-1D	|82.37	|-|	71.11|	-|	19.39|	-	|30.10|	-|
|OPUS-TASS|	83.40|	2.26E-03|	73.27|	1.35E-08	|18.85|	2.42E-03	|28.00|	3.70E-08|


### Performance of different predictors on CAMEO93

|Predictors|SS3|SS8|MAE(ϕ)|MAE(ψ)|
|:----:|:----:|:----:|:----:|:----:|
|SPOT-1D |	87.72|	77.15	|16.89	|23.02|
|w/ OPUS-Refine|	-	|-	|16.65|	22.51|
|OPUS-TASS|	**89.06**|	**78.87**	|16.56	|22.56|
|w/ OPUS-Refine|	-|	-	|**16.28**|	**21.98**|


## Reference 
```bibtex
@article{xu2020opus3,
  title={OPUS-TASS: A Protein Backbone Torsion Angles and Secondary Structure Predictor Based on Ensemble Neural Networks},
  author={Xu, Gang and Wang, Qinghua and Ma, Jianpeng},
  journal={Bioinformatics},
  year={2020},
  publisher={Oxford University Press}
}
```


