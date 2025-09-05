<div align="right">
    
![Views](https://views-counter.vercel.app/badge?pageId=abdhmohammadi%2FGLSTSVM-CIL&label=Views)   

</div>

# GLSTSVM-CIL
**G**ravitational **L**east **S**quares **T**win **S**upport **V**ector **M**achine based on Optimal Angle for **C**lass **I**mbalance **L**earning

_Submision: December 24, 2024, Revised: Jun 22, 2025, Accepted: Aug 26, 2025 in the Journal of Applied Mathematics and Computation_

👉 Read the paper here: [GLSTSVM-CIL](https://www.sciencedirect.com/science/article/abs/pii/S009630032500431X) 

## 🔍 Overview

This repository contains the source code and experimental setup for the method proposed in our research paper:

<!--***Data, code, and analysis will be fully uploaded after the article is completed and accepted.***
-->
Author: Abdullah Mohammadi<br>
Supervisors: <br>
    Dr. Sohrab Effati<br>
    Dr. Jalal A. Nasiri<br>
    *Department of Applied Mathematics, Faculty of Mathematical Sciences, Ferdowsi University of Mashhad, P. O. Box 1159, Mashhad 91775, Iran*
---
# Data
> Using two-dimensional synthetic data, the position of hyperplanes and the performance of the method against changes in the imbalance ratio are evaluated

> KEEL repositoy was used to evaluate the performance on imbalance and noisy datasets.

> The data available in <a href="http://archive.ics.uci.edu/ml">UCI (Dua and Graff (2017))</a> and Kaggle repositories were used to evaluate medical applications.
 
> Text data from UCI, Kaggle and <a href="http://www.cad.zju.edu.cn/home/dengcai/Data/TextData.html">TDT2</a> dataset (Cai (2024)) have been used to evaluate models in text data classification.

> To evaluate Proposed Method’s performance on large datasets, the <a href="https://research.cs.wisc.edu/dmi/svm/ndc/"> NDC (Normal Distributed Clusters)</a> dataset Musicant (1998) was utilized.
# Images
<table align='center' border='1'>
<tr>
	<td align='center'><img src='https://github.com/abdhmohammadi/GLSTSVM-CIL/blob/main/images/gravity-presentation.png' width='400' height='300'/></td>
    <td align='center'><img src='https://github.com/abdhmohammadi/GLSTSVM-CIL/blob/main/images/hyperplanes.png' width='400' height='300'/></td>
</tr>
</table>

<table align='center' border='1'>
<tr>
	<td align='center'><img src='https://github.com/abdhmohammadi/GLSTSVM-CIL/blob/main/images/surface-c2-c3.png' width='828' height='300'/></td>
</tr>
</table>

<table align='center' border='1'>
<tr>
	<td align='center'><img src='https://github.com/abdhmohammadi/GLSTSVM-CIL/blob/main/images/Imbalance-rate-plotes.png' width='828' height='600'/></td>
</tr>
</table>

#  Please give me a star ⭐ if this topic was helpful to you.

<!--
> **"An Angle-based Least Squares Generalized Weighted LSTSVM for Class Imbalance Learning"**  
> Submitted to *Journal of Applied Mathematics and Computation*, 2025.

Our proposed method is designed for binary classification tasks with high class imbalance and Gaussian noise. It integrates fuzzy membership weighting and angle-based optimization, outperforming other LS-SVM variants such as LS-ATWSVM and LSFLSTSVM-CIL.

## 🧪 Highlights

- Effective under varying class imbalance ratios (1:1 to 1:20)
- Robust to additive Gaussian noise (mean = 0, variance = 1)
- Combines the strengths of both fuzzy weighting and angle-based boundary design
- Evaluated with four performance metrics: **Accuracy**, **F1-Score**, **G-Mean**, and their **average**

## 📁 Project Structure

```
.
├── datasets/
│   └── synthetic/
├── results/
│   └── figures/
│       └── Figure_1.png
├── src/
│   ├── alsgw_lstsqvm.py
│   └── utils.py
├── notebooks/
│   └── experiments.ipynb
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Make sure you have Python 3.9 or later installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

## 🚀 Running the Code

To reproduce the main experiments on synthetic datasets with Gaussian noise and varying imbalance ratios:

```bash
python src/alsgw_lstsqvm.py
```

You can also explore the Jupyter notebook:

```bash
jupyter notebook notebooks/experiments.ipynb
```

## 📊 Sample Result

The figure below demonstrates model performance across varying class imbalance ratios:

![Imbalance Rate Chart](results/figures/Figure_1.png)

- **Top-left**: Accuracy  
- **Top-right**: F1-score  
- **Bottom-left**: G-Mean  
- **Bottom-right**: Average of all three metrics  

The proposed method shows superior or equal performance in most scenarios, particularly due to its integrated dual-weight optimization structure.

## 📎 Citation

If you use this code in your research, please cite our work:

```bibtex
@article{YourLastName2025ALSGW,
  title={An Angle-based Least Squares Generalized Weighted LSTSVM for Class Imbalance Learning},
  author={Your Name and Co-authors},
  journal={Journal of Applied Mathematics and Computation},
  year={2025}
}
```

## 📜 License

This project is open-source and available under the MIT License.


## 👤 Author

**[Your Full Name]**  
Department of Mathematics, [Your University]  
GitHub: [@yourusername](https://github.com/yourusername)  
Website: [yourwebsite.com](https://yourwebsite.com)
-->
