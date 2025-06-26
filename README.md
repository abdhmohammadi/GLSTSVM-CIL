
# GLSTSVM-CIL

A novel **Gravitational Least Squares Twin Support Vector Machine based on Optimal Angle for Class Imbalance Learning**

## 🔍 Overview

This repository contains the source code and experimental setup for the method proposed in our research paper:
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
