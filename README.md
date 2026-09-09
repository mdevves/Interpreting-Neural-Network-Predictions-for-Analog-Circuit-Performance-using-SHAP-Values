#  Analog Circuit Performance Prediction & Explainable AI (XAI) using PyTorch & SHAP

An end-to-end Machine Learning pipeline developed in Python and PyTorch to predict key performance metrics of CMOS Operational Amplifiers (Op-Amps) directly from design parameters. The project incorporates **Explainable AI (XAI)** via **SHAP (SHapley Additive exPlanations)** to gain physical and qualitative insights into circuit behavior, significantly accelerating the design iteration process over traditional SPICE simulations.

---

##  Project Overview

Designing analog integrated circuits typically requires time-consuming iteratively executed SPICE simulations. This repository presents a data-driven surrogate modeling approach using Deep Neural Networks (DNNs) to map structural transistor sizing parameters to circuit-level performance outputs.

Key highlights:
* **Surrogate Model:** Multi-Output Fully Connected Neural Network built with PyTorch.
* **Explainability:** Model-agnostic Kernel SHAP integration to evaluate parameter feature importance and physical impact on circuit metrics.
* **Automation:** Automated dataset generation/preprocessing pipelines and export of structural limits and data summaries directly to LaTeX format.

---

##  Circuit Parameters & Target Metrics

The model maps **5 Design Input Parameters** to **8 Circuit Output Metrics**:

###  Inputs (Sizing Parameters)
| Parameter | Description |
| :--- | :--- |
| `Cc` | Compensation Capacitance |
| `Wbp` | Transistor Width (PMOS Bias) |
| `Wdiff` | Transistor Width (Differential Pair) |
| `Wnb` | Transistor Width (NMOS Bias) |
| `Wpo` | Transistor Width (PMOS Output) |

###  Outputs (Performance Metrics)
| Metric | Description | Unit |
| :--- | :--- | :--- |
| `Phase Margin` | Stability Margin | Deg (°) |
| `UGBW` | Unity-Gain Bandwidth | Hz / MHz |
| `Gain` | Voltage Gain | dB |
| `ICMR_max` | Max Input Common-Mode Range | V |
| `ICMR_min` | Min Input Common-Mode Range | V |
| `Vout_max` | Max Output Voltage Swing | V |
| `Vout_min` | Min Output Voltage Swing | V |
| `Slew Rate` | Output Voltage Transition Speed | V/μs |

---

##  System Architecture & Pipeline
