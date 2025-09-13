# Credit Risk Assessment with Machine Learning

This repository contains all code and artefacts for a comprehensive study on **credit risk assessment** using the Lending Club dataset. The project combines traditional machine learning models with advanced methods such as **GAN/CGAN synthetic resampling**, **Transformer-based architectures**, **explainability via SHAP**, **fairness auditing**, and **statistical significance testing**.  

It is the implementation companion to the MSc dissertation:  
*“A Comparative Study of Machine Learning Techniques to Evaluate Creditworthiness” (University of Greenwich, 2025).*

---

## 📑 Project Overview

The research addresses the question:

> **What strategies can be employed to harness machine learning for precise credit risk prediction, while ensuring the results are interpretable and ethically aligned with responsible AI standards?**

Key contributions:
- **Baselines:** Logistic Regression (LR), Random Forest (RF), Multi-Layer Perceptron (MLP).  
- **Enhanced pipelines:** Stratified CV, SMOTETomek, bagging, threshold optimisation.  
- **GAN/CGAN augmentation:** Synthetic minority defaults to handle imbalance.  
- **Transformers:** CNN-Transformer and GRU-Transformer for tabular/sequence-style modelling.  
- **Explainability:** SHAP-based global feature importance and bar plots.  
- **Fairness auditing:** Demographic Parity Difference (DPD) and Equal Opportunity Difference (EOD).  
- **Significance testing:** McNemar’s test, Chi-square, KS-test, paired bootstrap for reliability.  

---

## 📂 Repository Structure

### Data
- `loans_full_schema.csv` — Lending Club dataset (10,000 loans, 55 features).  

### Core Implementations
- **Initial models:**  
  - `Group_LC_initial_implementation.py` — preprocessing + LR, RF, MLP.  
  - `Group_LC_all_models_initial_runner.py` — runs LR/RF/MLP + CNN/GRU Transformers side-by-side.  
  - `Group_LC_transformers_initial_implementation.py` — CNN/GRU Transformer architectures.  

- **Enhanced models:**  
  - `Group_LC_enhanced_implementation.py` — preprocessing, SMOTETomek, bagged LR, tuned RF, regularised MLP.  
  - `Group_LC_transformers_enhanced_implementation.py` — Transformers with imbalance-aware CV.  
  - `Group_LC_all_models_enhanced_runner.py` — unified CV runner for all enhanced models.  

### Resampling Techniques
- `sampler_smote.py` — SMOTETomek wrapper.  
- `sampler_gan.py` — vanilla GAN sampler.  
- `sampler_cgan.py` — conditional GAN sampler.  
- `GAN_imbalance_implementation.py` — standalone GAN resampling workflow.  
- `run_compare_sampling_now.py` — compares SMOTETomek, GAN, CGAN with metrics + plots.  

### Explainability
- `mlp_shap.py` — SHAP bar plot for MLP.  
- `shap_kernel_defaults_fixed.py` — CPU-safe SHAP (KernelExplainer) for LR.  

### Fairness Audits
- `fairness_audit_with_observed_eod.py` — DPD & EOD with proxy fallback.  
- `fairness_audit_robust_v2.py` — robust fairness audit with clear error handling.  

### Significance Testing
- `lc_prepare_and_predict.py` — generates `predictions_combined.csv` (test set preds).  
- `significance_tests_and_plots.py` — McNemar’s test + discordant pair plots.  

---

## ⚙️ Setup

### Requirements
- Python ≥ 3.10  
- Key libraries: `scikit-learn`, `tensorflow`, `torch`, `imblearn`, `shap`, `matplotlib`, `pandas`, `numpy`.  
- Install with:
```bash
pip install -r requirements.txt
```

### Data
Place `loans_full_schema.csv` in the project root.

---

## 🚀 Usage Examples

### 1. Baseline models
```bash
python Group_LC_initial_implementation.py
```

### 2. Enhanced models with resampling
```bash
python Group_LC_enhanced_implementation.py
```

### 3. Transformers (enhanced)
```bash
python Group_LC_transformers_enhanced_implementation.py --folds 5 --epochs 20
```

### 4. GAN/CGAN resampling comparison
```bash
python run_compare_sampling_now.py
```

### 5. SHAP explainability
```bash
python mlp_shap.py
python shap_kernel_defaults_fixed.py --data loans_full_schema.csv --outdir shap_out
```

### 6. Fairness audit
```bash
python fairness_audit_robust_v2.py --csv loans_full_schema.csv --group-col application_type
```

### 7. Significance testing
```bash
python lc_prepare_and_predict.py
python significance_tests_and_plots.py
```

---

## 📊 Outputs

- `metrics_all_models_initial.csv`, `metrics_all_models_enhanced.csv` — aggregated model metrics.  
- `sampling_comparison_metrics.csv` — GAN/CGAN/SMOTE metrics.  
- `shap_*` plots — feature importance visuals.  
- `dpd_*.png`, `eod_*.png` — fairness audit charts.  
- `discordant_*.png` — McNemar test discordant pairs.  

---

## 📚 References

Core references and methodology are detailed in the dissertation (see `Final_Report_Draft_with_Page_Numbers.docx`). Key supporting works include Bastani et al. (2020), Kozodoi et al. (2022), Nkambule et al. (2024), Siphuma & van Zyl (2025), and Goodfellow et al. (2014).  
