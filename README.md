# Optimizing Error Correction Frameworks for EEG Signal Processing
**A Validated Hybrid Wavelet and Deep Learning Approach**

## 1. Project Overview
This repository contains the technical implementation for a real-time EEG denoising framework. The project addresses the "Deployment Gap" in neuro-diagnostics by combining the mathematical reliability of Discrete Wavelet Transforms (DWT) with the non-linear learning capabilities of Denoising Autoencoders (DAE).

## 2. Directory Structure
To satisfy Phase 1 audit requirements, the repository is organized into the following auditable sections:

- **📄 **Phase 1 EEG Signal Processing.ipynb**: Contains `main_pipeline.ipynb` and preprocessing modules.
- **📂 validation_plots/**: Contains the formal `README.md` and visual evidence of signal reconstruction.

## 3. Technical Foundation (Phase 1 Status)
The current version has been validated for the following milestones:
- ✅ **19-Channel Montage Mapping:** Automated ingestion of clinical signals via the 10-20 system.
- ✅ **Artifact Simulation:** Mathematical injection of non-stationary EOG and EMG noise manifolds.
- ✅ **Stage 1 Denoising:** Baseline suppression of artifacts using DWT (db4 basis function).
- ✅ **Feature Extraction:** Quantified spectral power analysis across 5 neural bands.

## 4. Hardware & Performance Target
- **Target Hardware:** Ryzen 5 3500U (15W TDP Mobile CPU)
- **Latency Budget:** < 55ms per 100ms epoch
- **Framework:** PyTorch / NumPy / PyWavelets

---
**Principal Supervisor:** Rahul Kumar  
**Version:** 1.0.1 (Phase 1 Audit Complete)
