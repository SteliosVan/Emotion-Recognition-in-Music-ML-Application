# Emotion Recognition in Music: A Multimodal Review of EEG, Acoustic Features, and Deep Learning Approaches

**Author:** Stylianos Vantarakis (03121142)
**Affiliation:** National Technical University of Athens (NTUA)

## Overview

This paper surveys the state of the art in **music-induced emotion recognition** using EEG, acoustic features, and physiological signals (ECG, EMG, EDA), then presents an original multimodal EEG–audio experiment on the **DEAP dataset**, comparing classical ML and deep learning models under a subject-independent setting.

## Background: EEG Frequency Bands

The literature review centers on EEG oscillatory bands as primary markers of emotional/cognitive processing:

| Band | Frequency | Associated with |
|---|---|---|
| Delta | 0.5–4 Hz | Internal physiological regulation, deep relaxation |
| Theta | 4–8 Hz | Emotional arousal, focused attention, affective memory |
| Alpha | 8–12 Hz | Relaxed wakefulness; alpha decrease → heightened arousal |
| Beta | 13–30 Hz | Cognitive effort, attention; frontal asymmetry ↔ valence |
| Gamma | >30 Hz | High-level perceptual integration, emotional salience, tension |

Also reviewed: acoustic feature extraction (attack slope, onset rate, spectral flux, MFCCs, chroma, spectral centroid), the shift from classical ML (SVM, k-NN, decision trees) to deep architectures (notably **AT-DGNN**, a graph neural network treating EEG electrodes as nodes, achieving 85–86% on valence/arousal), and canonical datasets (DEAP, MAHNOB-HCI, SEED, SEED-IV, AMIGOS, MEEG).

## Subject-Dependent vs. Subject-Independent Recognition

- **Subject-dependent** models (trained/tested on the same person): 90–98% accuracy, but limited generalizability — well suited to personalized applications like music therapy.
- **Subject-independent** models (trained on some individuals, tested on unseen ones): accuracy drops to 50–70% due to inter-subject variability in physiology, musical background, and emotional style — a major open problem motivating GNNs, attention mechanisms, and domain adaptation.

## My Experiment

### Dataset & Problem Definition
- **DEAP dataset**: 32 participants × 40 audiovisual trials, 40-channel EEG @ 128 Hz, ~63s/trial.
- Valence and Arousal (1–9 scale) binarized at two thresholds tested (≥5 and ≥3, the latter to address class imbalance).
- Two independent binary classification tasks: high/low Valence, high/low Arousal.
- 1280 EEG trials total (32 × 40); 80/20 train-test split (`random_state=42`).

### EEG Feature Extraction
- **Continuous Wavelet Transform (CWT)**, Morlet wavelet, scales 1–128 → energy per scale → shape `(N, 40, 128)`.
- **Permutation Entropy (PE)**, 8 non-overlapping epochs per channel → shape `(N, 40, 8)`, capturing nonlinear signal complexity.

### Dimensionality Reduction & Feature Selection
- **PCA** applied to raw EEG, CWT features, and fused EEG–audio features (z-score normalized, reduced to top *k* components).
- **mRMR** (Minimum Redundancy Maximum Relevance) feature selection, run separately for Valence and Arousal.

### Audio Feature Extraction
- DEAP's 36 MIDI stimuli converted to WAV via **FluidSynth**; features extracted with **librosa**:
  - Chroma (12-dim), spectral flux (1-dim), spectral roll-off (1-dim) → 14-dim vector per stimulus.
- Aligned with first 36 EEG trials/participant → 1152 multimodal samples (32 × 36).

### Multimodal Fusion
- Early (feature-level) fusion: audio features replicated across 40 EEG channels, concatenated with EEG features, then PCA-reduced to 128 components → final tensor `(1152, 40, 128)`.

### Models
- **Classical:** SVM (RBF kernel, optional class-balanced weighting), Random Forest (500 trees, max depth 20, optional class weighting).
- **Deep learning:** LSTM (sequential EEG modeling), 1D CNN (spatial EEG patterns) — trained with BCE-with-logits loss, Adam, StepLR scheduler, 20 epochs, batch size 64.
- **Metrics:** Accuracy, Sensitivity (Recall), F1-score — evaluated separately for Valence and Arousal.

## Key Results

- **Accuracy alone is misleading**: most configurations show 70–80% accuracy while sensitivity/F1 collapse toward zero (especially EEG-only and RNN models), indicating majority-class collapse under class imbalance.
- **Audio > EEG** for generalization: audio-only Random Forest models consistently reach F1 > 0.50, especially for Arousal, while EEG-only models struggle to generalize across subjects despite CWT + permutation entropy features.
- **Multimodal fusion** improves sensitivity (SVM-fused models reach sensitivity up to ~0.75–0.99 with class weighting) but often at the cost of accuracy — a precision/recall trade-off.
- **Class weighting** dramatically boosts SVM sensitivity but can crater accuracy; Random Forest is more stable under weighting.
- **Lower binarization threshold (3 vs. 5)** improves class balance and generally raises sensitivity/F1, especially for audio-only and fused Random Forest models.
- **Deep learning underperforms**: RNN models consistently yield zero sensitivity/F1 across all configurations, likely due to limited data, no subject-specific adaptation, and information loss from aggressive dimensionality reduction — classical methods (notably Random Forest) proved more robust in this subject-independent, low-data setting.

## Summary of Findings

- Audio features generalize better than EEG features across subjects.
- EEG-based emotion recognition remains highly individualized and hard to generalize.
- Multimodal fusion trades accuracy for sensitivity.
- Random Forest is the most stable classifier across modalities/thresholds.
- Class weighting and threshold choice critically shape perceived performance.
- Deep learning does not inherently outperform classical ML here, absent subject-specific adaptation or much larger datasets.

## References

1. Xiao, M., et al. "MEEG and AT-DGNN: Advancing EEG Emotion Recognition with Music and Graph Learning." *arXiv:2407.05550* (2024).
2. Cui, X., et al. "A review: Music-emotion recognition and analysis based on EEG signals." *Frontiers in Neuroinformatics* 16 (2022): 997282.
3. Daly, I., et al. "Music-induced emotions can be predicted from a combination of brain activity and acoustic features." *Brain and Cognition* 101 (2015): 1-11.
4. Turchet, L., et al. "Emotion recognition of playing musicians from EEG, ECG, and acoustic signals." *IEEE Transactions on Human-Machine Systems* (2024).
5. Lyberatos, V., et al. "Music Interpretation and Emotion Perception: A Computational and Neurophysiological Investigation."
