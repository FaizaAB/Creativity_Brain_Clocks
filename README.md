# Creative Experiences and Brain Clocks: Model-Sensitivity Reanalysis

This repository contains code and materials for a reanalysis of the brain-age modeling pipeline from:

**Coronel-Oliveros, C., Migeot, J., Lehue, F., Amoruso, L., Kowalczyk-Grębska, N., Jakubowska, N., Mandke, K. N., ... Ibanez, A. (2025).  
Creative experiences and brain clocks. *Nature Communications*.**

The original study investigated whether creative expertise and learning are associated with lower brain age gap (BAG) using M/EEG-derived functional connectivity, support vector machine brain-age prediction, graph-theoretical measures, whole-brain modeling, and cognitive decoding.

This repository extends that work by testing whether the reported BAG effects are robust to alternative brain-age prediction models. Specifically, this reanalysis trains and evaluates multiple machine-learning models on the same functional connectivity training data and applies the same downstream BAG analysis pipeline to the creativity-related cohorts.

## Purpose of this Reanalysis

The main goal is to evaluate whether the paper’s conclusions are sensitive to the choice of brain-age model.

The original paper used an SVM-based brain-age model. In this repository, I compare the original modeling approach with:

- Optimized SVM
- Random Forest
- XGBoost
- Convolutional Neural Network

The downstream analyses include:

- Brain age prediction performance
- Brain age gap calculations
- Expert vs. non-expert group comparisons
- Pre/post learning comparisons
- Active-control comparisons
- Expertise and performance correlations
- Network and mechanistic associations

This reanalysis is intended to support reproducibility and model-sensitivity assessment, not to replace the original study.

## Important Data Note

Individual-level data from the music expertise design are not included because those data are restricted under GDPR regulations and are available only upon request from the original authors.
Therefore, analyses involving the music cohort are not reproduced here unless the user has obtained access to those restricted data.

## Repository Structure
```text
.
├── Gaming/
│   └── Data and outputs for the gaming expertise cohort
│
├── Global_coupling/
│   └── Global coupling model outputs and parameter files
│
├── Learning/
│   └── Pre/post training data and actions-per-minute analyses
│
├── Tango/
│   └── Data and outputs for the tango dancer cohort
│
├── Visual/
│   └── Data and outputs for the visual artist cohort
│
├── Training_SVMs_Data/
│   └── Functional connectivity matrices and age labels used for model training
│
├── neurosynth_spin_test/
│   ├── AAL_coordinates.txt
│   ├── parcellated_data.npy
│   ├── cognitive_terms.npy
│   └── Ds_files/
│
├── svm_hyperparamter_tune.py
│   └── Optimized SVM training, hyperparameter tuning, BAG prediction, and plotting
│
├── main_xgboost.py
│   └── XGBoost brain-age model training and downstream BAG analyses
│
├── main_cnn.py
│   └── CNN brain-age model training and downstream BAG analyses
│
├── main_rf.py
│   └── Random Forest brain-age model training and downstream BAG analyses
│
├── plot_violins.py
│   └── Customized violin plot functions
│
├── params_SVM.npy
│   └── Saved SVM hyperparameters
│
├── experts.svg
│   └── Word cloud visualization of cognitive correlations for experts
│
├── training.svg
│   └── Word cloud visualization of cognitive correlations for the training group
│
└── README.md



### Main Scripts

</> Bash python svm_hyperparamter_tune.py
This script tunes SVM hyperparameters, trains the optimized SVM brain-age model, computes BAG values, and generates downstream figures and statistical results.
   
2. Run the main script to generate plots:
   python svm_hyperparamter_tune.py

3.  Run the main script to generate plots:
   python main_xgboost.py

4.  Run the main script to generate plots:
   python main_cnn.py

4.  Run the main script to generate plots:
   python main_rf.py


Key Features
------------

- Brain Age Prediction: Models estimating brain age using functional connectivity data across creative domains.
- Group Comparisons: Contrasts between experts and non-experts in music, dance, gaming, and visual arts.
- Training Effects: Longitudinal assessment of training (e.g., Sonata project) on brain age gaps.
- Cognitive Decoding: Mapping neural data onto cognitive ontologies using spin tests and surrogate null models.
- Visualization: Word clouds, violin plots, cortical projections using surface-based mapping (e.g., FsLR surfaces).
