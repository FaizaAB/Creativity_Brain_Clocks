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
```


### Running the Reanalysis

1. Clone the repository:
   git clone https://github.com/<FaizaAB>/<Creativity_Brain_Clocks>.git

2. Create and activate a Python environment:
   conda create -n brain-clocks python=3.10
   conda activate brain-clocks

3. Install required packages:
   pip install numpy pandas scipy scikit-learn matplotlib seaborn xgboost tensorflow

4. Run the model-specific scripts:
   python svm_hyperparamter_tune.py
   python main_xgboost.py
   python main_cnn.py
   python main_rf.py

### Citation

If you use this repository, please cite the original paper:
Coronel-Oliveros, C., Migeot, J., Lehue, F., Amoruso, L.,
Kowalczyk-Grębska, N., Jakubowska, N., Mandke, K. N., et al. (2025).
Creative experiences and brain clocks. Nature Communications.

You may also cite this repository if using the reanalysis scripts directly.

### Disclaimer

This repository is an independent reanalysis and is not the official repository of the original authors. The original code, data, and study design belong to the original authors. This repository is intended for reproducibility, critique, and model-sensitivity analysis.
