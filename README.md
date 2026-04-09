# Federated Learning Fairness Project

This repository contains experiments on fairness in Federated Learning (FL) for gender classification across demographic groups.

The work progresses through:
- FL basics and attack simulation on MNIST
- fairness analysis on UTKFace
- final validation on FairFace

Primary outcome:
- Dynamic Client-Adaptive Fairness Weighting (DCA-FW) improves race-wise fairness while preserving model accuracy.

## Project Highlights

- Baseline FL can reach strong overall accuracy but still show race-wise disparity.
- Label-flipping attacks can seriously damage both performance and fairness.
- Static fairness methods are less effective in decentralized FL.
- Client-level dynamic fairness weighting gives the best balance.

Approximate summary results from project notes:
- UTKFace baseline fairness gap: about 0.085
- UTKFace with DCA-FW fairness gap: about 0.051
- FairFace baseline fairness gap: about 0.099
- FairFace with DCA-FW fairness gap: down to about 0.081 in later rounds

## Repository Structure

- `Mnist_FL/`
  - foundational FL notebooks (IID/Non-IID, attack experiments)
- `Utk_Face_Working/`
  - UTKFace fairness method notebooks (weighted loss, regularization, adversarial, dynamic weighting)
- `FairFace_working/`
  - FairFace scripts for baseline and dynamic fairness FL
  - utility scripts for dataset preparation and checks
- `Elaborated_summary.txt`
- `Refined_Summary.txt`
- `Attacks_and_Defence_FL.pdf`
- `FL_project_results.pdf`

## Environment Setup

Use Python 3.9+ (3.10 or 3.11 recommended).

### 1) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install torch torchvision numpy pandas scikit-learn pillow matplotlib jupyter
```

If you have a CUDA-enabled GPU, install the matching PyTorch build from the official PyTorch install page.

## Running Experiments

## A) MNIST phase (notebooks)

Open and run notebooks in `Mnist_FL/`:
- `Mnist_FL_CNN_implementation.ipynb`
- `FederatedLearning (1).ipynb`

## B) UTKFace phase (notebooks)

Open and run notebooks in `Utk_Face_Working/`:
- `UTK_Face_implementation(2).ipynb` (baseline)
- `FairClientFL_WeightedLoss_CrossEntropy(3).ipynb`
- `FairClientFL_FairnessRegularization(4).ipynb`
- `FairClientFL_AdversarialFairness(5).ipynb`
- `Dynamic_Client_Adaptive_Fairness_Weighting(6).ipynb`
- `OnlyAttackOnUTK_toReduceGap(7).ipynb`

## C) FairFace phase (scripts)

Move into the FairFace folder:

```powershell
cd FairFace_working
```

### 1) Prepare FairFace filenames to UTK-like format

This script renames images using labels from `train_labels.csv`:

```powershell
python rename.py
```

Expected naming format after rename:
- `age_gender_race_index.jpg`
- gender: Male=0, Female=1
- race: White=0, Black=1, East Asian=2, Indian=3, Latino_Hispanic=4, Middle Eastern=5, Southeast Asian=6

### 2) Optionally sample/copy subset

```powershell
python top_50k.py
```

Note:
- The script currently slices first 50000 files.
- Variable names mention 40k in places, but behavior copies 50k.

### 3) Run baseline FL on FairFace

```powershell
python baseline_fairface.py
```

### 4) Run dynamic fairness-weighted FL

```powershell
python dynamic_weighting.py
```

Alternative variant:

```powershell
python dynmaic_weighting_2.py
```

### 5) Quick bias distribution check

```powershell
python bias_check.py
```

## Important Data Path Assumptions

Main FairFace scripts assume the image folder is named:
- `FairFace_50k`

In `baseline_fairface.py`, `dynamic_weighting.py`, and `dynmaic_weighting_2.py`, adjust:
- `DATA_DIR = "FairFace_50k"`

if your local folder name is different.

## Metrics Used

- Overall gender classification accuracy
- Race-wise accuracy
- Fairness gap:
  - max(race_accuracy) - min(race_accuracy)
- In project notes/notebooks, additional fairness metrics are discussed (for example SPD and EOD).

## Reproducibility Notes

- Scripts set random seeds (`SEED = 42`) for Python, NumPy, and PyTorch.
- Full determinism may still vary by hardware/CUDA/cuDNN settings.

## Common Issues

- File not found errors:
  - Verify image folder name and location relative to script.
- CUDA not available:
  - Code falls back to CPU automatically.
- Memory issues on GPU:
  - Reduce `BATCH_SIZE` in script configs.

## Suggested Next Improvements

- Add a `requirements.txt` for fully pinned dependencies.
- Add unified CLI arguments (dataset path, rounds, clients, fairness ratio).
- Export run logs and plots for direct baseline vs DCA-FW comparison.

## Credits

Project carried out under guidance of Prof. Manisha Padala.
