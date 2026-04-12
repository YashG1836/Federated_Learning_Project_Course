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



## Most Important Files for further use:
1) For UTK Face:
- Code and Results (Baseline):  Utk_Face_Working\UTK_Face_implementation(2).ipynb
- Code and Results (After Fairness): Utk_Face_Working\Dynamic_Client_Adaptive_Fairness_Weighting(6).ipynb

2) For Fair Face:
- Baseline:
- Code : FairFace_working\baseline_fairface.py 
- Results : FairFace_working\baseline_fairface_results
- After Fairness : 
- Code : FairFace_working\dynamic_weighting.py
- Results : FairFace_working\dynamic_weighting

## Overall Results and Summary :
- Results (Exact Values) : Overall_Results.txt
- Project Summary (Quick Overview of the project): Refined_Summary.txt
- Elaborated Summary (Everything logic explained) : Elaborated_summary.txt
- Main Report (pdf) : Federated_Learning_Elaborated.pdf
- FL Attacks and Defence (Reference Research Paper) : Attacks_and_Defence_FL.pdf

