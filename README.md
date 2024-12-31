# Image-Reconstruction

Step 1: use preprocess.ipynb to preprocess raw images obtained from experiments

Step 2: use scene_setup to construct the Scene for reconstruction. This will out put the pkl file for scene models

Step 3: use nnscript_lens_lr to load Scene constructed in step2 and perform neural network training. This script is arranged as a .py file to be run on a cluster.

Step 4: use analyze_result.ipynb to visualize the reconstruction results
