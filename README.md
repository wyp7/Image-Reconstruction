# Image-Reconstruction

Step 1: use preprocess.ipynb to preprocess raw images obtained from experiments

Step 2: use scene_setup_pinhole to construct the Scene for reconstruction. This will out put the pkl file for scene modles

Step 3: use nnscript_pinhole to load scene constructed in step2 and perform neural network training. This script is arranged as a .py file to be run on a cluster.
