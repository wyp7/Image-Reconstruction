import pickle
import torch

import gradoptics as optics
from gradoptics.integrator import HierarchicalSamplingIntegrator
from ml.siren import Siren

import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# load scene and camera images
scene_objects = pickle.load(open("NW_mot_scene_components.pkl", "rb"))
data_intensities_all = pickle.load(open("NW_mot_images.pkl", "rb"))

loss_file_name = 'loss.csv'

targets = [];
for img in data_intensities_all:
    targets.append(img.flatten().cuda())

sel_mask = []
for img in targets:
    sel_mask.append(torch.ones(img.shape, dtype=torch.bool))

# set up SIREN model

device = 'cuda'
in_features = 3
hidden_features = 256
hidden_layers = 3
out_features = 1

model = Siren(in_features, hidden_features, hidden_layers, out_features,
              outermost_linear=True, outermost_linear_activation=nn.ReLU()).double().to(device)

# set up scene for rendering in training
# Region we want to integrate in + position
rad = 0.03
obj_pos = (0, 0, 0)

light_source = optics.LightSourceFromNeuralNet(model, optics.BoundingSphere(radii=rad, 
                                                                     xc=obj_pos[0], yc=obj_pos[1], zc=obj_pos[2]),
                                        rad=rad, x_pos=obj_pos[0], y_pos=obj_pos[1], z_pos=obj_pos[2])
scene_train = optics.Scene(light_source)

for obj in scene_objects:
    scene_train.add_object(obj)
    
# load neural net parameters from pre trained model
pretrain_model = pickle.load(open("pre_trained_model_state_dict.pkl", "rb"))
model.load_state_dict(pretrain_model)

sensor_list = [obj for obj in scene_train.objects if type(obj) == optics.Sensor]
lens_list = [obj for obj in scene_train.objects if type(obj) == optics.PerfectLens]

# Train
# split each batch into #sensor of mini-batches
# for each sensor pick 512/4 =128 random pixels

batch_size = 512
loss_fn = torch.nn.MSELoss()
integrator = HierarchicalSamplingIntegrator(64, 64)
optimizer = torch.optim.Adam(scene_train.light_source.network.parameters(), lr=2e-5)

losses = []

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

for i_iter in tqdm(range(int(5e4))):
    intensities_normalized = [];
    target_vals_normalized = [];
    for cam in range(len(targets)):
        sensor_here = sensor_list[cam]
        lens_here = lens_list[cam]
        h_here, w_here = sensor_here.resolution

        # Grab masked pixel indices + sample randomly
        idxs_all = torch.cartesian_prod(torch.arange(h_here//2, -h_here//2, -1), 
                                        torch.arange(w_here//2, -w_here//2, -1))
        
        idxs_all = idxs_all[sel_mask[cam].flatten()]
        
        rand_pixels = torch.randint(0, len(idxs_all), (int(batch_size/len(targets)),))
        target_vals = targets[cam][sel_mask[cam].flatten()][rand_pixels]  
        
        batch_pix_x = idxs_all[rand_pixels, 0]
        batch_pix_y = idxs_all[rand_pixels, 1]

        intensities_all = []
        
        device = 'cuda'
        lens_pos = lens_list[cam].transform.transform[:-1, -1].to(device)
        
        nb_pixels = len(batch_pix_x)
             
        origins = torch.zeros((nb_pixels,3),device = device,dtype = torch.float64)
        origins[:,0] = (batch_pix_x.to(device)-.5)* sensor_here.pixel_size[0]
        origins[:,1] = (batch_pix_y.to(device)-.5)* sensor_here.pixel_size[1]
        origins = sensor_here.c2w.apply_transform_(origins.reshape(-1, 3)).reshape((nb_pixels, 3))
        #pA = 1 / (sensor_here.resolution[0] * sensor_here.pixel_size[0] * sensor_here.resolution[1] * sensor_here.pixel_size[1])
        
        
        directions = lens_pos - origins
        directions = directions/torch.norm(directions, dim=1, keepdim=True)  

        
        outgoing_rays = optics.Rays(origins, directions, device='cuda')
        intensities = optics.backward_ray_tracing(outgoing_rays,scene_train, scene_train.light_source,integrator, max_iterations=6)

        # Scaling to help control loss values
        im_scale = targets[cam][sel_mask[cam].flatten()].mean().item()

        intensities_normalized.append(intensities/im_scale*5e5)
        target_vals_normalized.append(target_vals.double().cuda()/im_scale)
        

    # Calculate loss and update neural network parameters
    loss = loss_fn(torch.cat(tuple(intensities_normalized)), torch.cat(tuple(target_vals_normalized)))

    grads = gradient(intensities, scene_train.light_source.pdf_aux)
    reg = loss*0.5
    loss = loss + reg*torch.norm(grads, dim=1).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record and print out
    losses.append(loss.item())
    if i_iter % 100 == 0:
        print(loss.item())
        
    if i_iter % 100 == 0:
        with torch.no_grad():
            torch.save(scene_train.light_source.network.state_dict(),f'model_{i_iter}_NW_MOT_all_cameras_long.pt')


from numpy import savetxt            
savetxt(loss_file_name,losses,delimiter=',')

