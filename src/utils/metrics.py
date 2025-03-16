
# minkowski functionals

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
Z_DIM = 16
# fixed noise for display
fixed_noise = torch.randn(BATCH_SIZE,1, Z_DIM, Z_DIM, Z_DIM).to(DEVICE)
# generator
gen = Generator(in_channel=1, out_channel=1).to(DEVICE)
# optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))


area_real = np.zeros((len(DATASET[:,0,0,0])))
surface_real = np.zeros((len(DATASET[:,0,0,0])))
curvature_real = np.zeros((len(DATASET[:,0,0,0])))
euler_real = np.zeros((len(DATASET[:,0,0,0])))

for i in range((len(DATASET[:,0,0,0]))):
    area_real[i],surface_real[i], curvature_real[i],euler_real[i] = \
    mk.functionals(DATASET[i,:,:,:].numpy()<0.5,norm=True)
    
mean_phi_real = np.mean(area_real)
mean_surface_real = np.mean(surface_real)
mean_curvature_real = np.mean(curvature_real)
mean_euler_real = np.mean(euler_real)


# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
Z_DIM = 16
# fixed noise for display
fixed_noise = torch.randn(BATCH_SIZE,1, Z_DIM, Z_DIM, Z_DIM).to(DEVICE)
# generator
gen = Generator(in_channel=1, out_channel=1).to(DEVICE)
# optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# avoid epochs mistakes computing curvature (position 16,17,18) when jumping by a factor 2.
epochs = np.arange(0, 102, 2, dtype=int)
epochs = np.delete(epochs, [17,18,19])[1:]-1


# epochs = np.arange(0, 102, 2, dtype=int)[1:]-1
mean_phi_VWGAN = []
mean_surface_VWGAN = [] 
mean_curvature_VWGAN = [] 
mean_euler_VWGAN = []

for i in epochs:
    #Loading generator per epochs
    load_checkpoint(f"../checkpoints/generator/generator_no_poro_{i}_V2.pt", \
                    model=gen, optimizer=opt_gen, lr=1e-3)
    fake_images = gen(fixed_noise).detach().cpu().numpy().reshape(BATCH_SIZE,128,128,128)
    
    #Initialize matrices to store
    area_fake = np.zeros((len(fake_images[:,0,0,0])))
    surface_fake = np.zeros((len(fake_images[:,0,0,0])))
    curvature_fake = np.zeros((len(fake_images[:,0,0,0])))
    euler_fake = np.zeros((len(fake_images[:,0,0,0])))
    

    for j in range((len(fake_images[:,0,0,0]))):
        area_fake[j],surface_fake[j],curvature_fake[j], euler_fake[j] = \
        mk.functionals(fake_images[j,:,:,:]<0.5,norm=True)

    mean_phi_VWGAN.append(np.mean(area_fake)), mean_surface_VWGAN.append(np.mean(surface_fake)), \
    mean_curvature_VWGAN.append(np.mean(curvature_fake)), mean_euler_VWGAN.append(np.mean(euler_fake))

#Converting into arrays
mean_phi_VWGAN =  np.array(mean_phi_VWGAN)   
mean_surface_VWGAN = np.array(mean_surface_VWGAN) 
mean_curvature_VWGAN = np.array(mean_curvature_VWGAN) 
mean_euler_VWGAN = np.array(mean_euler_VWGAN) 


mean_phi = []
mean_surface = [] 
mean_curvature = [] 
mean_euler = []

#v4 used for the plots on the abstract
for i in epochs:
    #Loading generator per epochs
    load_checkpoint(f"../checkpoints/generator/generator_poro_{i}_v4.pt", \
                    model=gen, optimizer=opt_gen, lr=1e-3)
    fake_images = gen(fixed_noise).detach().cpu().numpy().reshape(BATCH_SIZE,128,128,128)
    
    #Initialize matrices to store
    area_fake = np.zeros((len(fake_images[:,0,0,0])))
    surface_fake = np.zeros((len(fake_images[:,0,0,0])))
    curvature_fake = np.zeros((len(fake_images[:,0,0,0])))
    euler_fake = np.zeros((len(fake_images[:,0,0,0])))
    

    for j in range((len(fake_images[:,0,0,0]))):
        area_fake[j],surface_fake[j],curvature_fake[j], euler_fake[j] = \
        mk.functionals(fake_images[j,:,:,:]<0.5,norm=True)

    mean_phi.append(np.mean(area_fake)), mean_surface.append(np.mean(surface_fake)), \
    mean_curvature.append(np.mean(curvature_fake)), mean_euler.append(np.mean(euler_fake))

#Converting into arrays
mean_phi =  np.array(mean_phi)   
mean_surface = np.array(mean_surface) 
mean_curvature = np.array(mean_curvature) 
mean_euler = np.array(mean_euler) 