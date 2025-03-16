import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# visualize 3D voxel models, input is a list of a batch of 3D arrays to visualize all conditions together  
def visualize_all(models , save = False, name = "output", fig_count = 4, fig_size = 5):
    
    fig = plt.figure()
    
    m = 0
    for model in models:
        
        if(model.dtype == bool):
            voxel = model
        else:
            voxel = np.squeeze(model) > 0.5
    
        ax = []
        colors = []
        
        for i in range(fig_count):
            ax.append( fig.add_subplot(len(models), fig_count, (m*fig_count) + i+1, projection='3d') )

        for i in range(fig_count):
            ax[i].voxels(voxel[i], facecolors='red', edgecolor='k', shade=False)
            ax[i].grid(False)
            ax[i].axis('off')
        
        m += 1
    
    plt.tight_layout()
    
    fig.set_figheight(fig_size)
    fig.set_figwidth(fig_size*fig_count)
    #plt.show()
    if(save):
        fig.savefig(name +'.png')
        plt.close(fig)
        fig.clear()
    else :
        plt.show()


# plot loss graph        
def plot_graph(lists, name):
    for l in lists:
        plt.plot(l)
    
    plt.savefig(name +'.png')
    plt.close()


# create the log folder   
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree()


def threshold_grid(grid, density):
    """
    Transforms a 30x30x30 grid into a binary grid by keeping only the `nb_ones` highest values.

    Computes nb_ones based on the formula:
    Density = 0.0882 * (Number of Ones)^0.2632

    Parameters:
    - grid: numpy array of shape (30, 30, 30), with continuous values (e.g., probabilities)
    - density: float, density value used to compute `nb_ones`

    Returns:
    - binary_grid: numpy array of shape (30, 30, 30), where only the top `nb_ones` values are 1, others are 0.
    """
    # Compute the number of ones to keep based on the given formula
    nb_ones = int((density / 0.0882) ** (1 / 0.2632))

    # Flatten grid and get the threshold value for the top `nb_ones` values
    sorted_values = np.sort(grid.flatten())[::-1]  # Sort in descending order
    threshold = (
        sorted_values[nb_ones - 1] if nb_ones > 0 else 1
    )  # Ensure valid threshold

    # Create binary grid by thresholding
    binary_grid = (grid >= threshold).astype(int)

    return binary_grid


def IoU(grid1, grid2, density, epsilon=1e-8):
    """
    Computes Intersection over Union (IoU) between two 3D binary grids.
    """
    # Transform generation into 0 and ones
    with torch.no_grad():
        binary_grid2 = threshold_grid(grid2.squeeze().cpu().numpy(), density)

    # Compute intersection and union
    intersection = torch.sum(grid1 * binary_grid2, dim=(0, 1, 2))
    union = torch.sum(grid1 + binary_grid2, dim=(0, 1, 2)) - intersection

    # Compute IoU
    iou = intersection / (union + epsilon)  # Avoid division by zero

    return iou


def plot_gen(grid1, grid2, index1, index2, density):
    """
    Plots two 3D scatter plots side by side.
    """

    # Transform generation into 0 and ones
    with torch.no_grad():
        binary_grid = threshold_grid(grid2.squeeze().cpu().numpy(), density)

    # Get the (x, y, z) coordinates where occupancy is 1 (pores)
    x1, y1, z1 = np.where(grid1 == 1)
    x2, y2, z2 = np.where(binary_grid == 1)

    # Normalize coordinates to be in the unit cube [0,1]
    x1, y1, z1 = x1 / 29, y1 / 29, z1 / 29
    x2, y2, z2 = x2 / 29, y2 / 29, z2 / 29

    # Create side-by-side 3D scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={"projection": "3d"})

    # Plot first grid
    axes[0].scatter(x1, y1, z1, c="blue", marker="o", alpha=0.5, s=5)
    axes[0].set_title(f"Simulation: {index1}", fontsize=18)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_zlabel("Z")

    # Plot second grid
    axes[1].scatter(x2, y2, z2, c="red", marker="o", alpha=0.5, s=5)
    axes[1].set_title(f"Simulation: {index2}", fontsize=18)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].set_zlabel("Z")

    plt.suptitle(f"Density Factor: {density}")
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()