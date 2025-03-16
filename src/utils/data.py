import torch  # type: ignore
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

tud = torch.utils.data


# def load_data(data_path: str) -> tuple:
#     """
#     Loads files and reshape into 30x30x30 tensor.

#     Parameters:
#     - data_path: Path to data folder.

#     Returns:
#     - data: simulation data
#     - pores: 3D (30, 30, 30) binary array
#     - density: density factors
#     """

#     data = {}
#     pores = {}
#     density = {}

#     files = sorted(Path(data_path).glob('*.npy'))

#     if len(files) == 0:
#         raise ValueError("No files found in the given directory.")

#     for file in files:
#         split_file_path = file._tail if isinstance(file, Path) else file.split("/")
#         prefix_name, index, density_factor = split_file_path[-1].replace(".npy", "").split("_")
#         grid = np.load(file)
#         flags = grid[:, 3].reshape(30, 30, 30)

#         data[index] = grid
#         pores[index] = flags
#         density[index] = float(density_factor)

#     return data, pores, density


class GridDataset(tud.Dataset):
    def __init__(self, data_path: str):
        import glob
        self.data_path = data_path
        self.files = sorted(glob.glob(data_path + '*.npy'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        density_factor = float(file_path.split("_")[-1].replace(".npy", ""))

        grid = np.load(file_path)
        pores = grid[:, 3].reshape(30, 30, 30)

        pores_tensor = torch.tensor(pores, dtype=torch.float32).unsqueeze(0)
        density_tensor = torch.tensor([density_factor], dtype=torch.float32)

        return pores_tensor, density_tensor


def verify_data(data: np.array, grid_size: int = 30):
    # 1. Extract coordinates (x, y, z) from the data (first three columns)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # 2. Check if coordinates are normalized between 0 and 1 (unit cube)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)

    print("Min x:", x_min, "Max x:", x_max)
    print("Min y:", y_min, "Max y:", y_max)
    print("Min z:", z_min, "Max z:", z_max)

    if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1 or z_min < 0 or z_max > 1:
        print("Coordinates are not normalized between 0 and 1!")
        # If they are not normalized, scale them:
        # (Note: If x, y, z are not in the range [0, 1], they should be normalized to fit a unit cube)
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        z = np.clip(z, 0, 1)

    # 3. Map coordinates to grid indices
    # grid_size = 30  # Size of the grid, for 30x30x30 grid

    # Normalize coordinates to grid indices: multiply by grid size (30) and floor to get index positions
    ix = np.floor(x * (grid_size - 1)).astype(int)
    iy = np.floor(y * (grid_size - 1)).astype(int)
    iz = np.floor(z * (grid_size - 1)).astype(int)

    # Check if any of the indices fall outside the grid bounds (shouldn't happen if data is normalized)
    print("Min ix:", np.min(ix), "Max ix:", np.max(ix))
    print("Min iy:", np.min(iy), "Max iy:", np.max(iy))
    print("Min iz:", np.min(iz), "Max iz:", np.max(iz))

    # 4. Verify that all indices cover the full grid from 0 to 29 (no missing or duplicate indices)
    all_indices = np.stack([ix, iy, iz], axis=1)

    # Check for duplicates (shouldn't happen in a proper distribution)
    unique_indices = np.unique(all_indices, axis=0)
    print(f"Number of unique indices: {len(unique_indices)}")
    print(f"Expected number of unique indices: {grid_size ** 3}")

    # If the data is well-organized, the unique indices should match the expected grid size
    # assert len(unique_indices) == grid_size ** 3, "The coordinates do not cover the entire grid without gaps!"
    if len(unique_indices) != grid_size ** 3:
        print("The coordinates do not cover the entire grid without gaps!")

    # 5. Verify if each voxel has been assigned a value
    # Check how many times each index (i.e., each voxel) has been assigned a flag
    index_counts = {tuple(idx): 0 for idx in unique_indices}

    for idx in all_indices:
        index_counts[tuple(idx)] += 1

    # Print indices with multiple assignments (if any)
    duplicates = {idx: count for idx, count in index_counts.items() if count > 1}
    if duplicates:
        print("Duplicate indices found:", duplicates)
    else:
        print("No duplicates found, every voxel is assigned only once.")


def generate_uniform_grid(labels, grid_size=30):
    # # Generate a uniform grid of (x, y, z) coordinates
    # x_vals = np.linspace(0, 1, gridsize)
    # y_vals = np.linspace(0, 1, gridsize)
    # z_vals = np.linspace(0, 1, gridsize)

    # # Create a mesh grid for the coordinates
    # x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals)

    # # Flatten the grid to create a list of (x, y, z) coordinates
    # coordinates = np.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], axis=-1)

    # return coordinates

    # Parameters
    # num_voxels = grid_size ** 3  # 27000 voxels grid szie (30x30x30)

    # Generate uniform grid of (x, y, z) coordinates
    x_vals = np.linspace(0, 1, grid_size)
    y_vals = np.linspace(0, 1, grid_size)
    z_vals = np.linspace(0, 1, grid_size)

    # Create a mesh grid for the coordinates
    x_grid, y_grid, z_grid = np.meshgrid(x_vals, y_vals, z_vals)

    # Flatten the grid to create a list of (x, y, z) coordinates
    coordinates = np.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], axis=-1)

    # Create a label grid with binary values (0 or 1), which corresponds to solid or pore
    # For simplicity, let's assume 1 (pore) for all coordinates here
    # labels = np.ones(num_voxels)  # Replace with your actual flag data (0 for solid, 1 for pore)

    # Combine the coordinates and labels
    data = np.column_stack([coordinates, labels])

    # Verify by reshaping into a 3D grid
    label_grid = labels.reshape(grid_size, grid_size, grid_size)

    # Check grid size and uniqueness
    print("Shape of label grid:", label_grid.shape)
    print("Number of unique indices:", len(np.unique(coordinates, axis=0)))
    print(f"Expected number of unique indices: {grid_size ** 3}")

    # Optionally visualize the distribution
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=labels, cmap='coolwarm', s=2, marker='o')
    plt.show()

    return data


class PorosityDataset(tud.Dataset):
    def __init__(self, data, grid_size=30):
        """
        Args:
            data (numpy array): Data of shape (27000, 4), with x, y, z coordinates and porosity flag.
            grid_size (int): Size of the 3D grid (e.g., 30 for a 30x30x30 grid).
        """
        self.data = data
        self.grid_size = grid_size
        
        # Preprocess data to create the voxel grid representation
        self.voxel_grids, self.coord_grids, self.porosity_factors = self.preprocess_data()
        
    def preprocess_data(self):
        """
        Preprocess the data to create voxel grids from (x, y, z) coordinates and porosity flags.
        Returns:
            voxel_grids (tensor): The voxel grids for each sample.
            coord_grids (tensor): The normalized coordinates for each voxel.
            porosity_factors (tensor): The porosity density factors.
        """
        voxel_grids = []
        coord_grids = []
        porosity_factors = []
        
        for sample in self.data:
            coords = sample[:, :3]  # First three columns: x, y, z
            porosity_flags = sample[:, 3]  # Last column: porosity flag
            
            # Initialize an empty voxel grid (size grid_size x grid_size x grid_size)
            grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
            
            # Map (x, y, z) coordinates to the grid
            for i, (x, y, z, flag) in enumerate(sample):
                ix, iy, iz = int(x * (self.grid_size - 1)), int(y * (self.grid_size - 1)), int(z * (self.grid_size - 1))
                grid[ix, iy, iz] = flag  # Set the voxel value
            
            # Create a coordinate grid to map each voxel's (x, y, z) in the unit cube
            coord_grid = np.indices((self.grid_size, self.grid_size, self.grid_size)).transpose(1, 2, 3, 0)  # Shape: (grid_size, grid_size, grid_size, 3)
            coord_grid = coord_grid / (self.grid_size - 1)  # Normalize to [0, 1]
            
            # For porosity density, we just take the average porosity of the sample (or a scalar factor)
            porosity_factor = np.mean(porosity_flags)
            
            voxel_grids.append(grid)
            coord_grids.append(coord_grid)
            porosity_factors.append(porosity_factor)
        
        return torch.tensor(voxel_grids, dtype=torch.float32), torch.tensor(coord_grids, dtype=torch.float32), torch.tensor(porosity_factors, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the grid, normalized coordinates, and porosity factor for this sample
        return self.voxel_grids[idx], self.coord_grids[idx], self.porosity_factors[idx]


class PorosityDatasetV2(tud.Dataset):
    def __init__(self, data_path, grid_size=30):
        """
        Args:
            data (numpy array): Data of shape (27000, 4), with x, y, z coordinates and porosity flag.
            grid_size (int): Size of the 3D grid (e.g., 30 for a 30x30x30 grid).
        """
        import glob
        self.data_path = data_path
        self.grid_size = grid_size
        self.files = sorted(glob.glob(data_path + "*.npy"))
        
        # Preprocess data to create the voxel grid representation
        self.voxel_grids, self.coord_grids, self.porosity_factors = self.preprocess_data()
        
    def preprocess_data(self):
        """
        Preprocess the data to create voxel grids from (x, y, z) coordinates and porosity flags.
        Returns:
            voxel_grids (tensor): The voxel grids for each sample.
            coord_grids (tensor): The normalized coordinates for each voxel.
            porosity_factors (tensor): The porosity density factors.
        """
        voxel_grids = []
        coord_grids = []
        porosity_factors = []
        
        for file in self.files:
            prefix_name, index, density_factor = file.split("/")[-1].replace(".npy", "").split("_")
            sample = np.load(file)
            # coords = sample[:, :3]  # First three columns: x, y, z
            # porosity_flags = sample[:, 3]  # Last column: porosity flag
            
            # Initialize an empty voxel grid (size grid_size x grid_size x grid_size)
            grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
            
            # Map (x, y, z) coordinates to the grid
            for i, (x, y, z, flag) in enumerate(sample):
                ix, iy, iz = int(x * (self.grid_size - 1)), int(y * (self.grid_size - 1)), int(z * (self.grid_size - 1))
                grid[ix, iy, iz] = flag  # Set the voxel value
            
            # Create a coordinate grid to map each voxel's (x, y, z) in the unit cube
            coord_grid = np.indices((self.grid_size, self.grid_size, self.grid_size)).transpose(1, 2, 3, 0)  # Shape: (grid_size, grid_size, grid_size, 3)
            coord_grid = coord_grid / (self.grid_size - 1)  # Normalize to [0, 1]
            
            # For porosity density, we just take the average porosity of the sample (or a scalar factor)
            # porosity_factor = np.mean(porosity_flags)
            
            voxel_grids.append(grid)
            coord_grids.append(coord_grid)
            porosity_factors.append(float(density_factor))
        
        return torch.tensor(np.array(voxel_grids), dtype=torch.float32), torch.tensor(np.array(coord_grids), dtype=torch.float32), torch.tensor(np.array(porosity_factors), dtype=torch.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Return the grid, normalized coordinates, and porosity factor for this sample
        return self.voxel_grids[idx], self.coord_grids[idx], self.porosity_factors[idx]