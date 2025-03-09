# 3p
import numpy as np
import cv2
from scipy.spatial import distance
from scipy.ndimage.filters import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# project
from utils import get_sparse_neighbor

def create_spacial_affinity_kernel(spatial_sigma: float, size: int = 15):
    """Create a kernel (`size` * `size` matrix) that will be used to compute the he spatial affinity based Gaussian weights.

    Arguments:
        spatial_sigma {float} -- Spatial standard deviation.

    Keyword Arguments:
        size {int} -- size of the kernel. (default: {15})

    Returns:
        np.ndarray - `size` * `size` kernel
    """
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j), (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))

    return kernel


def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3):
    """Compute the smoothness weights used in refining the illumination map optimization problem.

    Arguments:
        L {np.ndarray} -- the initial illumination map to be refined.
        x {int} -- the direction of the weights. Can either be x=1 for horizontal or x=0 for vertical.
        kernel {np.ndarray} -- spatial affinity matrix

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability. (default: {1e-3})

    Returns:
        np.ndarray - smoothness weights according to direction x. same dimension as `L`.
    """
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    T = convolve(np.ones_like(L), kernel, mode='constant')
    T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
    return T / (np.abs(Lp) + eps)


def fuse_multi_exposure_images(im: np.ndarray, under_ex: np.ndarray, over_ex: np.ndarray,
                               bc: float = 1, bs: float = 1, be: float = 1):
    """perform the exposure fusion method used in the DUAL paper.

    Arguments:
        im {np.ndarray} -- input image to be enhanced.
        under_ex {np.ndarray} -- under-exposure corrected image. same dimension as `im`.
        over_ex {np.ndarray} -- over-exposure corrected image. same dimension as `im`.

    Keyword Arguments:
        bc {float} -- parameter for controlling the influence of Mertens's contrast measure. (default: {1})
        bs {float} -- parameter for controlling the influence of Mertens's saturation measure. (default: {1})
        be {float} -- parameter for controlling the influence of Mertens's well exposedness measure. (default: {1})

    Returns:
        np.ndarray -- the fused image. same dimension as `im`.
    """
    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    images = [np.clip(x * 255, 0, 255).astype("uint8") for x in [im, under_ex, over_ex]]
    fused_images = merge_mertens.process(images)
    return fused_images


def refine_illumination_map_linear(L: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """Refine the illumination map based on the optimization problem described in the two papers.
       This function use the sped-up solver presented in the LIME paper.

    Arguments:
        L {np.ndarray} -- the illumination map to be refined.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem.
        kernel {np.ndarray} -- spatial affinity matrix.

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability (default: {1e-3}).

    Returns:
        np.ndarray -- refined illumination map. same shape as `L`.
    """
    # compute smoothness weights
    wx = compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
    wy = compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)

    n, m = L.shape
    L_1d = L.copy().flatten()

    # compute the five-point spatially inhomogeneous Laplacian matrix
    row, column, data = [], [], []
    for p in range(n * m):
        diag = 0
        for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
            weight = wx[k, l] if x else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    F = csr_matrix((data, (row, column)), shape=(n * m, n * m))

    # solve the linear system
    Id = diags([np.ones(n * m)], [0])
    A = Id + lambda_ * F
    L_refined = spsolve(csr_matrix(A), L_1d, permc_spec=None, use_umfpack=True).reshape((n, m))

    # gamma correction
    L_refined = np.clip(L_refined, eps, 1) ** gamma

    return L_refined


def correct_underexposure(im: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """correct underexposudness using the retinex based algorithm presented in DUAL and LIME paper.

    Arguments:
        im {np.ndarray} -- input image to be corrected.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem.
        kernel {np.ndarray} -- spatial affinity matrix.

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability (default: {1e-3})

    Returns:
        np.ndarray -- image underexposudness corrected. same shape as `im`.
    """

    # first estimation of the illumination map
    L = np.max(im, axis=-1)
    # illumination refinement
    L_refined = refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)

    # correct image underexposure
    L_refined_3d = np.repeat(L_refined[..., None], 3, axis=-1)
    im_corrected = im / L_refined_3d
    return im_corrected


class LOLDataset(Dataset):
    def __init__(self, low_light_folder, enhanced_folder, transform=None):
        self.low_light_images = sorted(os.listdir(low_light_folder))
        self.enhanced_images = sorted(os.listdir(enhanced_folder))
        self.low_light_folder = low_light_folder
        self.enhanced_folder = enhanced_folder
        self.transform = transform

    def __len__(self):
        return len(self.low_light_images)

    def __getitem__(self, idx):
        low_light_image_path = os.path.join(self.low_light_folder, self.low_light_images[idx])
        enhanced_image_path = os.path.join(self.enhanced_folder, self.enhanced_images[idx])

        low_light_image = cv2.imread(low_light_image_path)
        enhanced_image = cv2.imread(enhanced_image_path)

        # Convert to RGB
        low_light_image = cv2.cvtColor(low_light_image, cv2.COLOR_BGR2RGB)
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            low_light_image = self.transform(low_light_image)
            enhanced_image = self.transform(enhanced_image)

        return low_light_image, enhanced_image

# Define transformation (resize and normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

low_light_folder = './lol_dataset/our485/low/'
enhanced_folder = './lol_dataset/our485/high/'

dataset = LOLDataset(low_light_folder, enhanced_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


class DLEEModel(torch.nn.Module):
    def __init__(self):
        super(DLEEModel, self).__init__()
        # Define your layers here (the architecture of the DLEE model)
        # For example, you can add layers like conv, batchnorm, etc.
        self.layer1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)  
        self.layer2 = torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        # self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)  
        # self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.conv_out = torch.nn.Conv2d(128, 3, kernel_size=3, padding=1)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        # x = torch.nn.functional.relu(self.conv3(x))
        # x = torch.nn.functional.relu(self.conv4(x))
        
        # x = torch.sigmoid(self.conv_out(x))
        return x

def train(model, dataloader, num_epochs=10, learning_rate=1e-4):
    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0

        for i, (low_light_image, enhanced_image) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Move data to GPU if available
            low_light_image, enhanced_image = low_light_image, enhanced_image

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(low_light_image)

            # Compute loss
            loss = criterion(outputs, enhanced_image)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader)}")

    print("Training Complete!")

# Initialize model and move it to GPU if available
model = DLEEModel()

# Train the model
train(model, dataloader, num_epochs=10)

def enhance_with_dlee(image: np.ndarray):
    # Convert image from BGR (OpenCV) to RGB (PyTorch model expects RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize and convert to PyTorch tensor
    image_rgb = image_rgb.astype(np.float32) / 255.0
    transform = transforms.ToTensor()
    input_tensor = transform(image_rgb).unsqueeze(0)  # Add batch dimension

    # Perform inference with the model
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Convert the output tensor back to NumPy array
    output_image = output_tensor.squeeze().cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))  # HxWxC
    
    # Clip values to [0, 1] and convert back to BGR for OpenCV compatibility
    output_image = np.clip(output_image, 0, 1)
    output_image = (output_image * 255).astype(np.uint8)
    output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    return output_image_bgr

def enhance_image_exposure(im: np.ndarray, gamma: float, lambda_: float, dual: bool = True, sigma: int = 3,
                           bc: float = 1, bs: float = 1, be: float = 1, eps: float = 1e-3, use_dlee: bool = False):
    """Enhance input image, using either DUAL, LIME, or DLEE method."""
    if use_dlee:
        # If DLEE is selected, use the DLEE model to enhance exposure
        return enhance_with_dlee(im)
    
    # Otherwise, continue with the original enhancement methods (DUAL/LIME)
    kernel = create_spacial_affinity_kernel(sigma)
    im_normalized = im.astype(float) / 255.
    under_corrected = correct_underexposure(im_normalized, gamma, lambda_, kernel, eps)

    if dual:
        inv_im_normalized = 1 - im_normalized
        over_corrected = 1 - correct_underexposure(inv_im_normalized, gamma, lambda_, kernel, eps)
        im_corrected = fuse_multi_exposure_images(im_normalized, under_corrected, over_corrected, bc, bs, be)
    else:
        im_corrected = under_corrected

    return np.clip(im_corrected * 255, 0, 255).astype("uint8")


