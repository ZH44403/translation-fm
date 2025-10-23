import torch
import scipy
import numpy as np


def Inception_features(model, images):
    
    images = images.cuda() if torch.cuda.is_available() else images
    
    with torch.no_grad():
        features = model(images)
        
    return features


def calculate_fid(real_features, generated_features):
    
    # mean of real features and generated features
    mu_real = np.mean(real_features, axis=0)
    mu_generated = np.mean(generated_features, axis=0)
    
    # covariance of real features and generated features
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_generated = np.cov(generated_features, rowvar=False)
    
    delta_mu = mu_real - mu_generated
    sqrtm_sigma = scipy.linalg.sqrtm(sigma_real.dot(sigma_generated))
    
    fid = delta_mu.dot(delta_mu) + np.trace(sigma_real + sigma_generated - 2 * sqrtm_sigma)
    
    return fid


def calculate_psnr(real_image, generated_image):
    
    mse = np.mean((real_image - generated_image) ** 2)
    
    if mse <= 1e-5:
        return 100
    
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(real_image, generated_image):
    
    ...