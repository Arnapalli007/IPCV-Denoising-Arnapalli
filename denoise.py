import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# List of all Set12 images
image_paths = [
    '01.png','02.png','03.png','04.png','05.png','06.png',
    '07.png','08.png','09.png','10.png','11.png','12.png'
]

def add_gaussian_noise(img, sigma):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def add_salt_pepper_noise(img, density):
    noisy = img.copy()
    n_pixels = int(density * img.size)
    coords_r = np.random.randint(0, img.shape[0], n_pixels)
    coords_c = np.random.randint(0, img.shape[1], n_pixels)
    noisy[coords_r[:n_pixels//2], coords_c[:n_pixels//2]] = 255
    noisy[coords_r[n_pixels//2:], coords_c[n_pixels//2:]] = 0
    return noisy

def apply_filters(noisy_img):
    gaussian  = cv2.GaussianBlur(noisy_img, (5, 5), 1.5)
    median    = cv2.medianBlur(noisy_img, 5)
    bilateral = cv2.bilateralFilter(noisy_img, 9, 75, 75)
    return gaussian, median, bilateral

def evaluate(clean, denoised, name):
    p = psnr(clean, denoised, data_range=255)
    s = ssim(clean, denoised, data_range=255)
    m = mean_squared_error(clean.astype(float), denoised.astype(float))
    print(f'  {name:12s} | PSNR={p:.2f} dB | SSIM={s:.3f} | MSE={m:.1f}')
    return p, s, m

def save_comparison(clean, noisy, g, med, bil, filename, title):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, img, t in zip(axes,
                          [clean, noisy, g, med, bil],
                          ['Original', 'Noisy', 'Gaussian', 'Median', 'Bilateral']):
        ax.imshow(img, cmap='gray')
        ax.set_title(t, fontsize=12)
        ax.axis('off')
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Figure saved: {filename}')

# Main Experiment
print('='*65)
print('IPCV Denoising Experiment — Gayatri Naidu Arnapalli')
print('='*65)

for path in image_paths:
    clean = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if clean is None:
        print(f'Could not load {path}, skipping.')
        continue

    print(f'\nImage: {path}')

    #Gaussian noise sigma=20
    print(f'  Gaussian noise (sigma=20):')
    noisy_g = add_gaussian_noise(clean, sigma=20)
    g, med, bil = apply_filters(noisy_g)
    evaluate(clean, g,   'Gaussian')
    evaluate(clean, med, 'Median')
    evaluate(clean, bil, 'Bilateral')
    save_comparison(clean, noisy_g, g, med, bil,
                    f'fig_{path[:-4]}_gaussian.png',
                    f'{path} — Gaussian noise σ=20')

    #Salt & pepper density=0.05
    print(f'  Salt-and-pepper noise (density=0.05):')
    noisy_sp = add_salt_pepper_noise(clean, density=0.05)
    g, med, bil = apply_filters(noisy_sp)
    evaluate(clean, g,   'Gaussian')
    evaluate(clean, med, 'Median')
    evaluate(clean, bil, 'Bilateral')
    save_comparison(clean, noisy_sp, g, med, bil,
                    f'fig_{path[:-4]}_saltpepper.png',
                    f'{path} — Salt & Pepper noise d=0.05')

print('\n' + '='*65)
print('All done! Check your IPCV1 folder for the figure files.')
print('='*65)