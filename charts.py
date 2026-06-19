import matplotlib.pyplot as plt
import numpy as np

# Actual results
filters = ['Gaussian', 'Median', 'Bilateral']

# Average PSNR values from your experiment
psnr_gaussian = [25.83, 25.63, 26.72]
psnr_saltpepper = [24.58, 26.30, 22.86]

# Average SSIM values from your experiment
ssim_gaussian = [0.758, 0.717, 0.786]
ssim_saltpepper = [0.688, 0.822, 0.573]

x = np.arange(len(filters))
width = 0.35

#Figure 1: Average PSNR
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, psnr_gaussian, width, 
               label='Gaussian noise (σ=20)', color='steelblue')
bars2 = ax.bar(x + width/2, psnr_saltpepper, width, 
               label='Salt & Pepper (d=0.05)', color='coral')

ax.set_xlabel('Filter', fontsize=12)
ax.set_ylabel('Average PSNR (dB)', fontsize=12)
ax.set_title('Average PSNR by Filter and Noise Type\n(across all 12 Set12 images)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(filters, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(20, 30)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('chart_psnr.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: chart_psnr.png')

#Figure 2: Average SSIM
fig, ax = plt.subplots(figsize=(9, 5))
bars3 = ax.bar(x - width/2, ssim_gaussian, width, 
               label='Gaussian noise (σ=20)', color='steelblue')
bars4 = ax.bar(x + width/2, ssim_saltpepper, width, 
               label='Salt & Pepper (d=0.05)', color='coral')

ax.set_xlabel('Filter', fontsize=12)
ax.set_ylabel('Average SSIM', fontsize=12)
ax.set_title('Average SSIM by Filter and Noise Type\n(across all 12 Set12 images)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(filters, fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0.4, 0.9)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

for bar in bars3:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars4:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('chart_ssim.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: chart_ssim.png')

print('All charts done!')