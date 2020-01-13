import numpy as np
from skimage import io,metrics
import os
import csv

folders = os.listdir('.')
folders.remove('summary.py')

summary = []

for i,folder in enumerate(folders):
    fixed  = io.imread(os.path.join(folder, 'fixed.png'))
    moving = io.imread(os.path.join(folder, 'moving.png'))
    m1 = metrics.mean_squared_error(fixed,moving)
    m2 = metrics.structural_similarity(fixed,moving)
    m3 = metrics.normalized_root_mse(fixed,moving)
    summary.append([folder, m1, m2, m3])

with open('eval.csv', 'w+')as file:
    wr = csv.writer(file, quoting=csv.QUOTE_ALL)
    wr.writerow(["id", "MSE", "SSIM", "NRMSE"])
    for line in summary:
        wr.writerow(line)
