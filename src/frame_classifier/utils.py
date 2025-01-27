import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

def imshow(img, caption):
    npimg = img.numpy()
    npimg = np.clip(npimg / 2 + 0.5, min=0, max=1.0)     # unnormalize
    fig = plt.gcf()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(caption)
    plt.show()

def show_images(images, labels):
    caption = f"GT Labels: {labels}"
    # show images
    imshow(torchvision.utils.make_grid(images), caption)

def show_error_cases(y_hat, y_gts, dataset, class_id):
    error_cases = (y_hat == class_id) & (class_id != y_gts)
    images = []
    gt_cls = []
    for i in range(len(error_cases)):
        if error_cases[i]:
            images.append(dataset[i][0])
            gt_cls.append(dataset[i][1])

    if len(images) == 0:
        print(f"No cases falsely classified as class {class_id}")
        return
    
    print(f"Cases falsely classified as class {class_id}")
    show_images(images, gt_cls) 