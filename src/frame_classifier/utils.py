import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

def show_image_with_caption(img, caption):
    npimg = img.numpy()
    npimg = np.clip(npimg / 2 + 0.5, min=0, max=1.0)     # unnormalize
    fig = plt.gcf()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(caption)
    plt.show()

def show_images(images, labels = None, maxNumber = 4):
    num_imgs = len(images)
    if labels is not None:
        assert(len(labels) == num_imgs)
    if (maxNumber > 0):
        num_imgs = min(maxNumber, num_imgs)

    caption = "Images with no labels"
    if labels is not None:
        caption = f"Image Labels: {labels[0:num_imgs]}"
    # show images and labels
    show_image_with_caption(torchvision.utils.make_grid(images[0:num_imgs]), caption)

def collect_error_cases(y_hat, y_gts, dataset, class_id, show=True, save=False):
    # collect error cases (mis-classified as given class_id)
    error_cases = (y_hat == class_id) & (class_id != y_gts)
    images = []
    gt_cls = []
    for i in range(len(error_cases)):
        if error_cases[i]:
            images.append(dataset[i][0])
            gt_cls.append(dataset[i][1])

    num_of_errors = len(images)

    if (num_of_errors > 0) and (show):
        print(f"Cases falsely classified as class {class_id}")
        show_images(images, gt_cls) 

    return num_of_errors
