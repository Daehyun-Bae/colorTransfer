import os
import cv2 as cv
import numpy as np


def main(base_img_path, palette_img_path, result_path):
    src_img = cv.imread(base_img_path)
    target_img = cv.imread(palette_img_path)

    src_img = cv.cvtColor(src_img, cv.COLOR_BGR2LAB)
    target_img = cv.cvtColor(target_img, cv.COLOR_BGR2LAB)

    h, w, c = np.shape(src_img)
    src_img = cv.resize(src_img, (h//4, w//4))

    s_mean, s_std = get_mean_std(src_img)
    t_mean, t_std = get_mean_std(target_img)

    h, w, c = np.shape(src_img)
    for i in range(h):
        if i % 100 == 0: print(i)
        for j in range(w):
            for k in range(c):
                x = src_img[i, j, k]
                x = ((x - s_mean[k]) * (t_std[k] / s_std[k])) + t_mean[k]
                x = round(x)

                x = 0 if x < 0 else x
                x = 255 if x > 255 else x
                src_img[i, j, k] = x

    src_img = cv.cvtColor(src_img, cv.COLOR_LAB2BGR)
    cv.imwrite(result_path, src_img)


def get_mean_std(img):
    mean, std = cv.meanStdDev(img)
    mean = np.hstack(np.around(mean, 2))
    std = np.hstack(np.around(std, 2))
    return mean, std


if __name__ == '__main__':
    src_dir = './img/src'
    target_dir = './img/target'
    result_dir = './result'

    for i, src in enumerate(os.listdir(src_dir)):
        src_img = os.path.join(src_dir, src)
        for j, target in enumerate(os.listdir(target_dir)):
            target_img = os.path.join(target_dir, target)

            main(src_img, target_img, os.path.join(result_dir, 'result_{}_{}.jpg'.format(i, j)))

