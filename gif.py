import imageio
import os

experiment_number = 2
path_run = "C:/Users/josep/yolov7-segmentation/runs/predict-seg/"
exp = os.listdir(path_run)
exp.sort()
path_exp = os.path.join(path_run, exp[experiment_number - 1])
file = os.listdir(path_exp)
png_file = []
for name in file:
    if name.endswith('.png'):
        png_file.append(name)
path_gif = []
for i in range(len(name)):
    path_gif.append(os.path.join(path_exp, '{}.png'.format(i+1)))

with imageio.get_writer(os.path.join(path_exp, 'segmentation.gif'), fps = 10) as writer:
    for filename in path_gif:
        image = imageio.imread(filename)
        writer.append_data(image)
        