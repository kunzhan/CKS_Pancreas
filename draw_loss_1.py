import re
import matplotlib.pyplot as plt
import os.path as osp
fullpath = osp.abspath('/home/datasets/Pancreas82NIH/logs/FD0:X3_1_20211101_191714.txt')
# mode = {'Loss'}
mode = {'Coarse', 'Fine', 'Avg'}
#ipdb.set_trace()
 
filedir, filename = osp.split(fullpath)
count = 0
Coarse, Fine, Avg, x = [], [], [], []
with open(fullpath, 'r') as f:
    rbsh = f.readline()
    while True:
        line = f.readline()
        if line == '':
            break
        if not line.startswith('0X'):
            continue
        count += 1
        ipdb.set_trace()
        line = line.replace(' ', '').replace('\t', '')
        pattern = re.compile(r'\w*.\w+')
        find_list = pattern.findall(line)
        if mode == {'Loss'}:
            Loss.append(float(find_list[0]))
        elif mode == {'Coarse', 'Fine', 'Avg'}:
            Coarse.append(float(find_list[0]))
            Fine.append(float(find_list[1]))
            Avg.append(float(find_list[2]))
        x.append(count)
 
pngName = filename.split('.')[0]
 
if mode == {'Loss'}:
    plt.plot(x, Loss)
elif mode == {'Coarse', 'Fine', 'Avg'}:
    plt.plot(x, Coarse, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=1)
    plt.plot(x, Fine, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=1)
    plt.plot(x, Avg, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=1)
    plt.legend(labels=('Coarse', 'Fine', 'Avg'))
 
plt.savefig(osp.join(filedir, pngName))
plt.show()