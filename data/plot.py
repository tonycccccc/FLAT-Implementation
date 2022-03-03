import numpy as np
import matplotlib.pyplot as plt
import os
import re
if __name__=="__main__":
    line=[]
    currpath = os.getcwd()
    files = os.listdir(currpath)
    #Sort the file sequence based on the file number in the filename (idx variable in script)
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    for file in files:
        if (not os.path.isdir(file)):
            continue
        subdir = os.path.join(currpath, file, "data1.txt")
        line.append(np.loadtxt(subdir))

    # Can plot up to 8 lines: set the label and color here
    line_color = ['b','g', 'y', 'c', 'm', 'r', 'k', 'tab:brown']
    line_label = ["test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8"]
    x = np.arange(64, 16*1024, 64)
    for i in range(0, len(line), 1):
        plt.plot(x, line[i], line_color[i], label=line_label[i])
    plt.legend()
    plt.title("Runtime")
    plt.xlabel("Source Length")
    plt.ylabel("Time in seconds")
    plt.show()
