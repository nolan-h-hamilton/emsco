"""Plot Results from `phat_sweep.py'"""
import sys
import matplotlib.pyplot as plt
import numpy as np

# create list for each numeric column in results
p_hat = []
acc = []
conc = []
cost = []
fline = True
for line in open(sys.argv[1],'r'):
    if fline:
        # skip first line - header
        fline = False
        continue
    line = line.strip('\n')
    line = line.split('\t')
    p_hat.append(float(line[1]))
    acc.append(float(line[2]))
    conc.append(float(line[3]))
    cost.append(float(line[4]))

plt.scatter(cost, acc,marker='^', c=conc)
plt.plot(cost, acc)
cb = plt.colorbar()
cb.ax.set_title("$g_1$")
plt.xlabel("Test Cost")
plt.ylabel("Test Accuracy")
plt.title("Cost/Accuracy Trade-Off")
# save plot
plt.savefig(sys.argv[1].replace('.txt','_results.png'))