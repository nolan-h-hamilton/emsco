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

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7, 7))
# setup subplots
axs[0,0].set_xlim([min(p_hat),max(p_hat)])
axs[0,0].set_xlabel("$\hat{p}$")
axs[0,0].set_ylabel("Test Accuracy")
axs[0,1].set_xlim([min(p_hat),max(p_hat)])
axs[0,1].set_xlabel("$\hat{p}$")
axs[0,1].set_ylabel("Test Coverage")
axs[0,2].set_xlim([min(p_hat),max(p_hat)])
axs[0,2].set_xlabel("$\hat{p}$")
axs[0,2].set_ylabel("Test Cost")
axs[1,0].set_xlim([min(cost),max(cost)])
axs[1,0].set_title("Cost-Accuracy Tradeoff")
axs[1,0].set_xlabel("Test Cost")
axs[1,0].set_ylabel("Test Accuracy")
axs[1,1].set_xlim([min(conc),max(conc)])
axs[1,1].set_xlabel("Test Coverage")
axs[1,1].set_ylabel("Test Accuracy")
axs[1,2].set_xlim([min(conc),max(conc)])
axs[1,2].set_xlabel("Test Coverage")
axs[1,2].set_ylabel("Test Cost")

# generate each subplot
axs[0,0].plot(p_hat, acc,marker='^',mfc='red')
axs[0,1].plot(p_hat, conc, marker='^', mfc = 'red')
axs[0,2].plot(p_hat, cost, marker='^', mfc = 'red')
axs[1,0].scatter(cost, acc,marker='^', c=conc)
# write separate script that just produces the cost-accuracy graph.
#  use `plot` not `subplot` and then plt.colorbor()
axs[1,0].plot(cost, acc)
axs[1,0].text(cost[0],acc[0], "$g_1 = {}$".format(conc[0]),fontsize='xx-small')
axs[1,0].text(cost[-1],acc[-1], "$g_1 = {}$".format(conc[-1]),fontsize='xx-small')
axs[1,1].plot(conc,acc, marker='^', mfc = 'red')
axs[1,2].plot(conc,cost, marker='^', mfc = 'red')
fig.tight_layout()

# save plot
plt.savefig(sys.argv[1].replace('.txt','_results.png'))