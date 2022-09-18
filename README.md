# EMSCO - Evolutionary Multi-Stage Classifier Optimizer

EMSCO is used for constructing *budgeted*, *selective* classification systems. 
Three objectives: (i) accuracy, (ii) coverage, and (iii) processing cost are
optimized over a feasible region of *ordered feature set partitions* that define
a sequential, budgeted classification protocol with early-exits and a terminal
reject option. To effectively traverse a large $\theta(k^n)$ search space and
manage multiple objectives in a Pareto efficient manner, a problem-specific
multi-objective evolutionary algorithm is utilized.

![Pipeline](/resources/pipeline.png "Pipeline")

```emsco_sweep.py``` runs EMSCO over a sweep of confidence thresholds
to evaluate its performance in a full range of risk-averse, budgeted
contexts. Due to the stochastic nature of evolutionary algorithms, 
multiple (`--runs`) EA runs are conducted for each confidence threshold 
to compute average out-of-sample performance on the test set. This script,
with sufficient `--runs' and a reasonable range of confidence thresholds is
suggested if using EMSCO as a benchmark for comparison.

```resources/cost_acc.py``` accepts the output of ```emsco_sweep.py```
and produces a cost-accuracy trade-off curve color-coded by coverage

**Citation**
```
@INPROCEEDINGS{9870382,
  author={Hamilton, Nolan H. and Fulp, Errin W.},
  booktitle={2022 IEEE Congress on Evolutionary Computation (CEC)}, 
  title={Budgeted Classification with Rejection: An Evolutionary Method with Multiple Objectives}, 
  year={2022},
  volume={},
  number={},
  pages={1-10},
  doi={10.1109/CEC55065.2022.9870382}}
```

**Dependencies**

Python3

EMSCO uses a few popular Python packages as dependencies:

```
numpy
pandas
scikit-learn
matplotlib
```

See ```resources/environment_details.txt``` for exact versions of each package and Python
that produced the results in the **Example Use** section.

**Input**

* `--train_data`: a training split in CSV format (see `example/example_train.csv`)
* `--val_data`: a validation split in CSV format (see `example/example_val.csv`)
* `--test_data`: a test split in CSV format (see `example/example_test.csv`)
* `--cost_data`: a text file specifying acquisition costs for each feature (see `example/example_costs.txt`)

CSV files are expected to be formatted with header:
```f0,f1,f2,...,f[n],label```
where `f0` corresponds to the first feature value for the record,
and `label` corresponds to the label of the record.

The costs file should maintain the same feature order as in the
training/validation/test files. 

**Parameters**

Use ```python3 emsco_sweep.py -h``` for a list of all parameters. We provide extended
descriptions of several parameters here.

* `--pop_size`: default=300, number of chromosomes comprising each generation's population
* `--iter_num`: default=150, number of generations over which to optimize
* `--exp_num`: default=10, number of distinct confidence thresholds to test. see `--sweep`.
* `--min_prob`: default=0.55, minimum confidence threshold to accept prediction
* `--inc`: default=1, pop_size increment in case elite population grows to pop_size
* `--sweep`: default=.05, the tested confidence thresholds are given by {min_prob + i*sweep for i=0...exp_num-1}
* `--runs`: default=3, number of EMSCO runs for each confidence threshold to compute average performance. Increase as necessary to reduce variance of performance estimates.

**Example Use**

The results and plot in the ```example/``` directory can be generated with the following
commands. 

* run EMSCO 10 times for each confidence threshold in [.55,...,.95], and record average test performance.

```python3 emsco_sweep.py  --train_data example/example_train.csv --val_data example/example_val.csv --test_data example/example_test.csv --cost_data example/example_costs.txt --exp_num 9  --runs 10  --max_stages 5  --sweep .05 --min_prob .55 --out example/example_sweep_results.txt```

* Plot the results. The cost-accuracy tradeoff curve is an important tool for measuring and comparing performance of budgeted classifiers. Since EMSCO is also selective, coverage must be accounted for too, *so each point on the curve is color-coded according to this metric ($g_1$)*. 

```python3 resources/cost_acc.py example/example_sweep_results.txt```

![Test Performance](/example/example_res_plot.png "Cost-Accuracy Trade-Off Curve for Example Data Set")

