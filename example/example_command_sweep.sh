python3 emsco_sweep.py  --train_data example/example_train.csv --val_data example/example_val.csv --test_data example/example_test.csv --cost_data example/example_costs.txt --exp_num 9  --iter_num 150  --runs 10  --max_stages 5  --sweep .05 --min_prob .55  --pop_size 300 --out example/example_sweep_results.txt