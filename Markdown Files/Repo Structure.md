# Structure

Chain RL
│
├── Master.ipynb
│
├── Experiments_folder
│   ├── experiments.ipynb
│   └── utilities
│       └── file.py
│
└── Results
    └── environmentkey_folder
        └── default_folder
            ├── best_model.zip
            │   monitor.csv
                eval_folder
                ├─── eval_results.csv
                │    episode_folder
                │    └── data.csv
                │    figures_folder
                │    └── plots.pdf















# New

# Folders for Environments/Data -> NET1, IM0, etc.
## Folders for Models -> (includes model.zip, monitor etc.)
### Folders for Model Variations -> 'default', 'tuned'
#### Eval Folder ->
##### Evaluation Files and Folders for GetData (stock, reorder, profit etc.)

# dictionary.py 
## Contains:
    - algorithm_dictionary
    - environment_dictionary
    - tuned_params_dictionary (work in progress)


# Folders for examples -> 
## net vis, notebooks

# Eval.py - or maybe Eval.ipynb
.py for taking monitor and evaluating. Putting files in Eval Folder

# Train.py
.py for training models

# Tune.py
.py for tuning models

# Plotting.py 
.py Functionality for different kinds of plots (purely model results)

# Tuning Folder ????
-> PPO -> trials
-> alg_name -> trials







# Naming Structure for New Environ