# Structure

```
Chain RL
│
├── Master.ipynb
│
├── Experiments_folder
│   ├── bullwhip.ipynb
│   └── scripts_folder
│       └── plot.py
│            bullwhip.py
│   └── Experimentdata_folder
│       └── bullwhip_folder
│
└── Results
│    └── environmentkey_folder
│        └── algorithmkey_folder
│            └── paramskey_folder 
│                ├── best_model.zip
│                │   monitor.csv
│                │    eval_folder
│                    ├─── eval_results.csv
│                    │    episode_folder
│                    │    └── data.csv
│                    │    figures_folder
│                    │    └── plots.pdf
└── Misc.
```




# dictionary.py 
## Contains:
```
    - algorithm_dictionary # Add algorithms here
    - environment_dictionary # Add iterations of environments here
    - tuned_params_dictionary (work in progress)
```

# Naming Structure for New Environment Key/Folder
```
{ENVIRONMENT}_{NO. OF PERIODS}p_.....
```

