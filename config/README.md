# Config

The config folder contains the configuration files used in the experiments.

Most files use the following format:
``` 
{
    "domain1": {
        "task1": "path1",
        "task2": "path2",
        ...
        },
    ...
}
```
### 1. data_config_exp: Training profile
`config/data_config_exp_6.json` is a configuration file that trains selectors using data from six tasks and `config/data_config_exp_12.json` includes data for 12 tasks
### 2. update option config
The configuration when using the selector update method

`config/task_add_config.json`: the configuration that adds domains or tasks
`config/task_balance_config.json`: the configuration that enhances some existing tasks 
`config/task_delete_config.json`: the configuration that delete some existing domians or tasks

### 3. Inference with selector config

`config/adapters_pool_config.json`: Inference loads the adapter's configuration file, where the element is the adapter weight file path

### 4. deepspeed config

`config/ds_zero2_no_offload.json`: Use deepspeed zero2 to accelerate training