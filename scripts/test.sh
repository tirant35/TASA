#!/bin/bash


model_path="/Qwen/Qwen2-7B"
test_data="../data/test/test_exp2_265.json"
data_config_path="../config/data_config_exp_2.json"


selector_folders=(
    "../selector/exp6/selector_6_1_update_3"
    # more paths
)

for folder in "${selector_folders[@]}"; do
    
    for selector_path in $(find "$folder" -mindepth 1 -maxdepth 1 -type d); do
        if [[ $selector_path =~ ^.*checkpoint ]]; then

            python ../src/test_selector.py \
                --model_path "$model_path" \
                --selector_path "$selector_path" \
                --test_data "$test_data" \
                --data_config_path "$data_config_path"
            
            if [ $? -ne 0 ]; then
                echo "Error occurred while processing $selector_path"
            fi
        fi
    done
done