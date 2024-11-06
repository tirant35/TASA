model_path="Qwen/Qwen2-7B"
test_data="/data/data_adapters/fin/sa/test.json"

selector_folders=(
    "/models/adapters/fin/sa"
)

for folder in "${selector_folders[@]}"; do
    for selector_path in $(find "$folder" -mindepth 1 -maxdepth 1 -type d); do
        if [[ $selector_path =~ ^.*checkpoint ]]; then
            python ../test_adapter.py \
                --model_path "$model_path" \
                --adapter_model_id "$selector_path" \
                --data_path "$test_data" \
                --metric bert
            
            if [ $? -ne 0 ]; then
                echo "Error occurred while processing $selector_path"
            fi
        fi
    done
done