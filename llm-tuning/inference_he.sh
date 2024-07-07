# 遍历所有output 下所有模型；检查是否在inference/results 目录下存在同名的推理结果，如果存在，则跳过；
# 否则，进行推理：执行 inference/humaneval_inference.py 脚本；传递参数是模型的路径；

for model in /root/ljb/output/*; do
    model_name=$(basename "$model")
    result_file="/root/ljb/inference/results/${model_name}.jsonl"
    
    if [ -f "$result_file" ]; then
        echo "Inference result for $model_name already exists. Skipping..."
    else
        echo "Running inference for $model_name..."
        python /root/ljb/inference/humaneval_inference.py --model_path "$model"
    fi
done