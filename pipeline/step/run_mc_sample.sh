#!/bin/bash

# --- Configuration ---
# Specify the total number of processes you want to run

task=intercode_sql
iter=1
sample_num=5
per_sample_workers=4
data_path="sampled_trajectories_${iter}/1.5b/${task}"

# Define the two ports to be used
PORT_FIRST_HALF=9214
PORT_SECOND_HALF=9215
# --- End Configuration ---

mkdir -p nohup

# Calculate the midpoint for port assignment
# Integer division is fine here as we'll use less than for the first half
MID_POINT=$((per_sample_workers / 2))

echo "Total processes to launch: ${per_sample_workers}"
echo "First half processes (0 to $((MID_POINT - 1))) will use port: ${PORT_FIRST_HALF}"
echo "Second half processes ($((MID_POINT)) to $((per_sample_workers - 1))) will use port: ${PORT_SECOND_HALF}"
echo "-----------------------------------------------------"


for (( j=1; j<sample_num; j++ ))
do
    for (( part_idx=0; part_idx<per_sample_workers; part_idx++ ))
    do
        # Determine the current port based on part_idx
        current_port=""
        if [ "$part_idx" -lt "$MID_POINT" ]; then
            current_port=$PORT_FIRST_HALF
        else
            current_port=$PORT_SECOND_HALF
        fi

        api_base_url="http://localhost:${current_port}/v1"

        log_file="nohup/${task}/${task}_mc_1.5b_sample${j}_${part_idx}.log"

        echo "Starting process with sample_idx=${j}, part_idx=${part_idx}, port=${current_port}, log_file=${log_file}"
        nohup python pipeline/step/monte_carlo_sample.py \
            --model 1.5b \
            --exp_config ${task} \
            --part_num ${per_sample_workers} \
            --part_idx ${part_idx} \
            --sample_idx ${j} \
            --save_path test_path \
            --data_path ${data_path} \
            --model_name "models/${task}-1.5b-sft" \
            --temperature 1.0 \
            --iter ${iter} \
            --api_base ${api_base_url} \
            --api_key EMPTY > ${log_file} 2>&1 &

        sleep 30
    done
done

echo "-----------------------------------------------------"
echo "All ${TOTAL_PROCESSES} processes launched."
echo "You can check the log files in the 'nohup' directory."
echo "To check running processes, you can use: ps aux | grep monte_carlo_sample.py"