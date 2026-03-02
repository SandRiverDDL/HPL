#!/bin/bash


# --- Configuration ---
# Specify the total number of processes you want to run
TOTAL_PROCESSES=5
task=intercode_sql

# Define the two ports to be used
PORT_FIRST_HALF=9214
PORT_SECOND_HALF=9215
# --- End Configuration ---

mkdir -p nohup

# Calculate the midpoint for port assignment
# Integer division is fine here as we'll use less than for the first half
MID_POINT=$((TOTAL_PROCESSES / 2))

echo "Total processes to launch: ${TOTAL_PROCESSES}"
echo "First half processes (0 to $((MID_POINT - 1))) will use port: ${PORT_FIRST_HALF}"
echo "Second half processes ($((MID_POINT)) to $((TOTAL_PROCESSES - 1))) will use port: ${PORT_SECOND_HALF}"
echo "-----------------------------------------------------"

for (( part_idx=0; part_idx<TOTAL_PROCESSES; part_idx++ ))
do
  # Determine the current port based on part_idx
  current_port=""
  if [ "$part_idx" -lt "$MID_POINT" ]; then
    current_port=$PORT_FIRST_HALF
  else
    current_port=$PORT_SECOND_HALF
  fi

  api_base_url="http://localhost:${current_port}/v1"

  log_file="nohup/${task}/${task}_negative_1.5b_semantic_${part_idx}.log"

  echo "Starting process with part_idx=${part_idx}, port=${current_port}, log_file=${log_file}"
  nohup python pipeline/group/generate_negative_groups.py \
    --model 1.5b \
    --temperature 1.0 \
    --exp_config ${task} \
    --model_name test \
    --part_num ${TOTAL_PROCESSES} \
    --part_idx ${part_idx} \
    --save_path test_path \
    --model_name models/${task}-1.5b-sft \
    --api_base ${api_base_url} \
    --api_key EMPTY > ${log_file} 2>&1 &

  sleep 20
done

echo "-----------------------------------------------------"
echo "All ${TOTAL_PROCESSES} processes launched."
echo "You can check the log files in the 'nohup' directory."
echo "To check running processes, you can use: ps aux | grep generate_negative_groups.py"