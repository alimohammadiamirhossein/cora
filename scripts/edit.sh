#!/bin/bash

# Load configuration
CONFIG_FILE="scripts/config.sh"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found."
    exit 1
fi
source "$CONFIG_FILE"

source_steps_script() {
    case "$num_of_timesteps" in
        3) source ./scripts/steps_mode/3_steps.sh ;;
        4) source ./scripts/steps_mode/4_steps.sh ;;
        5) source ./scripts/steps_mode/5_steps.sh ;;
        10) source ./scripts/steps_mode/10_steps.sh ;;
        *)
            echo "Error: No matching script for timesteps=$timesteps"
            exit 1
            ;;
    esac
}

# Source the appropriate script
source_steps_script

# Construct the Python command dynamically
cmd="python main.py \
    --prompts_file \"$prompts_file\" \
    --seed ${seed} \
    --output_dir \"$output_dir\" \
    --num_of_timesteps \"$num_of_timesteps\" \
    --max_norm_zs ${max_norm_zs[@]} \
    --noise_shift_delta \"$noise_shift_delta\" \
    --noise_timesteps ${noise_timesteps[@]} \
    --timesteps ${timesteps[@]} \
    --mode \"$mode\" \
    --movement_intensifier \"$movement_intensifier\" \
    --dift_timestep \"$dift_timestep\" \
    --w1 \"$w1\" \
    --num_steps_inversion \"$num_steps_inversion\" \
    --step_start \"$step_start\""

if [ "$support_new_object" = true ]; then
    cmd="$cmd --support_new_object"
fi

if [ "$structural_alignment" = true ]; then
    cmd="$cmd --structural_alignment"
fi

if [ "$apply_dift_correction" = true ]; then
    cmd="$cmd --apply_dift_correction"
fi

# Execute the command
eval $cmd
