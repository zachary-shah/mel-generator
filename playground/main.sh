# main shell script to run train data

# first, make condas environment appropriate for creating all data
conda env create -f processing_env.yml
conda env create -f control_env.yml

# run data processing script in training environment
conda activate processing_env
python make_cnet_dataset.py --audio_dir "raw-audio/" --train_data_dir "train-data/" --prompt_file "prompt_labels.json"
conda deactivate

# run training script in training environment
conda activate control_env
python cnet_riff_training.py 