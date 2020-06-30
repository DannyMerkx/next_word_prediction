#! /bin/bash
#SBATCH --partition=long

source /home/dmerkx/torch_env/bin/activate

export CUDA_VISIBLE_DEVICES=1

nohup python -u nwp_gru.py -model_ids 1 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/gru_tf/ > gru1_.out &
sleep 1
nohup python -u nwp_gru.py -model_ids 2 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/gru_tf/ > gru2_.out &
sleep 1
nohup python -u nwp_gru.py -model_ids 3 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/gru_tf/ > gru3_.out &
sleep 1
nohup python -u nwp_gru.py -model_ids 4 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/gru_tf/ > gru4_.out &
sleep 1
nohup python -u nwp_gru.py -model_ids 5 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/gru_tf/ > gru5_.out &
sleep 1
nohup python -u nwp_gru.py -model_ids 6 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/gru_tf/ > gru6_.out &
sleep 1
nohup python -u nwp_gru.py -model_ids 7 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/gru_tf/ > gru7_.out &
sleep 1
nohup python -u nwp_gru.py -model_ids 8 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/gru_tf/ > gru8_.out &
