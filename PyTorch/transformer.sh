#! /bin/bash
#SBATCH --partition=long

source /home/dmerkx/torch_env/bin/activate

export CUDA_VISIBLE_DEVICES=1

nohup python -u nwp_tf.py -model_ids 1 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/tf_8layer/ > tf1_.out &
sleep 1
nohup python -u nwp_tf.py -model_ids 2 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/tf_8layer/ > tf2_.out &
sleep 1
nohup python -u nwp_tf.py -model_ids 3 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/tf_8layer/ > tf3_.out &
sleep 1
nohup python -u nwp_tf.py -model_ids 4 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/tf_8layer/ > tf4_.out &
sleep 1
nohup python -u nwp_tf.py -model_ids 5 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/tf_8layer/ > tf5_.out &
sleep 1
nohup python -u nwp_tf.py -model_ids 6 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/tf_8layer/ > tf6_.out &
sleep 1
nohup python -u nwp_tf.py -model_ids 7 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/tf_8layer/ > tf7_.out &
sleep 1
nohup python -u nwp_tf.py -model_ids 8 -param xavier -bias none -results_loc /vol/tensusers3/dmerkx/next_word_prediction/tf_8layer/ > tf8_.out &
