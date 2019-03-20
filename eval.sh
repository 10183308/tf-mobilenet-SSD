python3 eval_ssd_mobilenet.py --eval_dir=./logs \
--checkpoint_path=./logs \
--dataset_dir=./tf_records \
--dataset_name=pascalvoc_2007 \
--dataset_split_name=test \
--batch_size=1 \
--max_num_batches=10
