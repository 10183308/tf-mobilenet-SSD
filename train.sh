python3 train_ssd_mobilenet.py \
--train_dir=./logs \
--checkpoint_path=./ckpt/mobilenet_v1_1.0_224.ckpt \
--dataset_name=pascalvoc_2007 \
--dataset_split_name=train \
--dataset_dir=./tf_records \
--batch_size=32 \
--max_number_of_steps=100 \
--gpu_memory_fraction=0.1 \
--train_on_cpu=True
