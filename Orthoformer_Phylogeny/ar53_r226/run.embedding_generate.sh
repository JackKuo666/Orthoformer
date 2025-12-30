python ../generate_embeddings_cli.py \
    --model_dir ../../foundation_model/model/model_3M_2048_v10 \
    --dataset_path datasets/ar53.dataset \
    --output_dir embeddings \
    --batch_size 32 \
    --model_max_length 2048 \
    --use_alibi \
    --output_mode tokens