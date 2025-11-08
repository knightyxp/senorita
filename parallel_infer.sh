
export CUDA_VISIBLE_DEVICES=0,1,2,3


torchrun --nproc_per_node=4 parallel_control_cogvideox_pipeline.py \
  --tasks_json /scratch3/yan204/yxp/VideoX_Fun/data/test_json/4tasks_rem_add_swap_local-style_test_with_caption.json\
  --output_dir results/senorita_bench_test \
  --model_root /scratch3/yan204/models/cogvideox-5b-i2v \
  --control_checkpoint /scratch3/yan204/models/senorita-2m/models_half/ff_controlnet_half.pth \
  --seed 0 \
  --num_frames 33 \