python inference.py \
    --json_file  ./example/ride_horse/cond.json \
    --image_path ./example/ride_horse/first.png  \
    --video_name ride_horse \
    --control_weight_path ./models/LongVie/control.safetensors \
    --dit_weight_path ./models/LongVie/dit.safetensors
