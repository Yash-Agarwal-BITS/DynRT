CUDA_VISIBLE_DEVICES=0 python train.py config/DynRT.json

python test.py config/DynRT-test.json

python extract_router.py \
    --model_path exp/06-24-11_51_33/checkpoints/model_best.pth.tar \
    --config_path ./config/DynRT.json \
    --annotations_path ./annotations.json \
    --image_tensor_path ./image_tensor \
    --output_path ./router_values.json \
    --layer_to_hook 0



4 Runs in exp:

1) Published code on MMSD
2) Published code on MMSD2.0
3) Published code on MMSD2.0 tau = 1
4) Published code on MMSD2.0 tau = 0.1
