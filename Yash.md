CUDA_VISIBLE_DEVICES=0 python train.py config/DynRT.json

python test.py config/DynRT-test.json

python extract_router.py --config config/DynRT.json --gpu 0

4 Runs in exp:

1) Published code on MMSD
2) Published code on MMSD2.0
3) Published code on MMSD2.0 tau = 1
4) Published code on MMSD2.0 tau = 0.1