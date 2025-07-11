CUDA_VISIBLE_DEVICES=0 python train.py config/DynRT.json

python test.py config/DynRT-test.json

python extract_router.py --config config/DynRT.json --gpu 0

4 Runs in exp:

1) Published code on MMSD
2) Published code on MMSD2.0
3) Published code on MMSD2.0 tau = 1
4) Published code on MMSD2.0 tau = 0.1



The key question MANOVA answers is: Do the groups have different mean scores when we consider all the dependent variables together?

For our data, the groups are the four sarcasm 'types' (non-sarcastic, object, sentimental, situational), and the dependent variables are the ten 'router_vector' values. MANOVA tests whether the overall pattern of the 'router_vector' values is significantly different across the four sarcasm types.

