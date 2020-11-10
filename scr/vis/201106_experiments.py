cmds = [
    "python train.py --dataset data/shapes_train.h5 --encoder small --name B",
    "python train.py --dataset data/shapes_train.h5 --encoder small --name B_neg_in --same-ep-neg --only-same-ep-neg",
    "python train.py --dataset data/shapes_train.h5 --encoder small --name B_neg_in_out --same-ep-neg",
    "python train.py --dataset data/shapes_imm_train.h5 --encoder small --name B_imm",
    "python train.py --dataset data/shapes_imm_train.h5 --encoder small --name B_imm_neg_in --same-ep-neg --only-same-ep-neg",
    "python train.py --dataset data/shapes_imm_train.h5 --encoder small --name B_imm_neg_in_out --same-ep-neg",
    "python train.py --dataset data/shapes_imm_train.h5 --encoder small --name split_MLP --split-mlp",
    "python train.py --dataset data/shapes_imm_train.h5 --encoder small --name split_MLP_neg_in --same-ep-neg --only-same-ep-neg --split-mlp",
    "python train.py --dataset data/shapes_imm_train.h5 --encoder small --name split_MLP_neg_in_out --same-ep-neg --split-mlp"
]

cmds = [
    "python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/B --num-steps 1",
    "python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/B_neg_in --num-steps 1",
    "python eval.py --dataset data/shapes_eval.h5 --save-folder checkpoints/B_neg_in_out --num-steps 1",
    "python eval.py --dataset data/shapes_imm_eval.h5 --save-folder checkpoints/B_imm --num-steps 1",
    "python eval.py --dataset data/shapes_imm_eval.h5 --save-folder checkpoints/B_imm_neg_in --num-steps 1",
    "python eval.py --dataset data/shapes_imm_eval.h5 --save-folder checkpoints/B_imm_neg_in_out --num-steps 1",
    "python eval.py --dataset data/shapes_imm_eval.h5 --save-folder checkpoints/split_MLP --num-steps 1",
    "python eval.py --dataset data/shapes_imm_eval.h5 --save-folder checkpoints/split_MLP_neg_in --num-steps 1",
    "python eval.py --dataset data/shapes_imm_eval.h5 --save-folder checkpoints/split_MLP_neg_in_out --num-steps 1"
]
