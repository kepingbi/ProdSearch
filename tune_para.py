import sys
import os
import argparse

config_str = """#!/bin/bash

#SBATCH --partition=titanx-long #m40-long #titanx-long #1080ti-long #2080ti-long   # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=64000    # Memory in MB per node allocated
#SBATCH --ntasks-per-node=4
"""
datasets = ["Beauty",#dataset, divided into how many parts
            "Cell_Phones_and_Accessories",
            "Clothing_Shoes_and_Jewelry",
            "Health_and_Personal_Care",
            "Home_and_Kitchen",
            "Sports_and_Outdoors"
            "Electronics",
            "Movies_and_TV",
            "CDs_and_Vinyl", #CD too large
            "Kindle_Store", #Kindle too large
            ]

WORKING_DIR="/mnt/nfs/scratch1/kbi/review_transformer/working/Amazon"
OUTPUT_DIR="/mnt/nfs/scratch1/kbi/review_transformer/output/Amazon"

script_path = "python main.py"
#CONST_CMD_ARR = [("data_dir", data_dir),("input_train_dir", input_train_dir)]
#CONST_CMD = " ".join(["--{} {}".format(x[0], x[1]) for x in CONST_CMD_ARR])
pretrain_pv_root_dir = "/mnt/nfs/scratch1/kbi/review_transformer/working/paragraph_embeddings/reviews_##_5.json.gz.stem.nostop/min_count5/"
pv_path = "batch_size256.negative_sample5.learning_rate0.5.embed_size128.steps_per_checkpoint400.max_train_epoch20.L2_lambda0.0.net_structpv_hdc./"
#small_pvc_path = "batch_size256.negative_sample5.learning_rate0.5.embed_size128.use_local_contextTrue.steps_per_checkpoint400.max_train_epoch20.L2_lambda0.0.net_structcdv_hdc."
small_pvc_path = "batch_size256.negative_sample5.learning_rate0.5.embed_size128.subsampling_rate1e-05.use_local_contextTrue.steps_per_checkpoint400.max_train_epoch20.L2_lambda0.0.net_structcdv_hdc."
large_pvc_path = "batch_size512.negative_sample5.learning_rate0.5.embed_size128.subsampling_rate1e-06.use_local_contextTrue.steps_per_checkpoint400.max_train_epoch20.L2_lambda0.0.net_structcdv_hdc." #subsampling_rate can be 1e-5,1e-6,1e-7

#pretrain_pv_emb_dir = "/mnt/nfs/scratch1/kbi/review_transformer/working/paragraph_embeddings/reviews_##_5.json.gz.stem.nostop/min_count5/batch_size256.negative_sample5.learning_rate0.5.embed_size128.steps_per_checkpoint400.max_train_epoch20.L2_lambda0.0.net_structpv_hdc."
#pretrain_pvc_emb_dir = "/mnt/nfs/scratch1/kbi/review_transformer/working/paragraph_embeddings/reviews_##_5.json.gz.stem.nostop/min_count5/batch_size256.negative_sample5.learning_rate0.5.embed_size128.use_local_contextTrue.steps_per_checkpoint400.max_train_epoch20.L2_lambda0.0.net_structcdv_hdc."

para_names = ['pretrain_type', 'review_encoder_name', 'max_train_epoch', 'lr', 'warmup_steps', 'batch_size', 'valid_candi_size', \
        'embedding_size', 'review_word_limit', 'iprev_review_limit', 'dropout', \
        'prod_freq_neg_sample', 'fix_emb', 'pos_weight', 'l2_lambda', 'ff_size', 'inter_layers', 'use_user_emb', 'use_item_emb']
short_names = ['pretrain', 'enc', 'me', 'lr', 'ws', 'bs', 'vcs', 'ebs', \
                'rwl', 'irl', 'drop', 'prodneg', 'fixemb', 'poswt', 'lambda', 'ff', 'ly', 'ue', 'ie']

paras = [
        ('Cell_Phones_and_Accessories', 'pvc', 'fs', 20, 0.002, 8000, 128, 500, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, True, True),
        ('Cell_Phones_and_Accessories', 'pvc', 'pvc', 20, 0.002, 8000, 128, 500, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, True, True),
        #('Cell_Phones_and_Accessories', 'pv', 'fs', 20, 0.002, 8000, 128, 500, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, False, False),
        #('Cell_Phones_and_Accessories', 'pv', 'fs', 20, 0.002, 8000, 128, 500, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, True, True),
        #('Cell_Phones_and_Accessories', 'pvc', 'fs', 20, 0.002, 8000, 128, 500, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, False, False),
        #('Cell_Phones_and_Accessories', 'pvc', 'pvc', 20, 0.002, 8000, 128, 500, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, False, False),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 0.002, 8000, 128, 500, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, False, False),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 0.002, 8000, 128, 500, 128, 100, 30, 0.1, False, True, False, 0, 512, 2, False, False),

        #('Clothing_Shoes_and_Jewelry', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, True, True),
        #('Electronics', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, True, True),
        #('Movies_and_TV', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, True, True),
        #('Kindle_Store', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, True, True),
        #('Sports_and_Outdoors', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, True, True),

        #('Clothing_Shoes_and_Jewelry', 'pv', 'pv', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, True, False, 0, 512, 2, True, True),
        #('Electronics', 'pv', 'pv', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, True, False, 0, 512, 2, True, True),
        #('Movies_and_TV', 'pv', 'pv', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, True, False, 0, 512, 2, True, True),
        #('Kindle_Store', 'pv', 'pv', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, True, False, 0, 512, 2, True, True),
        #('Sports_and_Outdoors', 'pv', 'pv', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, True, False, 0, 512, 2, True, True),

        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, False, False),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, True, False),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2, False, True),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 1e-4, 512, 2, False, False),

        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 500, 128, 100, 40, 0.1, False, False, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 500, 256, 100, 40, 0.1, False, False, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 500, 128, 100, 40, 0.1, False, False, False, 1e-4, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 500, 128, 100, 40, 0.1, False, False, False, 0, 512, 4),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 0.002, 8000, 128, 1000, 128, 100, 40, 0.1, False, True, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 0.002, 8000, 128, 1000, 128, 100, 40, 0.1, False, False, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2),
        #('Health_and_Personal_Care', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2),
        #('Clothing_Shoes_and_Jewelry', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2),
        #('Sports_and_Outdoors', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.01, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.005, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 96, 100, 30, 0.1, False, False, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 64, 100, 30, 0.1, False, False, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 512, 1),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 256, 2),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 0.002, 8000, 128, 1000, 128, 100, 30, 0.1, False, False, False, 0, 256, 1),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, True, False, 0),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, True, False, 0),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, False, False, 0),
        #('Cell_Phones_and_Accessories', 'pvc', 'pv', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, True, False, 0),
        #('Cell_Phones_and_Accessories', 'pvc', 'pvc', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, False, False, 0),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, False, False, 0),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, False, True, 0),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, False, False, 0.001),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 10000, 128, 1000, 128, 150, 40, 0.1, False, False, False, 0),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 10000, 128, 1000, 128, 200, 40, 0.1, False, False, False, 0),
        #('Cell_Phones_and_Accessories', pretrain_pv_emb_dir, 'pv', 30, 3000, 128, 1000, 128, 100, 40, 0.1, True, True, True),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 8000, 32, 1000, 128, 100, 40, 0.1, False, True, False, 0),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 8000, 64, 1000, 128, 100, 40, 0.1, False, True, False, 0),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 8000, 128, 1000, 128, 100, 40, 0.1, False, True, False, 0),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 8000, 128, 1000, 128, 100, 40, 0.1, False, True, False, 0.001),
        #('Cell_Phones_and_Accessories', 'pv', 'pv', 20, 8000, 128, 1000, 128, 100, 40, 0.1, False, True, False, 0.0001),
        #('Cell_Phones_and_Accessories', pretrain_pv_emb_dir, 'pv', 30, 3000, 128, 1000, 128, 100, 40, 0.1, False, True, True),
        #('Cell_Phones_and_Accessories', pretrain_pv_emb_dir, 'pv', 30, 3000, 128, 1000, 128, 100, 40, 0.1, False, False, False),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 8000, 128, 1000, 128, 100, 40, 0.1, False, False, False, 0),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 8000, 128, 1000, 128, 100, 40, 0.1, False, False, False, 0.001),
        #('Cell_Phones_and_Accessories', None, 'fs', 20, 8000, 128, 1000, 128, 100, 40, 0.1, False, False, False, 0.0001),

        #('Clothing_Shoes_and_Jewelry', None, 'fs', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, False, False, 0),
        #('Clothing_Shoes_and_Jewelry', None, 'fs', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, False, False, 0.001),
        #('Clothing_Shoes_and_Jewelry', None, 'fs', 20, 10000, 128, 1000, 128, 100, 40, 0.2, False, False, False, 0),
        #('Clothing_Shoes_and_Jewelry', None, 'fs', 20, 10000, 128, 1000, 128, 100, 40, 0.3, False, False, False, 0),
        #('Clothing_Shoes_and_Jewelry', 'pv', 'pv', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, True, False, 0),
        #('Clothing_Shoes_and_Jewelry', 'pvc', 'pv', 20, 10000, 128, 1000, 128, 100, 40, 0.1, False, True, False, 0),
        #('Clothing_Shoes_and_Jewelry', 'fs', 20, 0, 8000, 128, 1000, 128, 100, 40, 0.1, 0.9),
        #('Movies_and_TV', 'fs', 20, 0, 8000, 128, 1000, 128, 100, 40, 0.1, 0.9),
        #('Sports_and_Outdoors', 'fs', 20, 0, 8000, 128, 1000, 128, 100, 40, 0.1, 0.9),
        #('Kindle_Store', 'fs', 20, 0, 8000, 128, 1000, 128, 100, 40, 0.1, 0.9),
        #('Cell_Phones_and_Accessories', 'fs', 20, 0, 8000, 128, 1000, 128, 100, 40, 0.1, 0.9),

        #('Cell_Phones_and_Accessories', 'pv', 10, 2, 3000, 32, 1000, 128, 100, 30, 0.1, 0.9),
        #('Cell_Phones_and_Accessories', 'pvc', 10, 2, 3000, 32, 1000, 128, 100, 30, 0.1, 0.9),
        #('Cell_Phones_and_Accessories', 'fs', 30, 0, 8000, 128, 1000, 128, 100, 30, 0.1, 0.9),
        #('Cell_Phones_and_Accessories', 'avg', 20, 0, 8000, 128, 1000, 128, 100, 30, 0.1, 0.9),

        #('CDs_and_Vinyl', 'pv', 10, 2, 3000, 32, 1000, 128, 100, 30, 0.1, 0.9),
        #('CDs_and_Vinyl', 'pvc', 10, 2, 3000, 32, 1000, 128, 100, 30, 0.1, 0.9),
        #('CDs_and_Vinyl', 'fs', 20, 0, 8000, 128, 1000, 128, 100, 30, 0.1, 0.9),
        #('CDs_and_Vinyl', 'avg', 20, 0, 8000, 128, 1000, 128, 100, 30, 0.1, 0.9),

        #('Electronics', 'pv', 10, 2, 3000, 32, 1000, 128, 100, 30, 0.1, 0.9),
        #('Electronics', 'pvc', 10, 2, 3000, 32, 1000, 128, 100, 30, 0.1, 0.9),
        #('Electronics', 'fs', 20, 0, 8000, 128, 1000, 128, 100, 30, 0.1, 0.9),

        #('Kindle_Store', 'pv', 10, 2, 3000, 32, 1000, 128, 100, 30, 0.1, 0.9),
        #('Kindle_Store', 'pvc', 10, 2, 3000, 32, 1000, 128, 100, 30, 0.1, 0.9),

        ]

if __name__ == '__main__':
    fscript = open("run_model.sh", 'w')
    parser = argparse.ArgumentParser()
    #parser.add_argument("--log_dir", type=str, default='exp_log')
    args = parser.parse_args()
    #os.system("mkdir -p %s" % args.log_dir)
    os.system("mkdir -p script")
    os.system("mkdir -p log")
    job_id = 0
    for para in paras:
        cmd_arr = []
        cmd_arr.append(script_path)
        dataset = para[0]
        if para[1] == 'pv':
            pretrain_emb_dir = os.path.join(pretrain_pv_root_dir.replace('##', dataset), pv_path)
        elif para[1] == 'pvc':
            if dataset in datasets[:-4]:
                pretrain_emb_dir = os.path.join(pretrain_pv_root_dir.replace('##', dataset), small_pvc_path)
            else:
                pretrain_emb_dir = os.path.join(pretrain_pv_root_dir.replace('##', dataset), large_pvc_path)
        else:
            pretrain_emb_dir = "None"
        dataset_name = "reviews_%s_5.json.gz.stem.nostop" % dataset
        data_dir = "%s/%s/min_count5" % (WORKING_DIR, dataset_name)
        input_train_dir = os.path.join(data_dir, "seq_query_split")
        #input_train_dir = os.path.join(data_dir, "query_split")
        cmd_arr.append('--data_dir {}'.format(data_dir))
        cmd_arr.append('--pretrain_emb_dir {}'.format(pretrain_emb_dir))
        cmd_arr.append('--input_train_dir {}'.format(input_train_dir))
        output_path = "%s/%s" % (OUTPUT_DIR, dataset_name)
        model_name = "_".join(["{}{}".format(x,y) for x,y in zip(short_names, para[1:])])
        save_dir = os.path.join(output_path, model_name)
        cur_cmd_option = " ".join(["--{} {}".format(x,y) for x,y in zip(para_names[1:], para[2:])])
        cmd_arr.append(cur_cmd_option)
        cmd_arr.append("--save_dir %s" % save_dir)
        model_name = "{}_{}".format(dataset, model_name)
        #cmd_arr.append("--log_file %s/%s.log" % (args.log_dir, model_name))
        #cmd_arr.append("&> %s/%s.log \n" % (args.log_dir, model_name))
        cmd = " " .join(cmd_arr)
        #print(cmd)
        #os.system(cmd)
        fname = "script/%s.sh" % model_name
        with open(fname, 'w') as fout:
            fout.write(config_str)
            fout.write("#SBATCH --job-name=%d.sh\n" % job_id)
            fout.write("#SBATCH --output=log/%s.txt\n" % model_name)
            fout.write("#SBATCH -e log/%s.err.txt\n" % model_name)
            fout.write("\n")
            fout.write(cmd)
            fout.write("\n\n")
            fout.write("exit\n")

        fscript.write("sbatch %s\n" % fname)
        job_id += 1
    fscript.close()



