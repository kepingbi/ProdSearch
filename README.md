# ProdSearch
Personalized Product Search with Product Reviews

## Amazon Search Dataset:
For each user, we sort his/her purchased items by time and divide items to train/validation/test in a chronological order. 

Download the code and follow the ''Data Preparation'' section in this [link](https://github.com/QingyaoAi/Explainable-Product-Search-with-a-Dynamic-Relation-Embedding-Model) except for splitting data in 4.3.
Use "python ./utils/AmazonDataset/sequentially_split_train_test_data.py <indexed_data_dir> 0.2 0.3" instead.

## Train/Test a TEM model [1]
To train a transformer-based embedding model (TEM) [1], run 

```
python main.py --model_name item_transformer \ # TEM
               --mode train \ # set it to test when evaluating a model
               --pretrain_emb_dir PATH/TO/PRETRAINED_EMB_DIR \ # DATA_DIR for the pretrained word embeddings using reviews. If set to "", embeddings will be trained from scratch 
               --data_dir PATH/TO/DATA \ # <indexed_data_dir> generated when preparing the data, e.g. Amazon/reviews_Sports_and_Outdoors_5.json.gz.stem.nostop/mincount_5
               --input_train_dir PATH/TO/SPLIT_DATA \ # Amazon/reviews_Sports_and_Outdoors_5.json.gz.stem.nostop/min_count5/seq_query_split
               --save_dir PATH/TO/SAVE/TRAINED/MODELS \ # where to store or load models. 
               --decay_method adam \ # use the weight decay method in adam instead of noam
               --max_train_epoch 20 --lr 0.0005 --batch_size 384 \
               --uprev_review_limit 20 \ # the number of historically purchased items used for user.
               --embedding_size 128 \
               --inter_layers 1 \ # the number of layers for transformer
               --ff_size 512 --heads 8 # other hyper-parameters that may need tune for training. 
```
## Train/Test a RTM model [2]
If you want to run a review-based transformer model (RTM) [2], simply use a different model_name: 
```
--model_name review_transformer
```
## References
[1] Keping Bi, Qingyao Ai, W. Bruce Croft. A Transformer-based Embedding Model for Personalized Product Search. In Proceedings of SIGIR'20.

[2] Keping Bi, Qingyao Ai, W. Bruce Croft. Learning a Fine-Grained Review-based Transformer Model for Personalized Product Search. In Proceedings of SIGIR'21.
