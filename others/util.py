import gzip
from others.logging import logger

def load_pretrain_embeddings(fname):
    embeddings = []
    word_index_dic = dict()
    with gzip.open(fname, 'rt') as fin:
        count = int(fin.readline().strip())
        emb_size = int(fin.readline().strip())
        line_no = 0
        for line in fin:
            arr = line.strip(' ').split('\t')#the first element is empty
            word_index_dic[arr[0]] = line_no
            line_no += 1
            vector = arr[1].split()
            vector = [float(x) for x in vector]
            embeddings.append(vector)
    logger.info("Loading {}".format(fname))
    logger.info("Count:{} Embeddings size:{}".format(len(embeddings), len(embeddings[0])))
    return word_index_dic, embeddings

def load_user_item_embeddings(fname):
    embeddings = []
    #with gzip.open(fname, 'rt') as fin:
    with open(fname, 'r') as fin:
        count = int(fin.readline().strip())
        emb_size = int(fin.readline().strip())
        for line in fin:
            arr = line.strip().split(' ')
            vector = [float(x) for x in arr]
            embeddings.append(vector)
    logger.info("Loading {}".format(fname))
    logger.info("Count:{} Embeddings size:{}".format(len(embeddings), len(embeddings[0])))
    return embeddings

def pad(data, pad_id, width=-1):
    if (width == -1):
        width = max(len(d) for d in data)
    rtn_data = [d[:width] + [pad_id] * (width - len(d)) for d in data]#if width < max(len(d)) of data
    return rtn_data

def pad_3d(data, pad_id, dim=1, width=-1):
    #dim = 1 or 2
    if dim < 1 or dim > 2:
        return data
    if (width == -1):
        if (dim == 1):
            #dim 0,2 is same across the batch
            width = max(len(d) for d in data)
        elif (dim == 2):
            #dim 0,1 is same across the batch
            for entry in data:
                width = max(width, max(len(d) for d in entry))
        #print(width)
    if dim == 1:
        rtn_data = [d[:width] + [[pad_id] * len(data[0][0])] * (width - len(d)) for d in data]
    elif dim == 2:
        rtn_data = []
        for entry in data:
            rtn_data.append([d[:width] + [pad_id] * (width - len(d)) for d in entry])
    return rtn_data

def pad_4d_dim1(data, pad_id, width=-1):
    if (width == -1):
        #max width of dim1
        width = max(width, max(len(d) for d in data))
    #print(width)
    rtn_data = [d[:width] + [[[pad_id] * len(data[0][0][0])]] * (width - len(d)) for d in data]
    return rtn_data

def pad_4d_dim2(data, pad_id, width=-1):
    #only handle padding to dim = 2
    if (width == -1):
        #max width of dim2
        for entry in data:
            width = max(width, max(len(d) for d in entry))
    #print(width)
    rtn_data = []
    for entry_dim1 in data:
        rtn_data.append([d[:width] + [[pad_id] * len(data[0][0][0])] * (width - len(d)) for d in entry_dim1])
    return rtn_data

def main():
    data = [[[[2,2,2],[2,2,2]],[[2,2,2]]],[[[2,2,2]]]]
    rtn = pad_4d_dim1(data, -1)
    rtn = pad_4d_dim2(rtn, -1)
    print(rtn)

if __name__ == "__main__":
    main()


