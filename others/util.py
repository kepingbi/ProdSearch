
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

