import numpy as np 

def csc_row_array_col_array(arr,row,col):
    
    indices = []
    indptr = np.zeros(len(col)+1,dtype=int)
    indptr[0] = 0
    ind_list = []
    for i,c in enumerate(col):
        inds = []
        for r in range(len(row)):
            s = np.searchsorted(arr.indices[arr.indptr[c]:arr.indptr[c+1]],row[r]) + arr.indptr[c]
            if arr.indices[s]==row[r]:
                inds.append(s)
                indices.append(r)
        ind_list.extend(inds)
        indptr[i+1] = indptr[i] + len(inds)
    indices = np.array(indices)
    data = arr.data[ind_list]
    return (data,indices,indptr)