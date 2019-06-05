import numpy as np 
from functools import reduce
from operator import mul




def convert_to_2d(inds,shape,compressed_shape):
    
    rows = []
    cols = []
    operations = reduce(mul,(len(ind) for ind in inds)) - 1
    key_vals = [ind[0] for ind in inds]
    previous_key = tuple(key_vals)
    last_axis = len(key_vals) - 1
    row,col = calculate_2d(key_vals,shape,compressed_shape)
    rows.append(row)
    cols.append(col)
    pos = len(inds)-1
    i = 0
    j = [0 for ind in inds]
    j[-1] = 1
    while i < operations:
        if key_vals[pos]==inds[pos][-1]:
            key_vals[pos] = inds[pos][0]
            j[pos] = 0
            pos -= 1
            j[pos] += 1
        else:
            key_vals[pos] = inds[pos][j[pos]]
            i += 1
            pos = len(inds)-1
            j[pos] +=1
            row,col = calculate_2d(key_vals,shape,compressed_shape)
            if row!=rows[-1]:
                rows.append(row)
            if col!=cols[-1]:
                cols.append(col)
    rows = np.unique(rows)
    cols = np.unique(cols)
    
    return rows,cols




def calculate_2d(idx,shape,compressed_shape):
    """converts nd-coords into 2d"""
    sl = len(shape) 
    row_idx = idx[:sl//2+1] if sl%2==1 else idx[:sl//2]
    col_idx = idx[sl//2+1:] if sl%2==1 else idx[sl//2:]
    first = shape[:sl//2+1] if sl%2==1 else shape[:sl//2]
    second = shape[sl//2+1:] if sl%2==1 else shape[sl//2:]
    row = 0
    for i in range(len(row_idx)-1):
        if i==len(row_idx)-2:
            row += row_idx[i] * np.prod(first[i+1:]) + row_idx[i+1]
        else:
            row += row_idx[i] * np.prod(first[i+1:])
    col = 0
    for i in range(len(col_idx)-1):
        if i==len(col_idx)-2:
            col += col_idx[i] * np.prod(second[i+1]) + col_idx[i+1]
        else:
            col += col_idx[i] * np.prod(second[i+1])
    return row,col

def uncompress_dimension(indptr,indices):
    """converts an index pointer array into an array of coordinates"""
    uncompressed = np.empty(indices.shape[0],dtype=np.intp)
    position = 0
    for i in range(len(indptr)-1):
        inds = indices[indptr[i]:indptr[i+1]].shape[0] 
        uncompressed[np.arange(position,inds + position)] = i
        position += inds
    return uncompressed

def compress_dimension(coords,indptr):
    """converts an array of coordinates into an index pointer array"""
    indptr[0] = 0
    j = 0
    if coords[0] == 0:
        start = 1
    else:
        start = 0
    for i in range(start,len(coords)):
        if i == 0:
            x = coords[0]
        else:
            x = coords[i]-coords[i-1]
        if x > 0:
            for k in range(x):
                j+=1
                indptr[j] = i
    indptr[j+1:] = coords.shape[0]
    return indptr
