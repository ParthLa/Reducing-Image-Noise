import numpy as np
def reconstruct_from_noisy_patches(input_dict, shape):
    """
    input_dict: 
    key: 4-tuple: (topleft_row, topleft_col, bottomright_row, bottomright_col): location of the patch in the original image. topleft_row, topleft_col are inclusive but bottomright_row, bottomright_col are exclusive. i.e. if M is the reconstructed matrix. M[topleft_row:bottomright_row, topleft_col:bottomright_col] will give the patch.

    value: 2d numpy array: the image patch.

    shape: shape of the original matrix.
    """
    # Initialization: Initialise M, black_count, mid_count, white_count, mid_total
    x=shape[0]
    y=shape[1]
    black_count=np.zeros((x,y))
    white_count=np.zeros((x,y))
    mid_count=np.zeros((x,y))
    mid_total=np.zeros((x,y))
    M=np.zeros((x,y))
    for topleft_row, topleft_col, bottomright_row, bottomright_col in input_dict: # no loop except this!
        tlr, tlc, brr, brc = topleft_row, topleft_col, bottomright_row, bottomright_col
        patch = input_dict[(tlr, tlc, brr, brc)]

        b=np.where(patch==0,1,0)
        black_count[tlr:brr,tlc:brc] += b

        w=np.where(patch==255,1,0)
        white_count[tlr:brr,tlc:brc] += w

        mc=np.where((0<patch) & (patch<255),1,0)
        mid_count[tlr:brr,tlc:brc] += mc

        mt=np.where((0<patch) & (patch<255),patch,0)
        mid_total[tlr:brr,tlc:brc] += mt
        # change black_count, mid_count, white_count, mid_total here
    # Finally change M here
    g4=np.where(mid_count>0,mid_total,0)
    g3=np.where(mid_count>0,mid_count,10)
    g1=np.divide(g4,g3)
    g5=np.where((g1==0) & (black_count<=white_count),255,0)
    g6=np.where((black_count==0) & (white_count==0) & (mid_count==0),255,0)
    g2=g5-g6
    M=g1+g2
    #g3=np.where(g2<=0,255,0)
    #g4=np.where(g2>0,0,0)
    return M # You have to return the reconstructed matrix (M).

