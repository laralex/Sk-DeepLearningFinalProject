import torch

def transform_to_2d(data, num_blocks):
    '''
    Parameters
    ----------
    data : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, dim_t]
        DESCRIPTION: Output transmission line data with 1d-time.
    num_blocks : TYPE: int
        DESCRIPTION: The number of blocks into which the data will be split
        during transformation into 2d-time data. Time dimension is splitted
        on blocks and reshaped to 2d dimensions with overlaps.

    Returns
    -------
    data : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, 2*dim_t_per_blok, num_bloks]
        DESCRIPTION: Output transmission line data with padded 2d-time.
    '''
    bs, dim_z, dim_t = data.shape
    dim_t_per_block = dim_t//num_blocks
    data = data.view(bs, dim_z, num_blocks, dim_t_per_block).transpose(-2, -1)
    
    data_up = data[:, :, :dim_t_per_block//2, 1:]
    data_down = data[:, :, dim_t_per_block//2:, :-1]
    
    padd_zeros = torch.zeros(bs, dim_z, dim_t_per_block//2, 1, device=data.device)
    
    data_up = torch.cat((data_up, padd_zeros), dim=-1)
    data_down = torch.cat((padd_zeros, data_down), dim=-1)
    
    data = torch.cat((data_down, data), dim=-2)
    data = torch.cat((data, data_up), dim=-2)
    
    return data

def transform_to_1d(data):
    '''
    Reverse operation to transform_to_2d
    
    Parameters
    ----------
    data : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, 2*dim_t_per_blok, num_bloks]
        DESCRIPTION: Data with 2d-time.

    Returns
    -------
    data : TYPE: torch.complex128 tensor of shape [batch_size, dim_z, dim_t]
        DESCRIPTION: Data with 1d-time.
    '''
    dim_padded_t = data.shape[-2]
    data = data[:, :, dim_padded_t//4:dim_padded_t*3//4, :]
    data = data.transpose(-2, -1).flatten(-2, -1)
    return data