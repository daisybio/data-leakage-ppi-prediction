import os
import numpy as np
#from keras import utils as ut

def letter2number( letter ):
    '''
    The dataset has 24 amino acids, encoded with letters (every letter but J and O). 
    No switch case in python, hence the ugly if elifs 
    '''
    number = 0
    if letter == 'A':
        number = 1
    elif letter == 'C':
        number = 2
    elif letter == 'D':
        number = 3
    elif letter == 'E':
        number = 4
    elif letter == 'F':
        number = 5
    elif letter == 'G':
        number = 6
    elif letter == 'H':
        number = 7
    elif letter == 'I':
        number = 8
    elif letter == 'K':
        number = 9
    elif letter == 'L':
        number = 10
    elif letter == 'M':
        number = 11
    elif letter == 'N':
        number = 12
    elif letter == 'P':
        number = 13
    elif letter == 'Q':
        number = 14
    elif letter == 'R':
        number = 15
    elif letter == 'S':
        number = 16
    elif letter == 'T':
        number = 17
    elif letter == 'V':
        number = 18
    elif letter == 'W':
        number = 19
    elif letter == 'Y':
        number = 20
    # Special letters here
    elif letter == 'B':
        number = 21
    elif letter == 'U':
        number = 22
    elif letter == 'X':
        number = 23
    elif letter == 'Z':
        number = 24

    return number

def sequence2array( sequence ):
    '''
    Turns a sequence of letters into a sequence of numbers
    to be used in a vector afterwards
    '''
    _list = []
    
    for letter in sequence:
        if letter != None:
            _list.append( letter2number( letter ) )

    return np.asarray( _list )

# def padding( array, max_size ):
#     input_shape = array.shape
#     if input_shape and input_shape[-1] == 1 and len( input_shape ) > 1:
#         input_shape = tuple( input_shape[:-1] )
#     array = array.ravel()
#     padded_array = np.zeros( max_size, dtype=np.uint8 )
#     padded_array = padded_array + array
#     return padded_array

def padding( array, max_size ):
    return np.pad( array, (0, max_size-array.size), 'constant', constant_values=0)

    
# slight modification of https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py
def one_hot(y, max_size, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    Arguments:
      y: class vector to be converted into a matrix
          (integers from 1 to num_classes).
      num_classes: total number of classes.
    Returns:
      A binary matrix representation of the input.
    """
    y = np.array( y, dtype='int' )
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len( input_shape ) > 1:
        input_shape = tuple( input_shape[:-1] )
    y = y.ravel()
    if not num_classes:
        num_classes = np.max( y )
    # the following line has been modified
    n = max_size
    categorical = np.zeros( (n, num_classes), dtype=np.uint8 )
    
    categorical[np.arange( y.shape[0] ), y-1] = 1
    # the following line has been modified
    output_shape = (max_size,) + (num_classes,)
    categorical = np.reshape( categorical, output_shape )
    return categorical
            
def load_data( file_name ):
    data = open( file_name, 'r' )
    
    protein1 = []
    protein2 = []
    output = []

    max_size = int( data.readline() )
    
    for line in data.readlines():
        one_line = line.rstrip('\n').split(' ')
        # one-hotting proteins
        protein1.append( one_hot( sequence2array( one_line[2] ), max_size, num_classes=24 ) )
        protein2.append( one_hot( sequence2array( one_line[3] ), max_size, num_classes=24 ) )
        output.append( int( one_line[4] ) )

    protein1 = np.asarray( protein1 )
    protein2 = np.asarray( protein2 )

    output = np.asarray( output, dtype=np.int8 )

    return protein1, protein2, output


def load_data_embed( file_name ):
    data = open( file_name, 'r' )
    
    protein1 = []
    protein2 = []
    output = []

    max_size = int( data.readline() )

    for line in data.readlines():
        one_line = line.rstrip('\n').split(' ')

        protein1.append( padding( sequence2array( one_line[2] ), max_size ) )
        protein2.append( padding( sequence2array( one_line[3] ), max_size ) )
        output.append( int( one_line[4] ) )

    protein1 = np.asarray( protein1, dtype=np.int8 )
    protein2 = np.asarray( protein2, dtype=np.int8 )

    output = np.asarray( output, dtype=np.int8 )

    return protein1, protein2, output
