



def write_deepPPI(path, all_pairs, seq_dict, max_len=1166):
    import random
    random.shuffle(all_pairs)
    removed_pairs = 0
    with open(path, 'w') as f:
        for pair in all_pairs:
            len1 = len(seq_dict[pair[0]])
            len2 = len(seq_dict[pair[1]])
            if len1 < max_len and len2 < max_len:
                f.write(f'{pair[0]} {pair[1]} {seq_dict[pair[0]]} {seq_dict[pair[1]]} {pair[2]}\n')
            else:
                removed_pairs += 1
    print(f'Removed {removed_pairs}/{len(all_pairs)} because of length restriction')
    with open(path, 'r') as original:
        data = original.read()
    with open(path, 'w') as modified:
        modified.write(f"{max_len}\n" + data)