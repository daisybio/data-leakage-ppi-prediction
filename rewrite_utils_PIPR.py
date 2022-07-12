


def write_pipr(path, all_pairs):
    with open(path, 'w') as f:
        for pair in all_pairs:
            f.write(f'{pair[0]}\t{pair[1]}\t{pair[2]}\n')