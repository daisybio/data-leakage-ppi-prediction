


def read_NO(file_name):
    #    # read sample from a file
    no = []
    header_dict = {}
    seq_dict = {}
    counter = 0
    with open(file_name, 'r') as fp:
        i = 0
        last_seq = ''
        for line in fp:
            if i % 2 == 0:
                uniprot_id = line.strip().split('|')[1].split(':')[1]
                if uniprot_id == '':
                    counter += 1
                no.append(uniprot_id)
                header_dict[uniprot_id] = line
                last_seq = uniprot_id
            else:
                seq_dict[last_seq] = line
            i = i + 1
    print(f'{counter} unmatched IDs ...')
    return no, header_dict, seq_dict


def write_deepFE(pathA, pathB, data, header_dict, seq_dict):
    proteinA = open(pathA, 'w')
    proteinB = open(pathB, 'w')

    for pair in data:
        proteinA.write(header_dict[pair[0]] +
                       seq_dict[pair[0]])
        proteinB.write(header_dict[pair[1]] +
                       seq_dict[pair[1]])
    proteinA.close()
    proteinB.close()
