

def write_SPRINT(path, data):
    file = open(path, 'w')
    for pair in data:
        file.write(f'{pair[0]} {pair[1]}\n')
    file.close()

