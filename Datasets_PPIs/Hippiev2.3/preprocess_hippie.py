import pandas as pd


def export_hippie_IDs():
    hippie_PPIs = pd.read_csv('hippie_current.txt', sep='\t', header=None, names=['ID_A', 'Entrez_A', 'ID_B', 'Entrez_B', 'confidence', 'experiments'])
    unique_prots = set(hippie_PPIs['ID_A'])
    unique_prots = unique_prots.union(set(hippie_PPIs['ID_B']))
    print(f'{len(unique_prots)} unique proteins')

    # export all unique protein IDs to map them to UniProt Identifiers
    with open('all_unique_ids.txt', 'w') as f:
        for prot in unique_prots:
            if isinstance(prot, str):
                f.write(prot+',')


def create_PPI_file():
    mapping_dict = dict()
    with open('hippie_mapping.tsv') as f:
        for line in f:
            if line.startswith('From'):
                continue
            else:
                id_from, id_to = line.strip().split('\t')
                mapping_dict[id_from] = id_to
    hippie_PPIs = pd.read_csv('hippie_current.txt', sep='\t', header=None, names=['ID_A', 'Entrez_A', 'ID_B', 'Entrez_B', 'confidence', 'experiments'])
    ppis_added = set()
    with open('hippie_PPIs.tsv', 'w') as f:
        f.write('ID_A\tID_B\tConfidence\n')
        for index, row in hippie_PPIs.iterrows():
            uniprot_a = mapping_dict.get(row['ID_A'])
            uniprot_b = mapping_dict.get(row['ID_B'])
            confidence = row['confidence']
            if uniprot_a is not None and uniprot_b is not None:
                if (uniprot_a, uniprot_b) not in ppis_added and uniprot_a != uniprot_b:
                    f.write(f'{uniprot_a}\t{uniprot_b}\t{confidence}\n')
                    ppis_added.add((uniprot_a, uniprot_b))
                    ppis_added.add((uniprot_b, uniprot_a))


if __name__ == '__main__':
    # export_hippie_IDs()
    # Upload all_unique_ids.txt to UniProt mapping service, export results file as hippie_mapping.tsv
    create_PPI_file()