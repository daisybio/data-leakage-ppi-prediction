
yeast_fasta = open('../../Datasets_PPIs/SwissProt/yeast_swissprot.fasta', 'r')
fasta_sprint = open('../../Datasets_PPIs/SwissProt/yeast_swissprot_oneliner.fasta', 'w')

last_id=''
last_seq=''
first_line=True
for line in yeast_fasta:
    if line.startswith('>'):
        if first_line:
            last_id = line.strip().split('|')[1]
            first_line = False
        else:
            fasta_sprint.write(f'>{last_id}\n{last_seq}\n')
            last_seq = ''
            last_id = line.strip().split('|')[1]
    else:
        last_seq += line.strip()
