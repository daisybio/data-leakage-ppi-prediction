import pandas as pd
import numpy as np
import scipy as sp
import sentencepiece as spm
import json


path  = 'all.tab'
int_path = 'interact.json'
seq_path = 'pretraining_data.txt'
model_path = 'm_reviewed.model'

def filter_seqs():
	"""
	Filter sequences by length

	"""
	data = pd.read_csv(path,sep='\t')
	seq_ls = [seq for seq in data['Sequence'] if len(seq)<1024]
	with open(seq_path, 'w') as filehandle:
		for listitem in seq_ls:
			filehandle.write('%s\n' % listitem)

def tokenize_family_data(model):
	"""
	Takes protein family data as a .tab file with "Sequence" and "Protein families" as two of the columns
	Output: A .csv file with "Tokenized Sequence" and "Protein families" for proteins with only one associated family
	"""
	data = pd.read_csv(path,sep='\t')
	print(len(data.columns))
	seq_ls = [seq for seq in data['Sequence']]
	print(seq_ls[110])
	toked = []
	for seq in seq_ls:
		to = model.encode_as_pieces(seq)
		toked.append(" ".join(to))
	print(toked[110])
	data.insert(loc=8, column = 'Tokenized Sequence', value=toked)
	data=data.drop(columns=['Sequence'])
	print(data[:3])
	single_fam = data['Protein families'].str.contains(",").fillna(True)
	sfam=data[~single_fam]
	sfam = sfam[['Tokenized Sequence', 'Protein families']]
	sfam.to_csv('Finetune_fam_data.csv', index=False)

def tokenize_interact_data(model):
	"""
	input: a JSON file with 'from', 'to' and 'link' for each interaction
	output: a .csv of the tokenized version of the input file
	"""
	with open(int_path) as f:
		i_dat = json.load(f)
	df = pd.DataFrame(i_dat)
	df = df[['from','to','link']]
	dfv= df.values
	out =[]
	for row in dfv:
		out.append([" ".join(model.encode_as_pieces(row[0]))," ".join(model.encode_as_pieces(row[1])),row[2]])
	print(out[101])
	out_df = pd.DataFrame(out, columns=df.columns)
	out_df.to_csv('Finetune_interact_tokenized.csv',index=False)

if __name__ == "__main__":
	filter_seqs()
	spm.SentencePieceTrainer.Train('--input=pretraining_data.txt --model_prefix=m_reviewed --vocab_size=10000 --character_coverage=1.0 --model_type=bpe --max_sentence_length=1024')
	model = spm.SentencePieceProcessor()
	model.load(model_path)
	tokenize_family_data(model)
	tokenize_interact_data(model)