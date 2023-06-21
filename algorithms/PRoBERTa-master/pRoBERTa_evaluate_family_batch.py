from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens
import sys
import pandas as pd
import numpy as np
import torch

if len(sys.argv) != 7:
    print(sys.argv[0] + " data_path binarized_data_path output_path model_folder classification_head batch_size")
    sys.exit()

# TSV file with columns: Tokenized Sequence, True Family Label
data_path=sys.argv[1]
# Path to binarized data created by fairseq-preprocess
binarized_path=sys.argv[2]
# Path to an output TSV file
output_path=sys.argv[3]
# Folder containing model checkpoints
model_folder=sys.argv[4]
classification_head=sys.argv[5]
batch_size=int(sys.argv[6])

sequence_col=0
label_col=1
preds_col=2

has_cuda = torch.cuda.device_count() > 0

data=pd.read_csv(data_path, header = None)
# Label dictionary replaces spaces with underscores
data[label_col] = data[label_col].str.replace(" ", "_")

model = RobertaModel.from_pretrained(model_folder, "checkpoint_best.pt", binarized_path, bpe=None)
model.eval()

if (has_cuda):
    model.cuda()

split_num = int(len(data) / batch_size)
batched_data=np.array_split(data, split_num)
print("Total batches: " + str(len(batched_data)))

with torch.no_grad():
    preds_df = pd.DataFrame(columns=[sequence_col, label_col, preds_col])
    for count, batch_df in enumerate(batched_data):
 
        batch=collate_tokens([torch.cat((model.encode(tokens), torch.ones(512, dtype = torch.long)))[:512]
            for tokens in batch_df[sequence_col]], pad_idx=1)

        logprobs = model.predict(classification_head, batch)
        preds = model.task.label_dictionary.string(logprobs.argmax(dim=1) + model.task.label_dictionary.nspecial)
        batch_df[preds_col] = preds.split()
        preds_df = preds_df.append(batch_df, ignore_index=True)
        
        if (has_cuda):
            torch.cuda.empty_cache()
        
        if count % 5 == 0:
            print("Batch " + str(count + 1) + " completed.")

preds_df = preds_df.rename(columns=
        {sequence_col : "Tokenized Sequence", label_col : "Family", preds_col : "Predicted"})
preds_df.to_csv(output_path, sep="\t", index = False)

n_correct = np.where(preds_df["Family"]==preds_df["Predicted"], 1, 0).sum()
print("Accuracy: " + str(n_correct / len(preds_df)))
