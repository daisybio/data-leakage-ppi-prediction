import pandas as pd


def read_train_file(path, skipon_var, skipon_val, model):
    setting = ''
    dataset = ''
    epoch = ''
    results = []

    with open(path, 'r') as file:
        for line in file:
            str_line = line.strip()
            if eval(skipon_var) == skipon_val:
                break
            if any(s in str_line for s in ['original', 'rewired', 'partition', 'gold']):
                setting = [s for s in ['original', 'rewired', 'partition', 'gold'] if s in str_line][0]
            if any(ds in str_line for ds in
                   ['guo_', 'huang_', 'du_', 'pan_', 'richoux_regular_', 'richoux_strict_', 'richoux_', 'dscript_', 'gold_']):
                dataset = [ds for ds in ['guo', 'huang', 'du', 'pan', 'richoux_regular', 'richoux_strict', 'richoux', 'dscript', 'gold'] if
                           ds in str_line][0]
                if dataset == 'dscript':
                    dataset = 'D-SCRIPT UNBAL.'
                if 'both_' in str_line or '0_' in str_line:
                    if 'both_' in str_line:
                        dataset += ' INTER->'
                    else:
                        dataset += ' INTRA0->'
                    if '0_es' in str_line:
                        dataset += 'INTRA0'
                    else:
                        dataset += 'INTRA1'
            if 'FC_' in str_line:
                model = 'Richoux_FC'
            elif 'LSTM_' in str_line:
                model = 'Richoux_LSTM'
            if str_line.startswith('Epoch'):
                epoch = str_line.split(' ')[1].split(':')[0]
            if line.strip().__contains__('val_loss'):
                spl_line = str_line.split(' ')
                train_loss = spl_line[7]
                train_acc = spl_line[10]
                val_loss = spl_line[13]
                val_acc = spl_line[16]
                results.append(
                    {'Setting': setting, 'Dataset': dataset, 'Model': model, 'Epoch': epoch, 'Loss': train_loss,
                     'Accuracy': train_acc, 'Split': 'Training'})
                results.append(
                    {'Setting': setting, 'Dataset': dataset, 'Model': model, 'Epoch': epoch, 'Loss': val_loss,
                     'Accuracy': val_acc, 'Split': 'Validation'})
    df = pd.DataFrame(results)
    return df

df_richoux_orig = read_train_file('algorithms/DeepPPI/keras/es_deepPPI.out', skipon_var='str_line', skipon_val='rewired', model='')
df_richoux_rest = read_train_file('algorithms/DeepPPI/keras/es_gold_deepPPI.out', skipon_var='epoch', skipon_val='-1', model='')
df_deepFE = read_train_file('algorithms/DeepFE-PPI/deepFE_es.out', skipon_var='epoch', skipon_val='-1', model='DeepFE')
df_deepFE_gold = read_train_file('algorithms/DeepFE-PPI/deepFE_es_gold.out', skipon_var='epoch', skipon_val='-1', model='DeepFE')
df_pipr = read_train_file('algorithms/seq_ppi/binary/model/lasagna/es_PIPR.out',  skipon_var='epoch', skipon_val='-1', model='PIPR')
df_pipr_rest = read_train_file('algorithms/seq_ppi/binary/model/lasagna/es_gold_PIPR.out',  skipon_var='epoch', skipon_val='-1', model='PIPR')
df = pd.concat([df_richoux_orig, df_richoux_rest, df_deepFE, df_deepFE_gold, df_pipr, df_pipr_rest])

df.to_csv('early_stopping_df.csv', index=False)





