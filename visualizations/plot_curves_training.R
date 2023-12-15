library(data.table)
library(ggplot2)

test <- 'original'

es_epochs_orig <- data.table(Model = c(rep('D-SCRIPT', 8), rep('Topsy-Turvy', 8), rep('DeepFE', 8), rep(c('Richoux_FC'), 8), rep('Richoux_LSTM', 8), rep('PIPR', 8)), 
                        Dataset = c(rep(c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBAL.', 'GOLD'), 6)), 
                        Epoch = c(5, 10, 9, 9, 1, 1, 7, 2,
                                  7, 7, 3, 2, 1, 5, 4, 10,
                                  5, 2, 1, 1, 1, 1, 15, 1,
                                  1, 1, 1, 16, 10, 1, 1, 2,
                                  1, 1, 1, 1, 17, 1, 1, 6,
                                  28, 1, 9, 3, 13, 6, 10, 3))

es_epochs_rew <-  data.table(Model = c(rep('D-SCRIPT', 7), rep('Topsy-Turvy', 7), rep('DeepFE', 7), rep(c('Richoux_FC'), 7), rep('Richoux_LSTM', 7), rep('PIPR', 7)), 
                             Dataset = c(rep(c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBAL.'), 6)), 
                             Epoch = c(8, 9, 9, 6, 1, 1, 2,
                                       4, 9, 10, 7, 1, 1, 3,
                                       12, 1, 1, 7, 1, 1, 16,
                                       1, 1, 1, 1, 11, 1, 1,
                                       1, 1, 1, 24, 16, 19, 1,
                                       1, 1, 4, 8, 4, 7, 1))

es_epochs_part <-  data.table(Model = c(rep('D-SCRIPT', 18), rep('Topsy-Turvy', 18), rep('DeepFE', 18), rep(c('Richoux_FC'), 18), rep('Richoux_LSTM', 18), rep('PIPR', 18)), 
                             Dataset = c(rep(c(rep('HUANG', 3), 
                                               rep('GUO', 3), 
                                               rep('DU', 3), 
                                               rep('PAN', 3), 
                                               rep('RICHOUX', 3), 
                                               rep('D-SCRIPT UNBAL.', 3) ), 6)), 
                             Setting = c(rep(c('INTER->INTRA0', 'INTER->INTRA1', 'INTRA0->INTRA1'), 36)),
                             Epoch = c(8, 9, 8, 6, 3, 2, 10, 10, 5, 8, 2, 1, 6, 1, 3, 10, 4, 1, 
                                       10, 1, 9, 8, 3, 3, 1, 4, 5, 7, 9, 1, 9, 2, 10, 5, 7, 1,
                                       1, 2, 3, 2, 3, 3, 1, 1, 2, 2, 2, 1, 2, 1, 1, 7, 9, 11,
                                       1, 1, 3, 1, 25, 1, 1, 3, 1, 1, 25, 25, 7, 5, 4, 1, 2, 2,            
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 13, 1, 1, 1, 
                                       6, 1, 14, 12, 15, 6, 6, 16, 10, 22, 9, 7, 12, 8, 6, 1, 2, 1))

es_epochs <- merge(es_epochs_orig, es_epochs_rew, by = c('Model', 'Dataset'))
es_epochs_part[, Partition := paste0(Setting, ': ', Epoch)]
es_epochs_part <- es_epochs_part[,  c(1,2,5)]
es_epochs_part <- es_epochs_part[, list(Partition = paste(Partition, collapse='; ')), by = c('Model', 'Dataset')]
es_epochs <- merge(es_epochs, es_epochs_part, by = c('Model', 'Dataset'), all.x = T, all.y = T)
es_epochs[, Model := factor(Model, levels = c('DeepFE', 'PIPR', 'Richoux_FC', 'Richoux_LSTM', 'D-SCRIPT', 'Topsy-Turvy'))]
colnames(es_epochs) <- c('Model', 'Dataset', 'Original', 'Rewired', 'Partition')


if(test == 'original'){
  es_epochs <- es_epochs_orig
}else if(test == 'rewired'){
  es_epochs <- es_epochs_rew
}else{
  es_epochs <- es_epochs_part
  es_epochs[, c('from', 'to') := tstrsplit(Setting, '_')]
  es_epochs[from == 'both', from := 'INTER']
  es_epochs[from == '0', from := 'INTRA0']
  es_epochs[to == '0', to := 'INTRA0']
  es_epochs[to == '1', to := 'INTRA1']
  es_epochs[, concat := paste0(from, '->', to)]
  es_epochs[, Dataset := paste(Dataset, concat)]
  es_epochs[, Dataset := as.factor(Dataset)]
}


#### other_dfs
es_results <- fread('../early_stopping_df.csv')
es_results[Setting == 'gold', Setting := 'original']
es_results <- es_results[Setting == test]
es_results <- es_results[, -c('Setting')]
es_results[, Dataset := toupper(Dataset)]
es_results[Dataset == 'RICHOUX_REGULAR', Dataset := 'RICHOUX-REGULAR']
es_results[Dataset == 'RICHOUX_STRICT', Dataset := 'RICHOUX-STRICT']

##### DSCRIPT 
get_values <- function(path, partitions = FALSE) {
  train_file <- as.data.table(read.delim2(
    path,
    skip = 42,
    header = F,
    sep = ' ',
    fill = T
  ))
  colnames(train_file) <-
    c('Time',
      'Epoch',
      'Training',
      'Percent',
      'Loss',
      'Accuracy',
      'MSE')
  train_file[, c('Time', 'Training', 'Percent') := NULL]
  train_file <-
    train_file[Epoch != 'Saving' & !startsWith(Epoch, 'Recall')]
  train_file[, Split := ifelse(Epoch == 'Finished', 'Validation', 'Training')]
  
  retain_indices <- c(1:20)
  epoch <- 1
  for (i in 2:nrow(train_file)) {
    if (train_file$Split[i] == "Validation") {
      train_file$Epoch[i] <- train_file$Epoch[i - 1]
      retain_indices[epoch] <- i - 1
      retain_indices[epoch + 1] <- i
      epoch <- epoch + 2
    }
  }
  
  train_file <- train_file[retain_indices]
  train_file[, Epoch := factor(tstrsplit(tstrsplit(Epoch, '/', keep = 1)[[1]], '\\[', keep =
                                           2)[[1]], levels = c(1:10))]
  train_file[, Loss := as.numeric(tstrsplit(tstrsplit(Loss, 'Loss=', keep =
                                                        2)[[1]], ',', keep = 1)[[1]])]
  train_file[, Accuracy := as.numeric(tstrsplit(tstrsplit(Accuracy, 'Accuracy=', keep =
                                                            2)[[1]], '%,', keep = 1)[[1]])/100]
  train_file[, MSE := as.numeric(tstrsplit(tstrsplit(MSE, 'MSE=', keep =
                                                       2)[[1]], ',', keep = 1)[[1]])]
  return(train_file)
}


if (test == 'original') {
  datasets <-
    c('huang',
      'guo',
      'du',
      'pan',
      'richoux_regular',
      'richoux_strict',
      'dscript',
      'gold')
      dataset_names <-
        c(
          'HUANG',
          'GUO',
          'DU',
          'PAN',
          'RICHOUX-REGULAR',
          'RICHOUX-STRICT',
          'D-SCRIPT UNBAL.',
          'GOLD'
        )
} else if (test == 'rewired') {
  datasets <-
    c('huang',
      'guo',
      'du',
      'pan',
      'richoux_regular',
      'richoux_strict',
      'dscript')
  dataset_names <-
    c('HUANG',
      'GUO',
      'DU',
      'PAN',
      'RICHOUX-REGULAR',
      'RICHOUX-STRICT',
      'D-SCRIPT UNBAL.')
} else{
  datasets <- c(
    'huang_both_0',
    'huang_both_1',
    'huang_0_1',
    'guo_both_0',
    'guo_both_1',
    'guo_0_1',
    'du_both_0',
    'du_both_1',
    'du_0_1',
    'pan_both_0',
    'pan_both_1',
    'pan_0_1',
    'richoux_both_0',
    'richoux_both_1',
    'richoux_0_1',
    'dscript_both_0',
    'dscript_both_1',
    'dscript_0_1'
  )
  dataset_names <-
    c(
      'HUANG INTER->INTRA0',
      'HUANG INTER->INTRA1',
      'HUANG INTRA0->INTRA1',
      'GUO INTER->INTRA0',
      'GUO INTER->INTRA1',
      'GUO INTRA0->INTRA1',
      'DU INTER->INTRA0',
      'DU INTER->INTRA1',
      'DU INTRA0->INTRA1',
      'PAN INTER->INTRA0',
      'PAN INTER->INTRA1',
      'PAN INTRA0->INTRA1',
      'RICHOUX INTER->INTRA0',
      'RICHOUX INTER->INTRA1',
      'RICHOUX INTRA0->INTRA1',
      'D-SCRIPT UNBAL. INTER->INTRA0',
      'D-SCRIPT UNBAL. INTER->INTRA1',
      'D-SCRIPT UNBAL. INTRA0->INTRA1'
    )
}

if (test == 'partition') {
  path_list <- as.list(
    paste0(
      '../algorithms/D-SCRIPT-main/results_dscript/partitions/train_', datasets, '.txt'
    )
  )
} else{
  path_list <- as.list(
    paste0(
      '../algorithms/D-SCRIPT-main/results_dscript/',
      test,
      '/',
      datasets,
      '_train.txt'
    )
  )
}
all_files_dscript <- lapply(path_list, get_values)
names(all_files_dscript) <- dataset_names
all_files_dscript <- rbindlist(all_files_dscript, idcol = "Dataset")
all_files_dscript[, Dataset := factor(Dataset, levels = dataset_names)]
all_files_dscript$Model <- 'D-SCRIPT'

if (test == 'partition') {
  path_list <- as.list(
    paste0(
      '../algorithms/D-SCRIPT-main/results_topsyturvy/partitions/train_', datasets, '.txt'
    )
  )
} else{
  path_list <- as.list(
    paste0(
      '../algorithms/D-SCRIPT-main/results_topsyturvy/',
      test,
      '/',
      datasets,
      '_train.txt'
    )
  )
}
all_files_tt <- lapply(path_list, get_values)
names(all_files_tt) <- dataset_names
all_files_tt <- rbindlist(all_files_tt, idcol = "Dataset")
all_files_tt[, Dataset := factor(Dataset, levels = dataset_names)]
all_files_tt$Model <- 'Topsy-Turvy'

all_files <- rbind(all_files_dscript, all_files_tt)
best_epochs <-
  all_files[all_files[Split == 'Validation', .I[which.max(Accuracy)], by = .(Model, Dataset)]$V1]
if(test=='partition'){
  best_epochs[, partition := tstrsplit(Dataset, '(HUANG|GUO|DU|PAN|RICHOUX|D-SCRIPT UNBAL.)', keep = 2)]
  best_epochs <- best_epochs[order(Model, partition, Dataset)]
}
#fwrite(best_epochs,
#       paste0('../algorithms/D-SCRIPT-main/best_epochs_', test, '.csv'))


##### Visualization
all_files <- all_files[, -c('MSE')]
all_files <- rbind(all_files, es_results)
all_files[, Model := factor(Model, levels = c('DeepFE', 'PIPR', 'Richoux_FC', 'Richoux_LSTM', 'D-SCRIPT', 'Topsy-Turvy'))]

custom_height <- ifelse(test == 'partition', 30, 16)
custom_width <- ifelse(test == 'partition', 30, 19)
custom_nrows <- ifelse(test == 'partition', 18, 8)

g <- ggplot(all_files, aes(
  x = Epoch,
  y = Loss,
  color = Split,
  group = Split
)) +
  geom_line(size = 1) +
  geom_vline(data=es_epochs, aes(xintercept = Epoch), color='red')+
  scale_x_discrete(guide = guide_axis(check.overlap = TRUE))+
  facet_wrap(Dataset ~ Model, scales = 'free', nrow = custom_nrows) +
  theme_bw() +
  theme(text = element_text(size = 20))

ggsave(
  paste0('plots/losses_all_', test, '.pdf'),
  plot=g,
  height = custom_height,
  width = custom_width
)

g <- ggplot(all_files, aes(
  x = Epoch,
  y = Accuracy,
  color = Split,
  group = Split
)) +
  geom_line(size = 1) +
  geom_vline(data=es_epochs, aes(xintercept = Epoch), color='red')+
  scale_x_discrete(guide = guide_axis(check.overlap = TRUE))+
  facet_wrap(Dataset ~ Model, scales = 'free', nrow=custom_nrows) +
  theme_bw() +
  theme(text = element_text(size = 20))
ggsave(
  paste0('plots/accuracies_all_', test, '.pdf'),
  plot=g, 
  height = custom_height,
  width = custom_width
)


get_values_partition <- function(path, model) {
  train_file <- as.data.table(read.delim2(
    paste0(
      '../algorithms/D-SCRIPT-main/results_',
      model,
      '/partitions/train_',
      path,
      '.txt'
    ),
    skip = 42,
    header = F,
    sep = ' ',
    fill = T
  ))
  colnames(train_file) <-
    c('Time',
      'Epoch',
      'Training',
      'Percent',
      'Loss',
      'Accuracy',
      'MSE')
  train_file[, c('Time', 'Training', 'Percent') := NULL]
  train_file <-
    train_file[Epoch != 'Saving' & !startsWith(Epoch, 'Recall')]
  train_file[, Split := ifelse(Epoch == 'Finished', 'Validation', 'Training')]
  
  retain_indices <- c(1:20)
  epoch <- 1
  for (i in 2:nrow(train_file)) {
    if (train_file$Split[i] == "Validation") {
      train_file$Epoch[i] <- train_file$Epoch[i - 1]
      retain_indices[epoch] <- i - 1
      retain_indices[epoch + 1] <- i
      epoch <- epoch + 2
    }
  }
  
  train_file <- train_file[retain_indices]
  train_file[, Epoch := factor(tstrsplit(tstrsplit(Epoch, '/', keep = 1)[[1]], '\\[', keep =
                                           2)[[1]], levels = c(1:10))]
  train_file[, Loss := as.numeric(tstrsplit(tstrsplit(Loss, 'Loss=', keep =
                                                        2)[[1]], ',', keep = 1)[[1]])]
  train_file[, Accuracy := as.numeric(tstrsplit(tstrsplit(Accuracy, 'Accuracy=', keep =
                                                            2)[[1]], '%,', keep = 1)[[1]])]
  train_file[, MSE := as.numeric(tstrsplit(tstrsplit(MSE, 'MSE=', keep =
                                                       2)[[1]], ',', keep = 1)[[1]])]
  return(train_file)
}


all_files_dscript <- lapply(as.list(c(
  paste0(c(
    'huang', 'guo', 'du', 'pan', 'richoux', 'dscript'
  ), '_both_0'),
  paste0(c(
    'huang', 'guo', 'du', 'pan', 'richoux', 'dscript'
  ), '_both_1'),
  paste0(c(
    'huang', 'guo', 'du', 'pan', 'richoux', 'dscript'
  ), '_0_1')
)),
get_values_partition, model = 'dscript')
names(all_files_dscript) <-
  c(
    paste0(
      c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'),
      '_INTER->INTRA0'
    ),
    paste0(
      c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'),
      '_INTER->INTRA1'
    ),
    paste0(
      c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'),
      '_INTRA0->INTRA1'
    )
  )
all_files_dscript <- rbindlist(all_files_dscript, idcol = "Dataset")
all_files_dscript$Model <- 'D-SCRIPT'

all_files_tt <- lapply(as.list(c(
  paste0(c(
    'huang', 'guo', 'du', 'pan', 'richoux', 'dscript'
  ), '_both_0'),
  paste0(c(
    'huang', 'guo', 'du', 'pan', 'richoux', 'dscript'
  ), '_both_1'),
  paste0(c(
    'huang', 'guo', 'du', 'pan', 'richoux', 'dscript'
  ), '_0_1')
)),
get_values_partition, model = 'topsyturvy')
names(all_files_tt) <-
  c(
    paste0(
      c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'),
      '_INTER->INTRA0'
    ),
    paste0(
      c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'),
      '_INTER->INTRA1'
    ),
    paste0(
      c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'),
      '_INTRA0->INTRA1'
    )
  )
all_files_tt <- rbindlist(all_files_tt, idcol = "Dataset")
all_files_tt$Model <- 'TOPSY-TURVY'

all_files <- rbind(all_files_dscript, all_files_tt)
all_files[, c('Dataset', 'Partition') := tstrsplit(Dataset, '_')]
all_files[, Partition := paste(Split, Partition, sep = '\n')]
all_files[, Dataset := factor(Dataset,
                              levels = c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'))]

best_epochs <-
  all_files[all_files[Split == 'Validation', .I[which.max(Accuracy)], by = .(Model, Dataset, Partition)]$V1]
#fwrite(best_epochs,
#       '../algorithms/D-SCRIPT-main/best_epochs_partition.csv')

ggplot(all_files,
       aes(
         x = Epoch,
         y = Loss,
         color = Partition,
         group = Partition
       )) +
  geom_line(size = 1) +
  facet_grid(Model ~ Dataset, scales = 'free') +
  scale_color_manual(values = c(
    '#F0E442',
             '#E69F00',
             '#D55E00',
             '#009E73',
             '#0072B2',
             '#CC79A7'
  )) +
  theme_bw() +
  theme(text = element_text(size = 20))
#ggsave(
#  paste0('plots/losses_dscript_tt_partitions.pdf'),
#  height = 4,
#  width = 19
#)

ggplot(all_files,
       aes(
         x = Epoch,
         y = Accuracy,
         color = Partition,
         group = Partition
       )) +
  geom_line(size = 1) +
  facet_grid(Model ~ Dataset, scales = 'free') +
  scale_color_manual(values = c(
    '#F0E442',
             '#E69F00',
             '#D55E00',
             '#009E73',
             '#0072B2',
             '#CC79A7'
  )) +
  theme_bw() +
  theme(text = element_text(size = 20))
#ggsave(
#  paste0('plots/accuracies_dscript_tt_partitions.pdf'),
#  height = 4,
#  width = 19
#)


all_results_dscript_gold <-
  fread('../algorithms/D-SCRIPT-main/results_dscript/original/all_results_gold.tsv')
all_results_dscript_gold$Model <- 'D-SCRIPT'
all_results_tt_gold <-
  fread('../algorithms/D-SCRIPT-main/results_topsyturvy/original/all_results_gold.tsv')
all_results_tt_gold$Model <- 'Topsy-Turvy'
all_results_gold <-
  rbind(all_results_dscript_gold, all_results_tt_gold)
all_results_gold[, Epoch := as.numeric(tstrsplit(Dataset, '_', keep = 2)[[1]])]

ggplot(all_results_gold[!Metric %in% c("TP", "FP", "TN", "FN", "Sensitivity")], aes(x = Epoch, y = Value, color = Metric)) +
  geom_line(size = 1) +
  geom_text(aes(label = round(Value, 3)), nudge_y = 0.1) +
  facet_grid(Model ~ Metric) +
  theme_bw() +
  theme(text = element_text(size = 20))
#ggsave('~/Downloads/test_metrics_all_epochs.pdf',
#       width = 35,
#       height = 4)


## robustness tests
# path_list <- as.list(paste0('../algorithms/D-SCRIPT-main/robustness_tests/robustness_', 
#                             rep(c('huang', 'guo', 'du', 'pan', 'richoux_regular'), 3), 
#                             '_dscript_original_train_', c(rep(0, 5), rep(1, 5), rep(2, 5)), '.txt'))
# all_files_dscript <- lapply(path_list, get_values)
# dataset_names <- paste0(rep(c('huang', 'guo', 'du', 'pan', 'richoux-regular'), 3), '_', c(rep(0, 5), rep(1, 5), rep(2, 5)))
# names(all_files_dscript) <- dataset_names
# all_files_dscript <- rbindlist(all_files_dscript, idcol = "Dataset")
# all_files_dscript[, Dataset := factor(Dataset, levels = dataset_names)]
# all_files_dscript[, c("ds", "iter") := tstrsplit(Dataset, '_')]
# ggplot(all_files_dscript, aes(x = Epoch, y = Loss, color = iter, group = Dataset))+
#   geom_line(size=1)+
#   facet_grid(ds~Split)+
#   theme_bw()
# 
# ggplot(all_files_dscript, aes(x = Epoch, y = Accuracy, color = iter, group = Dataset))+
#   geom_line(size=1)+
#   facet_grid(ds~Split)+
#   theme_bw()

