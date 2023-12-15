library(data.table)
library(ggplot2)

get_values <- function(path) {
  train_file <- as.data.table(read.delim2(
    paste0(
      '../algorithms/D-SCRIPT-main/', 
      path, 
      '_train.txt'), 
    skip=42, header=F, sep = ' ', fill=T))
  colnames(train_file) <- c('Time', 'Epoch', 'Training', 'Percent', 'Loss', 'Accuracy', 'MSE')
  train_file[, c('Time', 'Training', 'Percent') := NULL]
  train_file <- train_file[Epoch != 'Saving' & !startsWith(Epoch, 'Recall')]
  train_file[, Split := ifelse(Epoch == 'Finished', 'Validation', 'Training')]
  
  retain_indices <- c(1:20)
  epoch <- 1
  for (i in 2:nrow(train_file)) {
    if (train_file$Split[i] == "Validation") {
      train_file$Epoch[i] <- train_file$Epoch[i-1]
      retain_indices[epoch] <- i-1
      retain_indices[epoch+1] <- i
      epoch <- epoch + 2
    }
  }
  
  train_file <- train_file[retain_indices]
  train_file[, Epoch := factor(tstrsplit(tstrsplit(Epoch, '/', keep=1)[[1]], '\\[', keep=2)[[1]], levels = c(1:10))]
  train_file[, Loss := as.numeric(tstrsplit(tstrsplit(Loss, 'Loss=', keep=2)[[1]], ',', keep=1)[[1]])]
  train_file[, Accuracy := as.numeric(tstrsplit(tstrsplit(Accuracy, 'Accuracy=', keep=2)[[1]], '%,', keep=1)[[1]])]
  train_file[, MSE := as.numeric(tstrsplit(tstrsplit(MSE, 'MSE=', keep=2)[[1]], ',', keep=1)[[1]])]
  return(train_file)
}

test <- 'original'

all_files_dscript <- lapply(
  as.list(
    paste0(
      'results_dscript/', test, '/', 
      c('huang', 'guo', 'du', 'pan', 'richoux_regular', 'richoux_strict', 'dscript')
      )
    ), get_values)
names(all_files_dscript) <- c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBAL.')
all_files_dscript <- rbindlist(all_files_dscript, idcol="Dataset")
all_files_dscript[, Dataset := factor(Dataset, levels = c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBAL.'))]
all_files_dscript$Model <- 'D-SCRIPT'

all_files_tt <- lapply(
  as.list(
    paste0(
      'results_topsyturvy/', test, '/', 
      c('huang', 'guo', 'du', 'pan', 'richoux_regular', 'richoux_strict', 'dscript')
    )
  ), get_values)
names(all_files_tt) <- c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBAL.')
all_files_tt <- rbindlist(all_files_tt, idcol="Dataset")
all_files_tt[, Dataset := factor(Dataset, levels = c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBAL.'))]
all_files_tt$Model <- 'Topsy-Turvy'

all_files <- rbind(all_files_dscript, all_files_tt)

ggplot(all_files, aes(x = Epoch, y = Loss, color = Split, group=Split))+
  geom_line(size=1)+
  facet_grid(Model~Dataset, scales = 'free')+
  theme_bw()+
  theme(text = element_text(size = 20))
ggsave(paste0('plots/losses_dscript_tt_', test, '.pdf'), height = 4, width = 19)

ggplot(all_files, aes(x = Epoch, y = MSE, color = Split, group=Split))+
  geom_line(size=1)+
  facet_grid(Model~Dataset, scales = 'free')+
  theme_bw()+
  theme(text = element_text(size = 20))
ggsave(paste0('plots/mses_dscript_tt_', test, '.pdf'), height = 4, width = 19)

ggplot(all_files, aes(x = Epoch, y = Accuracy, color = Split, group=Split))+
  geom_line(size=1)+
  facet_grid(Model~Dataset, scales = 'free')+
  theme_bw()+
  theme(text = element_text(size = 20))
ggsave(paste0('plots/accuracies_dscript_tt_', test, '.pdf'), height = 4, width = 19)


get_values_partition <- function(path, model) {
  train_file <- as.data.table(read.delim2(
    paste0(
      '../algorithms/D-SCRIPT-main/results_', model, '/partitions/train_', path, 
      '.txt'), 
    skip=42, header=F, sep = ' ', fill=T))
  colnames(train_file) <- c('Time', 'Epoch', 'Training', 'Percent', 'Loss', 'Accuracy', 'MSE')
  train_file[, c('Time', 'Training', 'Percent') := NULL]
  train_file <- train_file[Epoch != 'Saving' & !startsWith(Epoch, 'Recall')]
  train_file[, Split := ifelse(Epoch == 'Finished', 'Validation', 'Training')]
  
  retain_indices <- c(1:20)
  epoch <- 1
  for (i in 2:nrow(train_file)) {
    if (train_file$Split[i] == "Validation") {
      train_file$Epoch[i] <- train_file$Epoch[i-1]
      retain_indices[epoch] <- i-1
      retain_indices[epoch+1] <- i
      epoch <- epoch + 2
    }
  }
  
  train_file <- train_file[retain_indices]
  train_file[, Epoch := factor(tstrsplit(tstrsplit(Epoch, '/', keep=1)[[1]], '\\[', keep=2)[[1]], levels = c(1:10))]
  train_file[, Loss := as.numeric(tstrsplit(tstrsplit(Loss, 'Loss=', keep=2)[[1]], ',', keep=1)[[1]])]
  train_file[, Accuracy := as.numeric(tstrsplit(tstrsplit(Accuracy, 'Accuracy=', keep=2)[[1]], '%,', keep=1)[[1]])]
  train_file[, MSE := as.numeric(tstrsplit(tstrsplit(MSE, 'MSE=', keep=2)[[1]], ',', keep=1)[[1]])]
  return(train_file)
}


all_files_dscript <- lapply(
  as.list(c(paste0(c('huang', 'guo', 'du', 'pan', 'richoux', 'dscript'), '_both_0'), 
       paste0(c('huang', 'guo', 'du', 'pan', 'richoux', 'dscript'), '_both_1'),
       paste0(c('huang', 'guo', 'du', 'pan', 'richoux', 'dscript'), '_0_1'))), 
  get_values_partition, model='dscript')
names(all_files_dscript) <- c(paste0(c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'), '_INTER->INTRA0'), 
                              paste0(c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'), '_INTER->INTRA1'), 
                              paste0(c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'), '_INTRA0->INTRA1'))
all_files_dscript <- rbindlist(all_files_dscript, idcol="Dataset")
all_files_dscript$Model <- 'D-SCRIPT'

all_files_tt <- lapply(
  as.list(c(paste0(c('huang', 'guo', 'du', 'pan', 'richoux', 'dscript'), '_both_0'), 
            paste0(c('huang', 'guo', 'du', 'pan', 'richoux', 'dscript'), '_both_1'),
            paste0(c('huang', 'guo', 'du', 'pan', 'richoux', 'dscript'), '_0_1'))), 
  get_values_partition, model='topsyturvy')
names(all_files_tt) <- c(paste0(c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'), '_INTER->INTRA0'), 
                              paste0(c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'), '_INTER->INTRA1'), 
                              paste0(c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'), '_INTRA0->INTRA1'))
all_files_tt <- rbindlist(all_files_tt, idcol="Dataset")
all_files_tt$Model <- 'TOPSY-TURVY'

all_files <- rbind(all_files_dscript, all_files_tt)
all_files[, c('Dataset', 'Partition') := tstrsplit(Dataset, '_')]
all_files[, Partition := paste(Split, Partition, sep='\n')]
all_files[, Dataset := factor(Dataset, levels = c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX', 'D-SCRIPT UNBAL.'))]


ggplot(all_files, aes(x = Epoch, y = Loss, color = Partition, group=Partition))+
  geom_line(size=1)+
  facet_grid(Model~Dataset, scales = 'free')+
  scale_color_manual(values = c('#F0E442', '#E69F00','#D55E00', '#009E73', '#0072B2', '#CC79A7'))+
  theme_bw()+
  theme(text = element_text(size = 20))
ggsave(paste0('plots/losses_dscript_tt_partitions.pdf'), height = 4, width = 19)

ggplot(all_files, aes(x = Epoch, y = MSE, color = Partition, group=Partition))+
  geom_line(size=1)+
  facet_grid(Model~Dataset, scales = 'free')+
  scale_color_manual(values = c('#F0E442', '#E69F00','#D55E00', '#009E73', '#0072B2', '#CC79A7'))+
  theme_bw()+
  theme(text = element_text(size = 20))
ggsave(paste0('plots/mses_dscript_tt_partitions.pdf'), height = 4, width = 19)

ggplot(all_files, aes(x = Epoch, y = Accuracy, color = Partition, group=Partition))+
  geom_line(size=1)+
  facet_grid(Model~Dataset, scales = 'free')+
  scale_color_manual(values = c('#F0E442', '#E69F00','#D55E00', '#009E73', '#0072B2', '#CC79A7'))+
  theme_bw()+
  theme(text = element_text(size = 20))
ggsave(paste0('plots/accuracies_dscript_tt_partitions.pdf'), height = 4, width = 19)
