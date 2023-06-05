library(data.table)
library(ggplot2)
library(RColorBrewer)

#### result prefixes
custom_res <- '../algorithms/Custom/results/'
deepFE_res <- '../algorithms/DeepFE-PPI/result/custom/'
deepPPI_res <- '../algorithms/DeepPPI/keras/results_custom/'
seqppi_res <- '../algorithms/seq_ppi/binary/model/lasagna/results/'
sprint_res <- '../algorithms/SPRINT/results/'
dscript_res <- '../algorithms/D-SCRIPT-main/results_dscript/'
tt_res <- '../algorithms/D-SCRIPT-main/results_topsyturvy/'

#### time files
all_times <- data.table(Test = character(), Model=character(), Dataset=character(), `Time [s]`=numeric())
# custom
custom_times <- lapply(paste0(custom_res, list.files(custom_res, pattern='time*')), fread)
file_names <- tstrsplit(list.files(custom_res, pattern='time*'), '.txt', keep=1)[[1]]
file_names[grepl('richoux_regular', file_names, fixed=TRUE)] <- gsub('richoux_regular', 'Richoux-Regular', file_names[grepl('richoux_regular', file_names, fixed=TRUE)])
file_names[grepl('richoux_strict', file_names, fixed=TRUE)] <- gsub('richoux_strict', 'Richoux-Strict', file_names[grepl('richoux_strict', file_names, fixed=TRUE)])
file_names <- tstrsplit(file_names, '_', keep=c(2, 3, 4))
names(custom_times) <- paste(file_names[[1]], file_names[[2]], file_names[[3]], sep='_')
custom_times_df <- rbindlist(custom_times[!grepl('partition|deg', names(custom_times))], idcol = 'filename')
partition_times_df <- rbindlist(custom_times[grepl('partition_(dscript|du|guo|huang|pan|richoux)', names(custom_times))], idcol = 'filename')
custom_times_df <- custom_times_df[, c('test', 'dataset', 'encoding') := tstrsplit(filename, '_')]
partition_times_df <- partition_times_df[, c('test', 'dataset', 'encoding') := tstrsplit(filename, '_')]
names(custom_times_df) <- c('Filename', 'Algorithm', 'Time [s]', 'Test', 'Dataset', 'Encoding')
names(partition_times_df) <- c('Filename', 'Part_Train', 'Part_Test', 'Algorithm', 'Time [s]', 'Test', 'Dataset', 'Encoding')
custom_times_df$Part_Train <- NA
custom_times_df$Part_Test <- NA
custom_times_df[, Model := paste(Algorithm, Encoding, sep='_')]
partition_times_df[, Model := paste(Algorithm, Encoding, Part_Train, Part_Test, sep='_')]
all_times_custom <- rbind(custom_times_df, partition_times_df)

deg_times_df <- rbindlist(custom_times[grepl('deg', names(custom_times))], idcol = 'filename')
colnames(deg_times_df) <- c('Filename', 'Dataset', 'Algorithm', 'Time [s]')
deg_times_df[, Test := tstrsplit(Filename, '_', keep=1)]
deg_times_df$Encoding <- NA
deg_times_df[Dataset == 'gold_standard', Dataset := 'gold']
deg_times_df[Dataset == 'richoux_regular', Dataset := 'Richoux-Regular']
deg_times_df[Dataset == 'richoux_strict', Dataset := 'Richoux-Strict']
deg_times_df[, c('Dataset','Part_Train', 'Part_Test') := tstrsplit(Dataset, '_')]
deg_times_df[, Model := Algorithm]
all_times_custom <- rbind(all_times_custom, deg_times_df)
fwrite(all_times_custom, '../algorithms/Custom/results/run_t.csv')

all_times <- rbind(all_times, all_times_custom[, c('Test', 'Model', 'Dataset','Time [s]')])

# deepFE
deepFE_times <- lapply(paste0(deepFE_res, list.files(deepFE_res, pattern = 'time*', recursive = TRUE)), fread)
file_names <- tstrsplit(basename(list.files(deepFE_res, pattern = 'time*', recursive = TRUE)), '.txt', keep=1)[[1]]
file_names[grepl('richoux_regular', file_names, fixed=TRUE)] <- gsub('richoux_regular', 'Richoux-Regular', file_names[grepl('richoux_regular', file_names, fixed=TRUE)])
file_names[grepl('richoux_strict', file_names, fixed=TRUE)] <- gsub('richoux_strict', 'Richoux-Strict', file_names[grepl('richoux_strict', file_names, fixed=TRUE)])
names(deepFE_times) <- file_names 
deepFE_times <- rbindlist(deepFE_times, idcol='Filename')
deepFE_times <- deepFE_times[, c('Test', 'Dataset', 'Part_Train', 'Part_Test') := tstrsplit(Filename, '_', keep=c(2,3,4,5), fill = NA)]
deepFE_times <- deepFE_times[, Model := ifelse(is.na(Part_Train),
                                               'DeepFE',
                                               paste('DeepFE', Part_Train, Part_Test, sep='_'))]
colnames(deepFE_times) <-c('Filename', 'Time [s]', 'Test', 'Dataset', 'Part_Train', 'Part_Test', 'Model')
fwrite(deepFE_times, '../algorithms/DeepFE-PPI/result/custom/run_t.csv')

all_times <- rbind(all_times, deepFE_times[, c('Test', 'Model', 'Dataset', 'Time [s]')])

# deepPPI
deepPPI_times <- fread(paste0(deepPPI_res, 'all_times.txt'))
colnames(deepPPI_times) <- c('Filename', 'Time [s]')
deepPPI_times$Filename[grepl('richoux_regular', deepPPI_times$Filename, fixed=TRUE)] <- gsub('richoux_regular', 'Richoux-Regular', deepPPI_times$Filename[grepl('richoux_regular', deepPPI_times$Filename, fixed=TRUE)])
deepPPI_times$Filename[grepl('richoux_strict', deepPPI_times$Filename, fixed=TRUE)] <- gsub('richoux_strict', 'Richoux-Strict', deepPPI_times$Filename[grepl('richoux_strict', deepPPI_times$Filename, fixed=TRUE)])
deepPPI_times <- deepPPI_times[, c('Model', 'Test', 'Dataset', 'Part_Train', 'Part_Test') := tstrsplit(Filename, '_', fill=NA)]
deepPPI_times <- deepPPI_times[, Part_Train := tstrsplit(Part_Train, 'tr', keep = 2)]
deepPPI_times <- deepPPI_times[, Part_Test := tstrsplit(Part_Test, 'te', keep = 2)]
deepPPI_times <- deepPPI_times[, Model := ifelse(is.na(Part_Train), 
                              paste0('Richoux-', Model),
                              paste('Richoux', Model, Part_Train, Part_Test, sep='-'))]

all_times <- rbind(all_times, deepPPI_times[,  c('Test', 'Model', 'Dataset', 'Time [s]')])

# seq_ppi
seq_ppi_times <-  fread(paste0(seqppi_res, 'all_times.txt'))
colnames(seq_ppi_times) <- c('Filename', 'Time [s]')
seq_ppi_times$Filename[grepl('richoux_regular', seq_ppi_times$Filename, fixed=TRUE)] <- gsub('richoux_regular', 'richoux-regular', seq_ppi_times$Filename[grepl('richoux_regular', seq_ppi_times$Filename, fixed=TRUE)])
seq_ppi_times$Filename[grepl('richoux_strict', seq_ppi_times$Filename, fixed=TRUE)] <- gsub('richoux_strict', 'richoux-strict', seq_ppi_times$Filename[grepl('richoux_strict', seq_ppi_times$Filename, fixed=TRUE)])
seq_ppi_times <- seq_ppi_times[, c('Test', 'Dataset', 'Part_Train', 'Part_Test') := tstrsplit(Filename, '_', fill=NA)]
seq_ppi_times <- seq_ppi_times[, Model := ifelse(is.na(Part_Train), 
                                                 'PIPR',
                                                 paste('PIPR', Part_Train, Part_Test, sep='_'))]

all_times <- rbind(all_times, seq_ppi_times[, c('Test' ,'Model', 'Dataset', 'Time [s]')])

# SPRINT
sprint_times <- lapply(paste0(sprint_res, list.files(sprint_res, pattern = '*time*', recursive = TRUE)), function(x){
  tmp <- fread(x, header=FALSE, sep='\t')
  # Input string
  time_string <- tmp[1,2]
  time_string <- gsub('s', '', time_string)
  time_string <- gsub(',', '.', time_string)
  # Split minutes and seconds
  time_parts <- tstrsplit(time_string, 'm')
  # Convert minutes and seconds to numeric
  minutes <- as.numeric(time_parts[1])
  seconds <- as.numeric(time_parts[2])
  # Calculate total seconds
  total_seconds <- (minutes * 60) + seconds
  return(data.table('Time [s]' = total_seconds))
})
filenames <- list.files(sprint_res, pattern = '*time*', recursive = TRUE)
test <- tstrsplit(filenames, '/', keep=1)[[1]]
filenames[grepl('richoux_regular', filenames, fixed=TRUE)] <- gsub('richoux_regular', 'Richoux-Regular', filenames[grepl('richoux_regular', filenames, fixed=TRUE)])
filenames[grepl('richoux_strict', filenames, fixed=TRUE)] <- gsub('richoux_strict', 'Richoux-Strict', filenames[grepl('richoux_strict', filenames, fixed=TRUE)])
dataset <- tstrsplit(tstrsplit(filenames, '_time', keep=1)[[1]], '/', keep=2)[[1]]
names(sprint_times) <- paste(test, dataset, sep='.')
sprint_df <- rbindlist(sprint_times, idcol='filename')
sprint_df[, c("Test", "Rest") := tstrsplit(filename, '\\.')]
sprint_df <- sprint_df[, Rest := gsub('(train_|test_)', '', Rest)]
sprint_df <- sprint_df[, Rest := gsub('gold_standard', 'gold', Rest)]
sprint_df[, c("Dataset", "Part_Train", "Part_Test") := tstrsplit(Rest, '_')]
sprint_df[, Model := 'SPRINT']

fwrite(sprint_df, '../algorithms/SPRINT/results/run_t.csv')
all_times <- rbind(all_times, sprint_df[,  c('Test', 'Model', 'Dataset', 'Time [s]')])

#D-SCRIPT
extract_dscript_time <- function(directory) {
  time_files <- paste0(directory, list.files(directory, pattern = '*time\\.txt', recursive = TRUE))
  time_files <- grep('train_', time_files, invert = TRUE, value = TRUE)
  times <- lapply(time_files, function(x){
    tmp <- fread(x, header=FALSE, sep='\t')
    # Input string
    time_string <- tmp[1,2]
    time_string <- gsub('s', '', time_string)
    time_string <- gsub(',', '.', time_string)
    # Split minutes and seconds
    time_parts <- tstrsplit(time_string, 'm')
    # Convert minutes and seconds to numeric
    minutes <- as.numeric(time_parts[1])
    seconds <- as.numeric(time_parts[2])
    # Calculate total seconds
    total_seconds <- (minutes * 60) + seconds
    return(data.table('Time [s]' = total_seconds))
  })
  filenames <- list.files(directory, pattern = '*time\\.txt', recursive = TRUE)
  filenames <- grep('train_', filenames, invert = TRUE, value = TRUE)
  test <- tstrsplit(filenames, '/', keep=1)[[1]]
  filenames[grepl('richoux_regular', filenames, fixed=TRUE)] <- gsub('richoux_regular', 'Richoux-Regular', filenames[grepl('richoux_regular', filenames, fixed=TRUE)])
  filenames[grepl('richoux_strict', filenames, fixed=TRUE)] <- gsub('richoux_strict', 'Richoux-Strict', filenames[grepl('richoux_strict', filenames, fixed=TRUE)])
  dataset <- tstrsplit(tstrsplit(filenames, '_time', keep=1)[[1]], '/', keep=2)[[1]]
  names(times) <- paste(test, dataset, sep='.')
  df <- rbindlist(times, idcol='filename')
  df[, c("Test", "Rest") := tstrsplit(filename, '\\.')]
  df <- df[, Rest := gsub('(train_|test_)', '', Rest)]
  df[, c("Dataset", "Part_Train", "Part_Test") := tstrsplit(Rest, '_')]
  return(df)
}

dscript_df <- extract_dscript_time(dscript_res)
dscript_df[, Model := 'D-SCRIPT']
all_times <- rbind(all_times, dscript_df[,  c('Test', 'Model', 'Dataset', 'Time [s]')])

# Topsy-Turvy
tt_df <- extract_dscript_time(tt_res)
tt_df[, Model := 'Topsy-Turvy']
all_times <- rbind(all_times, tt_df[,  c('Test', 'Model', 'Dataset', 'Time [s]')])

all_times$Dataset <- stringr::str_to_title(all_times$Dataset)

all_times <- all_times[, Dataset := factor(Dataset, 
                                           levels = c("Huang", "Guo", "Du", "Pan", "Richoux-Regular", "Richoux-Strict", "Richoux", "Dscript"))]

##### original 
# training data size
sprint_data_dir <- '../algorithms/SPRINT/data/original/'
training_files_pos <- list.files(path=sprint_data_dir, pattern = 'train_pos')
training_files_neg <- list.files(path=sprint_data_dir, pattern = 'train_neg')
train_sizes_pos <- sapply(paste0(sprint_data_dir, training_files_pos), function(x){
  as.integer(system2("wc",
                     args = c("-l",
                              x,
                              " | awk '{print $1}'"),
                     stdout = TRUE))
}
)
train_sizes_neg <- sapply(paste0(sprint_data_dir, training_files_neg), function(x){
  as.integer(system2("wc",
                     args = c("-l",
                              x,
                              " | awk '{print $1}'"),
                     stdout = TRUE))
}
)
train_sizes <- train_sizes_pos + train_sizes_neg
training_files_pos[grepl('richoux', training_files_pos, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', training_files_pos[grepl('richoux', training_files_pos, fixed=TRUE)])
names(train_sizes) <- stringr::str_to_title(tstrsplit(training_files_pos, '_', keep=1)[[1]])
#train_sizes <- prettyNum(train_sizes, big.mark = ',')

all_times_orig <- all_times[Test == 'original']
all_times_orig[, Model := gsub('RF_', 'RF-', Model)]
all_times_orig[, Model := gsub('SVM_', 'SVM-', Model)]
all_times_orig <- all_times_orig[, Model := factor(Model, 
                                         levels=c("RF-PCA","SVM-PCA", "RF-MDS", "SVM-MDS",
                                                  "RF-node2vec",  "SVM-node2vec", "Harmonic Function", 
                                                  "Local and Global Consistency", "SPRINT", 
                                                  "Richoux-FC", "Richoux-LSTM",  
                                                  "DeepFE", "PIPR", "D-SCRIPT", "Topsy-Turvy"))]
all_times_orig[, n_train := train_sizes[as.character(Dataset)]]
all_times_orig <- all_times_orig[!is.na(n_train), ]


# visualization
colors <- c('#e6194b', '#f032e6', '#ffe119', '#4363d8', '#f58231', '#911eb4',
                     '#3cb44b','#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324',
                     '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075')

ggplot(all_times_orig[!is.na(Dataset)], aes(x=n_train, y = `Time [s]`, color = Model, group=Model, shape=Model))+
  geom_point(size=3)+
  geom_line(size=2, alpha=0.7)+
  scale_x_continuous(breaks = unique(all_times_orig$n_train), 
                     labels = paste0(c('D-SCRIPT UNBALANCED (', 'HUANG (', 'GUO (', 'DU (', 'PAN (', 'RICHOUX-REGULAR (', 'RICHOUX-STRICT ('),
                                     unique(all_times_orig$n_train), rep(')', 6)),
                     guide = guide_axis(check.overlap = T),
                     trans = 'log10', 
                     limits = c(min(unique(all_times_orig$n_train)), max(unique(all_times_orig$n_train))))+
  labs(x = "Dataset (n training)", y = "Time [s]") +
  geom_hline(yintercept = 1800, color='red') +
  geom_hline(yintercept = 3600, color='red') +
  geom_hline(yintercept = 36000, color='red') +
  geom_hline(yintercept = 86400, color='red') +
  geom_text(aes(min(unique(all_times_orig$n_train)), 1800, label = '30 min', vjust = -1, hjust=0), color='red', check_overlap = T) +
  geom_text(aes(min(unique(all_times_orig$n_train)), 3600, label = '1 h', vjust = -1, hjust=0), color='red', check_overlap = T) +
  geom_text(aes(min(unique(all_times_orig$n_train)), 36000, label = '10 h', vjust = -1, hjust=0), color='red', check_overlap = T) +
  geom_text(aes(min(unique(all_times_orig$n_train)), 86400, label = '24 h', vjust = -1, hjust=0), color='red', check_overlap = T) +
  scale_y_continuous(trans='log10')+
  scale_color_manual(values = colors)+
  scale_shape_manual(values = c(0, 15, 1, 16, 2, 3, 4, 17, 5, 6, 18, 7, 8, 19, 9))+
  theme_bw()+
  theme(text = element_text(size=15), axis.text.x = element_text(angle = 15, hjust=1))
ggsave("./plots/all_times_original.pdf",height=6, width=12)

# without pipr
ggplot(all_times_orig[Model != "PIPR"], aes(x=n_train, y = `Time [s]`, color = Model, group=Model))+
  geom_point(size=3)+
  geom_line(size=2, alpha=0.5)+
  scale_x_continuous(breaks = unique(all_times_orig$n_train), 
                     labels = paste0(c('Huang (', 'Guo (', 'Du (', 'Pan (', 'Richoux Regular (', 'Richoux Strict ('),
                                     unique(all_times_orig$n_train), rep(')', 6)),
                     guide = guide_axis(n.dodge = 2))+
  labs(x = "Dataset (n training)", y = "Time [s]") +
  geom_hline(yintercept = 1800, color='red') +
  geom_hline(yintercept = 3600, color='red') +
  geom_text(aes(0, 1800, label = '30 min', vjust = -1, hjust=0), color='red') +
  geom_text(aes(0, 3600, label = '1 h', vjust = -1, hjust=0), color='red')+
  scale_color_manual(values = brewer.pal(12, "Paired")[-c(11, 12)])+
  theme_bw()+
  theme(text = element_text(size=20))

##### rewired
# training data size
sprint_data_dir <- '../algorithms/SPRINT/data/rewired/'
training_files <- list.files(path=sprint_data_dir, pattern = 'train_pos')
train_sizes <- sapply(paste0(sprint_data_dir, training_files), function(x){
  as.integer(system2("wc",
                     args = c("-l",
                              x,
                              " | awk '{print $1}'"),
                     stdout = TRUE)) * 2
}
)
training_files[grepl('richoux', training_files, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', training_files[grepl('richoux', training_files, fixed=TRUE)])
names(train_sizes) <- stringr::str_to_title(tstrsplit(training_files, '_', keep=1)[[1]])
#train_sizes <- prettyNum(train_sizes, big.mark = ',')

all_times_rew <- all_times[Test == 'rewired']
all_times_rew <- all_times_rew[, Model := factor(Model, 
                                                   levels=c("RF_PCA","SVM_PCA", "RF_MDS", "SVM_MDS",
                                                            "RF_node2vec",  "SVM_node2vec", "SPRINT", 
                                                            "DeepPPI_FC", "DeepPPI_LSTM",  
                                                            "DeepFE", "PIPR"))]
all_times_rew[, n_train := train_sizes[as.character(Dataset)]]

ggplot(all_times_rew, aes(x=n_train, y = `Time [s]`, color = Model, group=Model))+
  geom_point(size=3)+
  geom_line(size=2, alpha=0.5)+
  scale_x_continuous(breaks = unique(all_times_rew$n_train), 
                     labels = paste0(c('Huang (', 'Guo (', 'Du (', 'Pan (', 'Richoux regular (', 'Richoux strict ('),
                                     unique(all_times_orig$n_train), rep(')', 6)),
                     guide = guide_axis(n.dodge = 2))+
  labs(x = "Dataset (n training)", y = "Time [s]") +
  geom_hline(yintercept = 1800, color='red') +
  geom_hline(yintercept = 3600, color='red') +
  geom_hline(yintercept = 7200, color='red') +
  geom_text(aes(0, 1800, label = '30 min', vjust = -1, hjust=0), color='red') +
  geom_text(aes(0, 3600, label = '1 h', vjust = -1, hjust=0), color='red') +
  geom_text(aes(0, 7200, label = '2 h', vjust = -1, hjust=0), color='red') +
  scale_color_manual(values = brewer.pal(12, "Paired")[-11])+
  theme_bw()+
  theme(text = element_text(size=20))
ggsave("./plots/all_times_rewired.pdf",height=8, width=12)

#### partitions
sprint_data_dir <- '../algorithms/SPRINT/data/partitions/'
training_files <- list.files(path=sprint_data_dir, pattern = 'pos')
partition_sizes <- sapply(paste0(sprint_data_dir, training_files), function(x){
  as.integer(system2("wc",
                     args = c("-l",
                              x,
                              " | awk '{print $1}'"),
                     stdout = TRUE)) * 2
}
)
filenames <- tstrsplit(training_files, '_', keep=c(1,3))
names(partition_sizes) <- stringr::str_to_title(paste(filenames[[1]], filenames[[2]]))

all_times_part <- all_times[Test == 'partition']
all_times_part[, training_dataset := stringr::str_to_title(paste(Dataset, lapply(strsplit(all_times_part$Model, '_'), function(x) x[length(x)-1])))]
all_times_part[, n_train := partition_sizes[as.character(training_dataset)]]

ggplot(all_times_part, aes(x=n_train, y = `Time [s]`, color = Model, group=Model))+
  geom_point(size=3)+
  geom_line(size=2, alpha=0.5)+
  labs(x = "Dataset (n training)", y = "Time [s]") +
  scale_x_continuous(breaks = unique(all_times_part[, .(training_dataset, n_train)])$n_train, 
                     labels = paste0(unique(all_times_part[, .(training_dataset, n_train)])$training_dataset,
                                     rep(' (', 6),
                                     unique(all_times_part[, .(training_dataset, n_train)])$n_train, 
                                     rep(')', 6)),
                     guide = guide_axis(n.dodge = 6))+
  geom_hline(yintercept = 1800, color='red') +
  geom_hline(yintercept = 3600, color='red')+
  geom_text(aes(0, 1800, label = '30 min', vjust = -1, hjust=0), color='red') +
  geom_text(aes(0, 3600, label = '1 h', vjust = -1, hjust=0), color='red')+
  theme_bw()+
  theme(text = element_text(size=10))

