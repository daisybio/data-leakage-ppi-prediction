library(data.table)
library(ggplot2)
library(RColorBrewer)

measure <- 'Accuracy'

#### result prefixes
custom_res <- '../algorithms/Custom/results/'
deepFE_res <- '../algorithms/DeepFE-PPI/result/custom/'
deepPPI_res <- '../algorithms/DeepPPI/keras/results_custom/'
seqppi_res <- '../algorithms/seq_ppi/binary/model/lasagna/results/'
sprint_res <- '../algorithms/SPRINT/results/rewired/'
dscript_res <- '../algorithms/D-SCRIPT-main/results_dscript/rewired/'
tt_res <- '../algorithms/D-SCRIPT-main/results_topsyturvy/rewired/'

# read in data
all_results <- data.table(1)[, `:=` (c("Model", "Dataset", measure), NA)][, V1 := NULL][.0]

# custom
custom_results <- lapply(paste0(custom_res,  list.files(custom_res, pattern='^rewired_(du|guo|huang|pan|richoux).*.csv')), fread)
file_names <- tstrsplit(list.files(custom_res, pattern='^rewired_(du|guo|huang|pan|richoux).*.csv'), '.csv', keep=1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
names(custom_results) <- file_names
custom_results <- rbindlist(custom_results, idcol = 'filename')
custom_results[, c('dataset', 'encoding', 'model') := tstrsplit(filename, '_', keep=c(2,3,4))]
custom_results[is.na(model), model := 'degree']
custom_results[, Model := paste(model, encoding, sep = '_')]
if(measure == 'Recall'){
  custom_results <- custom_results[V1 == 'Sensitivity']
}else{
  custom_results <- custom_results[V1 == measure]
}
colnames(custom_results) <- c('filename', 'Measure', measure, 'Dataset', 'Encoding', 'Method', 'Model')

all_results <- rbind(all_results, custom_results[, c('Model', 'Dataset', measure), with = FALSE])

# deepFE
deepFE_results <- lapply(paste0(deepFE_res, list.files(deepFE_res, pattern = '^rewired_scores_(du|guo|huang|pan|richoux_regular|richoux_strict).csv', recursive = TRUE)), fread)
file_names <- tstrsplit(list.files(deepFE_res, pattern = '^rewired_scores_(du|guo|huang|pan|richoux_regular|richoux_strict).csv', recursive = TRUE), '/', keep = 1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
names(deepFE_results) <- file_names
deepFE_results <- rbindlist(deepFE_results, idcol = 'Dataset')
deepFE_results <- deepFE_results[V1 == measure]
colnames(deepFE_results) <- c('Dataset', 'Measure', measure)
deepFE_results$Model <- 'DeepFE'

all_results <- rbind(all_results, deepFE_results[, c('Model', 'Dataset', measure), with = FALSE])

# deepPPI
deepPPI_results <- lapply(paste0(deepPPI_res, list.files(deepPPI_res, pattern='(FC|LSTM)_rewired_(du|guo|huang|pan|richoux).*.csv')), fread)
file_names <- tstrsplit(list.files(deepPPI_res, pattern='(FC|LSTM)_rewired_(du|guo|huang|pan|richoux).*.csv'), '.csv', keep=1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
names(deepPPI_results) <- file_names
deepPPI_results <- rbindlist(deepPPI_results, idcol='filename')
deepPPI_results <- deepPPI_results[, c('Model', 'Dataset') := tstrsplit(filename, '_', keep = c(1,3))]
n_train <- unique(deepPPI_results[variable == 'n_train', c('Dataset', 'variable', 'value')])
deepPPI_results <- deepPPI_results[variable == measure]
deepPPI_results[, Model := paste('deepPPI', Model, sep='_')]
colnames(deepPPI_results) <- c('filename', 'variable', measure, 'Model', 'Dataset')

all_results <- rbind(all_results, deepPPI_results[, c('Model', 'Dataset', measure), with = FALSE])

# PIPR
pipr_results <- lapply(paste0(seqppi_res, list.files(seqppi_res, pattern='^rewired_(du|guo|huang|pan|richoux_regular|richoux_strict).csv')), fread)
file_names <- tstrsplit(list.files(seqppi_res, pattern='^rewired_(du|guo|huang|pan|richoux_regular|richoux_strict).csv'), '.csv', keep=1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
names(pipr_results) <- file_names
pipr_results <- rbindlist(pipr_results, idcol='Filename')
pipr_results[, Dataset := tstrsplit(Filename, 'rewired_', keep=2)]
pipr_results <- pipr_results[V1 == measure]
pipr_results$Model <- 'PIPR'
colnames(pipr_results) <- c('Filename', 'Measure', measure, 'Dataset', 'Model')

all_results <- rbind(all_results, pipr_results[, c('Model', 'Dataset', measure), with = FALSE])

# SPRINT
sprint_results <- fread(paste0(sprint_res, 'all_results.tsv'))
sprint_results$Model <- 'SPRINT'
if(measure == 'Accuracy'){
  colnames(sprint_results) <- c('Dataset', measure, 'AUPR', 'Model')
}else{
  colnames(sprint_results) <- c('Dataset', 'AUC', measure, 'Model')
}
sprint_results$Dataset[grepl('richoux', sprint_results$Dataset, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', sprint_results$Dataset[grepl('richoux', sprint_results$Dataset, fixed=TRUE)])
all_results <- rbind(all_results, sprint_results[, c('Model', 'Dataset', measure), with = FALSE])

# D-Script
dscript_results <- fread(paste0(dscript_res, 'all_results.tsv'))
dscript_results <- dscript_results[Metric == measure]
colnames(dscript_results) <- c('Model', 'Dataset', 'Metric', measure, 'Split')
dscript_results$Model <- 'D-SCRIPT'
dscript_results[Dataset == 'richoux_regular', Dataset := 'richoux-regular']
dscript_results[Dataset == 'richoux_strict', Dataset := 'richoux-strict']
all_results <- rbind(all_results, dscript_results[, c('Model', 'Dataset', measure), with=FALSE])

# Topsy_Turvy
tt_results <- fread(paste0(tt_res, 'all_results.tsv'))
tt_results <- tt_results[Metric == measure]
colnames(tt_results) <- c('Model', 'Dataset', 'Metric', measure, 'Split')
tt_results$Model <- 'Topsy_Turvy'
tt_results[Dataset == 'richoux_regular', Dataset := 'richoux-regular']
tt_results[Dataset == 'richoux_strict', Dataset := 'richoux-strict']
all_results <- rbind(all_results, tt_results[, c('Model', 'Dataset', measure), with=FALSE])

# visualization
all_results <- all_results[, Dataset := factor(Dataset, 
                                               levels = c("huang", "guo", "du", "pan", "richoux-regular", "richoux-strict"))]

all_results <- all_results[, Model := factor(Model, 
                                             levels=c("RF_PCA","SVM_PCA", "RF_MDS", "SVM_MDS",
                                                      "RF_node2vec",  "SVM_node2vec", "degree_cons", "degree_hf", "SPRINT", 
                                                      "deepPPI_FC", "deepPPI_LSTM",  
                                                      "DeepFE", "PIPR", "D-SCRIPT", "Topsy_Turvy"))]
fwrite(all_results, file=paste0('results/rewired_', measure, '.csv'))

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
names(train_sizes) <- tstrsplit(training_files, '_', keep=1)[[1]]
train_sizes <- prettyNum(train_sizes, big.mark = ',')

ggplot(all_results, aes(x=Dataset, y = get(measure), color = Model, group=Model))+
  geom_line(size=1, alpha=0.7)+
  geom_point(size=3)+
  scale_x_discrete(labels=c("huang" = paste0("Huang (", train_sizes["huang"], ")"), 
                            "guo" = paste0("Guo (", train_sizes["guo"], ")"),
                            "du" = paste0("Du (", train_sizes["du"], ")"), 
                            "pan" = paste0("Pan (", train_sizes["pan"], ")"),
                            "richoux-regular" = paste("Richoux regular (", train_sizes["richoux-regular"], ")"),
                            "richoux-strict" = paste("Richoux strict (", train_sizes["richoux-strict"], ")")))+
  #ylim(0.5, 1.0)+
  labs(x = "Dataset (n training)", y = paste0(measure, "/", ifelse(measure=='Accuracy', 'AUC', 'AUPR'), " for SPRINT")) +
  scale_color_manual(values = c(brewer.pal(12, "Paired")[-11], '#FF3393', '#21D5C1'))+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))
#ggsave(paste("plots/all_results_rewired_", measure, ".png"),height=8, width=12)  
