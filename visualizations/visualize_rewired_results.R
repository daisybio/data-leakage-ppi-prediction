library(data.table)
library(ggplot2)
library(RColorBrewer)

#### result prefixes
custom_res <- '../algorithms/Custom/results/'
deepFE_res <- '../algorithms/DeepFE-PPI/result/custom/'
deepPPI_res <- '../algorithms/DeepPPI/keras/results_custom/'
seqppi_res <- '../algorithms/seq_ppi/binary/model/lasagna/results/'
sprint_res <- '../algorithms/SPRINT/results/rewired/'

# read in data
all_results <- data.table(Model=character(), Dataset=character(), Accuracy=numeric())

# custom
custom_results <- lapply(paste0(custom_res,  list.files(custom_res, pattern='^rewired_(du|guo|huang|pan|richoux).*.csv')), fread)
file_names <- tstrsplit(list.files(custom_res, pattern='^rewired_(du|guo|huang|pan|richoux).*.csv'), '.csv', keep=1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
names(custom_results) <- file_names
custom_results <- rbindlist(custom_results, idcol = 'filename')
custom_results[, c('dataset', 'encoding', 'model') := tstrsplit(filename, '_', keep=c(2,3,4))]
custom_results[, Model := paste(model, encoding, sep = '_')]
custom_results <- custom_results[V1 == 'Accuracy']
colnames(custom_results) <- c('filename', 'Measure', 'Accuracy', 'Dataset', 'Encoding', 'Method', 'Model')

all_results <- rbind(all_results, custom_results[, c('Model', 'Dataset', 'Accuracy')])

# deepFE
deepFE_results <- lapply(paste0(deepFE_res, list.files(deepFE_res, pattern = '^rewired_scores_(du|guo|huang|pan|richoux_regular|richoux_strict).csv', recursive = TRUE)), fread)
file_names <- list.files(deepFE_res)[-c(5, 8)]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
names(deepFE_results) <- file_names
deepFE_results <- rbindlist(deepFE_results, idcol = 'Dataset')
deepFE_results <- deepFE_results[V1 == 'Accuracy']
colnames(deepFE_results) <- c('Dataset', 'Measure', 'Accuracy')
deepFE_results$Model <- 'DeepFE'

all_results <- rbind(all_results, deepFE_results[, c('Model', 'Dataset', 'Accuracy')])

# deepPPI
deepPPI_results <- lapply(paste0(deepPPI_res, list.files(deepPPI_res, pattern='(FC|LSTM)_rewired_(du|guo|huang|pan|richoux).*.csv')), fread)
file_names <- tstrsplit(list.files(deepPPI_res, pattern='(FC|LSTM)_rewired_(du|guo|huang|pan|richoux).*.csv'), '.csv', keep=1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
names(deepPPI_results) <- file_names
deepPPI_results <- rbindlist(deepPPI_results, idcol='filename')
deepPPI_results <- deepPPI_results[, c('Model', 'Dataset') := tstrsplit(filename, '_', keep = c(1,3))]
n_train <- unique(deepPPI_results[variable == 'n_train', c('Dataset', 'variable', 'value')])
deepPPI_results <- deepPPI_results[variable == 'Accuracy']
deepPPI_results[, Model := paste('deepPPI', Model, sep='_')]
colnames(deepPPI_results) <- c('filename', 'variable', 'Accuracy', 'Model', 'Dataset')

all_results <- rbind(all_results, deepPPI_results[, c('Model', 'Dataset', 'Accuracy')])

# PIPR
pipr_results <- lapply(paste0(seqppi_res, list.files(seqppi_res, pattern='^rewired_(du|guo|huang|pan|richoux_regular|richoux_strict).csv')), fread)
file_names <- tstrsplit(list.files(seqppi_res, pattern='^rewired_(du|guo|huang|pan|richoux_regular|richoux_strict).csv'), '.csv', keep=1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
names(pipr_results) <- file_names
pipr_results <- rbindlist(pipr_results, idcol='Filename')
pipr_results[, Dataset := tstrsplit(Filename, 'rewired_', keep=2)]
pipr_results <- pipr_results[V1 == 'Accuracy']
pipr_results$Model <- 'PIPR'
colnames(pipr_results) <- c('Filename', 'Measure', 'Accuracy', 'Dataset', 'Model')

all_results <- rbind(all_results, pipr_results[, c('Model', 'Dataset', 'Accuracy')])

# SPRINT
sprint_results <- fread(paste0(sprint_res, 'all_results.tsv'))
sprint_results$Model <- 'SPRINT'
colnames(sprint_results) <- c('Dataset', 'Accuracy', 'AUPR', 'Model')
sprint_results$Dataset[grepl('richoux', sprint_results$Dataset, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', sprint_results$Dataset[grepl('richoux', sprint_results$Dataset, fixed=TRUE)])
all_results <- rbind(all_results, sprint_results[, c('Model', 'Dataset', 'Accuracy')])

# visualization
all_results <- all_results[, Dataset := factor(Dataset, 
                                               levels = c("huang", "guo", "du", "pan", "richoux-regular", "richoux-strict"))]

all_results <- all_results[, Model := factor(Model, 
                                             levels=c("RF_PCA","SVM_PCA", "RF_MDS", "SVM_MDS",
                                                      "RF_node2vec",  "SVM_node2vec", "SPRINT", 
                                                      "deepPPI_FC", "deepPPI_LSTM",  
                                                      "DeepFE", "PIPR"))]
fwrite(all_results, file='results/rewired.csv')

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

ggplot(all_results, aes(x=Dataset, y = Accuracy, color = Model, group=Model))+
  geom_line(size=1, alpha=0.7)+
  geom_point(size=3)+
  scale_x_discrete(labels=c("huang" = paste0("Huang (", train_sizes["huang"], ")"), 
                            "guo" = paste0("Guo (", train_sizes["guo"], ")"),
                            "du" = paste0("Du (", train_sizes["du"], ")"), 
                            "pan" = paste0("Pan (", train_sizes["pan"], ")"),
                            "richoux-regular" = paste("Richoux regular (", train_sizes["richoux-regular"], ")"),
                            "richoux-strict" = paste("Richoux strict (", train_sizes["richoux-strict"], ")")))+
  ylim(0.5, 1.0)+
  labs(x = "Dataset (n training)", y = "Accuracy/AUC for SPRINT") +
  scale_color_manual(values = brewer.pal(12, "Paired")[-11])+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))
ggsave("./all_results_rewired.png",height=8, width=12)  
