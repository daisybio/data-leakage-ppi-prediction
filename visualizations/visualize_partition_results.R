library(data.table)
library(ggplot2)
library(RColorBrewer)

measure <- 'Accuracy'

#### result prefixes
custom_res <- '../algorithms/Custom/results/partition_tests/'
deepFE_res <- '../algorithms/DeepFE-PPI/result/custom/'
deepPPI_res <- '../algorithms/DeepPPI/keras/results_custom/'
seqppi_res <- '../algorithms/seq_ppi/binary/model/lasagna/results/'
sprint_res <- '../algorithms/SPRINT/results/partitions/'
dscript_res <- '../algorithms/D-SCRIPT-main/results_dscript/partitions/'
tt_res <- '../algorithms/D-SCRIPT-main/results_topsyturvy/partitions/'

# read in data
all_results <- data.table(1)[, `:=` (c("Model", "Dataset", measure, "Partition"), NA)][, V1 := NULL][.0]

# custom
custom_results <- lapply(paste0(custom_res,  list.files(custom_res, pattern = '^(du|guo|huang|pan|richoux)')), fread)
file_names <- tstrsplit(list.files(custom_res, pattern = '^(du|guo|huang|pan|richoux)'), '.csv', keep=1)[[1]]
names(custom_results) <- file_names
custom_results <- rbindlist(custom_results, idcol = 'filename')
custom_results[, c('dataset', 'encoding', 'model', 'train', 'test') := tstrsplit(filename, '_')]

degree_results <- lapply(paste0(custom_res,  list.files(custom_res, pattern = '^partition')), fread)
file_names <- tstrsplit(list.files(custom_res, pattern = '^partition'), '.csv', keep=1)[[1]]
names(degree_results) <- file_names
degree_results <- rbindlist(degree_results, idcol = 'filename')
degree_results[, c('partition', 'dataset', 'train', 'test', 'model') := tstrsplit(filename, '_')]

if(measure == 'Recall'){
  custom_results <- custom_results[V1 == 'Sensitivity']
  degree_results <- degree_results[V1 == 'Sensitivity']
}else{
  custom_results <- custom_results[V1 == measure]
  degree_results <- degree_results[V1 == measure]
}
custom_results[, Model := paste(model, encoding, sep = '_')]
degree_results[, Model := paste('degree', model, sep = '_')]
colnames(custom_results) <- c('filename', 'Measure', measure, 'Dataset', 'Encoding', 'model', 'Train', 'Test', 'Model')
colnames(degree_results) <- c('filename', 'Measure', measure, 'partition','Dataset', 'Train', 'Test', 'model', 'Model')
custom_results[, Partition := paste(Train, Test, sep='->')]
degree_results[, Partition := paste(Train, Test, sep='->')]

all_results <- rbind(all_results, custom_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])
all_results <- rbind(all_results, degree_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])

# deepFE
deepFE_results <- lapply(paste0(deepFE_res, list.files(deepFE_res, pattern = '^partition_scores_(du|guo|huang|pan|richoux)_(both|0)_(0|1).csv', recursive = TRUE)), fread)
file_names <- tstrsplit(list.files(deepFE_res, pattern = '^partition_scores_(du|guo|huang|pan|richoux)_(both|0)_(0|1).csv', recursive = TRUE), '/', keep=2)[[1]]
names(deepFE_results) <- tstrsplit(file_names, '.csv', keep=1)[[1]]
deepFE_results <- rbindlist(deepFE_results, idcol = 'filename')
deepFE_results <- deepFE_results[V1 == measure]
deepFE_results[, c('dataset', 'train', 'test') := tstrsplit(filename, '_', keep=c(3,4,5))]
colnames(deepFE_results) <- c('filename', 'Measure', measure, 'Dataset', 'Train', 'Test')
deepFE_results$Model <- 'DeepFE'
deepFE_results[, Partition := paste(Train, Test, sep='->')]

all_results <- rbind(all_results, deepFE_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])

# deepPPI
deepPPI_results <- lapply(paste0(deepPPI_res, list.files(deepPPI_res, pattern='partition_(du|guo|huang|pan|richoux).*.csv')), fread)
file_names <- tstrsplit(list.files(deepPPI_res, pattern='partition_(du|guo|huang|pan|richoux).*.csv'), '.csv', keep=1)[[1]]
names(deepPPI_results) <- file_names
deepPPI_results <- rbindlist(deepPPI_results, idcol='filename')
deepPPI_results <- deepPPI_results[, c('Model', 'Dataset', 'Train', 'Test') := tstrsplit(filename, '_', keep = c(1,3, 4, 5))]
deepPPI_results[, Train := tstrsplit(Train, 'tr', keep=2)]
deepPPI_results[, Test := tstrsplit(Test, 'te', keep=2)]
deepPPI_results[, Partition := paste(Train, Test, sep='->')]
n_train <- unique(deepPPI_results[variable %in% c('n_train'), c('Dataset', 'Partition', 'variable', 'value')])
deepPPI_results <- deepPPI_results[variable == measure]
deepPPI_results[, Model := paste('deepPPI', Model, sep='_')]
colnames(deepPPI_results) <- c('filename', 'variable', measure, 'Model', 'Dataset', 'Train', 'Test', 'Partition')

all_results <- rbind(all_results, deepPPI_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])

# PIPR
pipr_results <- lapply(paste0(seqppi_res, list.files(seqppi_res, pattern='^partition_(du|guo|huang|pan|richoux)_(both|0)_(0|1).csv')), fread)
file_names <- tstrsplit(list.files(seqppi_res, pattern='^partition_(du|guo|huang|pan|richoux)_(both|0)_(0|1).csv'), '.csv', keep=1)[[1]]
names(pipr_results) <- file_names
pipr_results <- rbindlist(pipr_results, idcol='filename')
pipr_results <- pipr_results[, c('Dataset', 'Train', 'Test') := tstrsplit(filename, '_', keep=c(2,3,4))]
pipr_results <- pipr_results[V1 == measure]
pipr_results$Model <- 'PIPR'
colnames(pipr_results) <- c('filename', 'Measure', measure, 'Dataset', 'Train', 'Test','Model')
pipr_results[, Partition := paste(Train, Test, sep='->')]

all_results <- rbind(all_results, pipr_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])

# SPRINT
sprint_results <- fread(paste0(sprint_res, 'all_results.tsv'))
sprint_results$Model <- 'SPRINT'
if(measure == 'Accuracy'){
  colnames(sprint_results) <- c('Dataset', 'Train', 'Test', measure, 'AUPR', 'Model')
}else{
  colnames(sprint_results) <- c('Dataset', 'Train', 'Test', 'AUC', measure, 'Model')
}
sprint_results[, Partition := paste(Train, Test, sep='->')]

all_results <- rbind(all_results, sprint_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])

# D-Script
dscript_results <- fread(paste0(dscript_res, 'all_results.tsv'))
dscript_results <- dscript_results[Metric == measure]
colnames(dscript_results) <- c('Model', 'Dataset', 'Metric', measure, 'Partition')
dscript_results$Model <- 'D-SCRIPT'
dscript_results[Dataset == 'richoux_regular', Dataset := 'richoux-regular']
dscript_results[Dataset == 'richoux_strict', Dataset := 'richoux-strict']
all_results <- rbind(all_results, dscript_results[, c('Model', 'Dataset', measure, 'Partition'), with=FALSE])

# Topsy_Turvy
tt_results <- fread(paste0(tt_res, 'all_results.tsv'))
tt_results <- tt_results[Metric == measure]
colnames(tt_results) <- c('Model', 'Dataset', 'Metric', measure, 'Partition')
tt_results$Model <- 'Topsy_Turvy'
tt_results[Dataset == 'richoux_regular', Dataset := 'richoux-regular']
tt_results[Dataset == 'richoux_strict', Dataset := 'richoux-strict']
all_results <- rbind(all_results, tt_results[, c('Model', 'Dataset', measure, 'Partition'), with=FALSE])

# visualization
all_results <- all_results[, Dataset := factor(Dataset, 
                                               levels = c("huang", "guo", "du", "pan", "richoux"))]

all_results <- all_results[, Model := factor(Model, 
                                             levels=c("RF_PCA","SVM_PCA", "RF_MDS", "SVM_MDS",
                                                      "RF_node2vec",  "SVM_node2vec", "degree_cons", "degree_hf", "SPRINT", 
                                                      "deepPPI_FC", "deepPPI_LSTM",  
                                                      "DeepFE", "PIPR", "D-SCRIPT", "Topsy_Turvy"))]
fwrite(all_results, file=paste0('results/partition_', measure, '.csv'))

# training data size
sprint_data_dir <- '../algorithms/SPRINT/data/partitions/'
training_files <- list.files(path=sprint_data_dir, pattern = 'pos')
train_sizes <- sapply(paste0(sprint_data_dir, training_files), function(x){
  as.integer(system2("wc",
                     args = c("-l",
                              x,
                              " | awk '{print $1}'"),
                     stdout = TRUE)) * 2
}
)
filenames <- tstrsplit(training_files, '_', keep=c(1,3))
names(train_sizes) <- paste(filenames[[1]], filenames[[2]])
train_sizes <- prettyNum(train_sizes, big.mark = ',')

ggplot(all_results, aes(x=Dataset, y = get(measure), color = Model, group=Model))+
  geom_line(size=1, alpha=0.7)+
  geom_point(size=3)+
  scale_x_discrete(labels=c("huang" = paste0("Huang\n(",train_sizes["huang 0"], "|\n", train_sizes["huang both"], ")"), 
                            "guo" = paste0("Guo\n(",train_sizes["guo 0"], "|\n", train_sizes["guo both"], ")"),
                            "du" = paste0("Du\n(",train_sizes["du 0"], "|\n", train_sizes["du both"], ")"), 
                            "pan" = paste0("Pan\n(",train_sizes["pan 0"], "|\n", train_sizes["pan both"], ")"), 
                            "richoux" = paste0("Richoux\n(",train_sizes["richoux 0"], "|\n", train_sizes["richoux both"], ")"))
  )+
  ylim(0.4, 1.0)+
  facet_wrap(~Partition)+
  labs(x = "Dataset\n(n training partition 0|both)", y = paste0(measure, "/", ifelse(measure=='Accuracy', 'AUC', 'AUPR'), " for SPRINT")) +
  scale_color_manual(values = c(brewer.pal(12, "Paired")[-11], '#FF3393', '#21D5C1'))+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 0, vjust = 0.5, hjust=0.5))
#ggsave(paste0("plots/all_results_partition_", measure, ".png"),height=8, width=18)  

