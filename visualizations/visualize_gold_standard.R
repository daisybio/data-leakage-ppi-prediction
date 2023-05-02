library(data.table)
library(ggplot2)
library(RColorBrewer)

measure <- 'Accuracy'

#### result prefixes
custom_res <- '../algorithms/Custom/results/'
deepFE_res <- '../algorithms/DeepFE-PPI/result/custom/gold_standard_test/'
deepPPI_res <- '../algorithms/DeepPPI/keras/results_custom/'
seqppi_res <- '../algorithms/seq_ppi/binary/model/lasagna/results/'
sprint_res <- '../algorithms/SPRINT/results/original/'
dscript_res <- '../algorithms/D-SCRIPT-main/results_dscript/original/'
tt_res <- '../algorithms/D-SCRIPT-main/results_topsyturvy/original/'

# read in data
all_results <- data.table(1)[, `:=` (c("Model", "Dataset", measure), NA)][, V1 := NULL][.0]

# custom
custom_results <- lapply(paste0(custom_res,  list.files(custom_res, pattern='^gold.*.csv')), fread)
file_names <- tstrsplit(list.files(custom_res, pattern='^gold.*.csv'), '.csv', keep=1)[[1]]
names(custom_results) <- file_names
custom_results <- rbindlist(custom_results, idcol = 'filename')
custom_results[, c('dataset', 'encoding', 'method') := tstrsplit(filename, '_', keep=c(1,3,4))]
custom_results[, Model := paste(method, encoding, sep = '_')]
if(measure == 'Recall'){
  custom_results <- custom_results[V1 == 'Sensitivity']
}else{
  custom_results <- custom_results[V1 == measure]
}
colnames(custom_results) <- c('filename', 'Measure', measure, 'Dataset', 'Encoding', 'Method', 'Model')

all_results <- rbind(all_results, custom_results[, c('Model', 'Dataset', measure), with=FALSE])

# deepFE
deepFE_results <- lapply(paste0(deepFE_res, list.files(deepFE_res, pattern = '^gold.*csv', recursive = TRUE)), fread)
names(deepFE_results) <- c('gold')
deepFE_results <- rbindlist(deepFE_results, idcol = 'Dataset')
deepFE_results <- deepFE_results[V1 == measure]
colnames(deepFE_results) <- c('Dataset', 'Measure', measure)
deepFE_results$Model <- 'DeepFE'

all_results <- rbind(all_results, deepFE_results[, c('Model', 'Dataset', measure), with=FALSE])

# deepPPI
deepPPI_results <- lapply(paste0(deepPPI_res, list.files(deepPPI_res, pattern='gold.*.csv')), fread)
file_names <- tstrsplit(list.files(deepPPI_res, pattern='gold.*.csv'), '.csv', keep=1)[[1]]
names(deepPPI_results) <- file_names
deepPPI_results <- rbindlist(deepPPI_results, idcol='filename')
deepPPI_results <- deepPPI_results[, c('Model', 'Dataset') := tstrsplit(filename, '_', keep = c(1,3))]
deepPPI_results <- deepPPI_results[variable == measure]
deepPPI_results[, Model := paste('deepPPI', Model, sep='_')]
colnames(deepPPI_results) <- c('filename', 'variable', measure, 'Model', 'Dataset')

all_results <- rbind(all_results, deepPPI_results[, c('Model', 'Dataset', measure), with=FALSE])

# PIPR
pipr_results <- lapply(paste0(seqppi_res, list.files(seqppi_res, pattern='^gold.*.csv')), fread)
names(pipr_results) <- 'gold'
pipr_results <- rbindlist(pipr_results, idcol='Filename')
pipr_results <- pipr_results[V1 == measure]
pipr_results <- pipr_results[, c('Test', 'Dataset') := tstrsplit(Filename, '_')]
pipr_results$Model <- 'PIPR'
colnames(pipr_results) <- c('Filename', 'Measure', measure, 'Test', 'Dataset', 'Model')

all_results <- rbind(all_results, pipr_results[, c('Model', 'Dataset', measure), with=FALSE])

# SPRINT
sprint_results <- fread(paste0(sprint_res, 'all_results.tsv'))
sprint_results$Model <- 'SPRINT'
if(measure == 'Accuracy'){
  colnames(sprint_results) <- c('Dataset', measure, 'AUPR', 'Model')
}else{
  colnames(sprint_results) <- c('Dataset', 'AUC', measure, 'Model')
}
sprint_results <- sprint_results[Dataset == 'gold_standard']
all_results <- rbind(all_results, sprint_results[, c('Model', 'Dataset', measure), with=FALSE])

# DSCRIPT
dscript_results <- fread(paste0(dscript_res, 'all_results.tsv'))
dscript_results <- dscript_results[Dataset == 'gold' & Metric == measure]
colnames(dscript_results) <- c('Model', 'Dataset', 'Metric', measure, 'Split')
dscript_results$Model <- "D-SCRIPT"
all_results <- rbind(all_results, dscript_results[, c('Model', 'Dataset', measure), with=FALSE])

# TopsyTurvy
tt_results <- fread(paste0(tt_res, 'all_results.tsv'))
tt_results <- tt_results[Dataset == 'gold' & Metric == measure]
colnames(tt_results) <- c('Model', 'Dataset', 'Metric', measure, 'Split')
tt_results$Model <- "Topsy_Turvy"
all_results <- rbind(all_results, tt_results[, c('Model', 'Dataset', measure), with=FALSE])

all_results$Dataset <- 'gold_standard'

all_results <- all_results[, Model := factor(Model, 
                                             levels=c("RF_PCA","SVM_PCA", "RF_MDS", "SVM_MDS",
                                                      "RF_node2vec",  "SVM_node2vec", "SPRINT", 
                                                      "deepPPI_FC", "deepPPI_LSTM",  
                                                      "DeepFE", "PIPR", "D-SCRIPT", "Topsy_Turvy"))]
fwrite(all_results, file=paste0('results/gold_standard_', measure, '.csv'))

