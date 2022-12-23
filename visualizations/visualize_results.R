library(data.table)
library(ggplot2)
library(RColorBrewer)
library(pheatmap)
library(latex2exp)

measure <- 'Accuracy'

original_results <- fread(paste0('results/original_', measure, '.csv'))
gold_standard_results <- fread(paste0('results/gold_standard_', measure, '.csv'))
original_results <- rbind(original_results, gold_standard_results)
original_results$Test <- 'Original'
rewired_results <- fread(paste0('results/rewired_', measure, '.csv'))
rewired_results$Test <- 'Rewired'
partition_results <- fread(paste0('results/partition_', measure, '.csv'))
colnames(partition_results) <- c('Model', 'Dataset', measure, 'Test')

all_results <- rbind(original_results, rewired_results)
all_results <- all_results[, Model := factor(Model, 
                                             levels=c('SPRINT', 
                                                      'deepPPI_FC', 'deepPPI_LSTM',  
                                                      'DeepFE', 'PIPR', 'RF_PCA','SVM_PCA', 'RF_MDS', 'SVM_MDS',
                                                      'RF_node2vec',  'SVM_node2vec'))]
all_results <- all_results[, Dataset := factor(Dataset, 
                                               levels = c('gold_standard', 'huang', 'guo', 'du', 'pan', 'richoux-regular', 'richoux-strict'))]

all_results <- rbind(all_results, partition_results)
all_results[, Test := factor(Test, levels=c('Original', 'Rewired', 'both->0', 'both->1', '0->1'))]
all_results[, Model := gsub('deepPPI_FC', 'Richoux-FC', Model)]
all_results[, Model := gsub('deepPPI_LSTM', 'Richoux-LSTM', Model)]
all_results[, Model := gsub('RF_PCA', 'RF PCA', Model)]
all_results[, Model := gsub('SVM_PCA', 'SVM PCA', Model)]
all_results[, Model := gsub('RF_MDS', 'RF MDS', Model)]
all_results[, Model := gsub('SVM_MDS', 'SVM MDS', Model)]
all_results[, Model := gsub('RF_node2vec', 'RF node2vec', Model)]
all_results[, Model := gsub('SVM_node2vec', 'SVM node2vec', Model)]
all_results <- all_results[, Model := factor(Model, 
                                             levels=c('SPRINT', 
                                                      'Richoux-FC', 'Richoux-LSTM',  
                                                      'DeepFE', 'PIPR', 'RF PCA','SVM PCA', 'RF MDS', 'SVM MDS',
                                                      'RF node2vec',  'SVM node2vec'))]

colorBlindBlack8  <- c('#000000', '#E69F00', '#56B4E9', '#009E73', 
                       '#F0E442', '#0072B2', '#D55E00', '#CC79A7')
result_mat <- as.matrix(dcast(all_results, Model ~ Dataset + Test, value.var = measure))
rownames(result_mat) <- result_mat[, 'Model']
result_mat <- result_mat[, -1]
class(result_mat) <- 'numeric'
colnames(result_mat)[colnames(result_mat) == 'gold_standard_Original'] <- 'Gold_Original'
annotation_col <- as.data.frame(tstrsplit(colnames(result_mat), '_', keep = 2), col.names = c('Test'))
rownames(annotation_col) <- colnames(result_mat)
annotation_col$Test <- gsub('both', 'Inter' ,annotation_col$Test)
annotation_col$Test <- gsub('0', 'Intra-0' ,annotation_col$Test)
annotation_col$Test <- gsub('1', 'Intra-1' ,annotation_col$Test)
annotation_col$Test <- factor(annotation_col$Test, 
                              levels = c('Original', 'Rewired', 'Inter->Intra-1', 'Inter->Intra-0', 'Intra-0->Intra-1'))
annotation_col <- annotation_col[order(annotation_col$Test), , drop = FALSE]
result_mat <- result_mat[, rownames(annotation_col)]

# training data sizes
get_sizes <- function(directory) {
  sprint_data_dir <- paste0('../algorithms/SPRINT/data/', directory, '/')
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
  return(train_sizes)
}

original_sizes <- get_sizes('original')
original_sizes <- c(original_sizes, c('gold' = prettyNum(as.integer(system2("wc",
                   args = c("-l", '../Datasets_PPIs/Hippiev2.3/Intra0_Intra1_pos_rr.txt',
                            " | awk '{print $1}'"),
                   stdout = TRUE)) * 2, big.mark = ',')))
rewired_sizes <- get_sizes('rewired')
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
names(partition_sizes) <- paste(filenames[[1]], filenames[[2]])
partition_sizes <- prettyNum(partition_sizes, big.mark = ',')

pheatmap(t(result_mat),
         annotation_row = annotation_col,
         annotation_colors = list(
                                  Test = c('Original'='#AA4499', 'Rewired'='#DDCC77','Inter->Intra-1'='#888888', 'Inter->Intra-0'='#44AA99', 'Intra-0->Intra-1'='#661100')
                                  ),
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         gaps_col = 5,
         gaps_row = c(7,13,18,23),
         display_numbers = TRUE,
         legend = FALSE,
         filename = paste0('plots/heatmap_results_', measure, '.pdf'),
         width=8,
         height=10,
         cex = 1,
         labels_row = c(
           paste0('GOLD STANDARD (', original_sizes['gold'], ')'),
           paste0('HUANG (', original_sizes['huang'], ')'),
           paste0('GUO (', original_sizes['guo'], ')'),
           paste0('DU (', original_sizes['du'], ')'),
           paste0('PAN (', original_sizes['pan'], ')'),
           paste0('RICHOUX-REGULAR (', original_sizes['richoux-regular'], ')'),
           paste0('RICHOUX-STRICT (', original_sizes['richoux-strict'], ')'),
           #rewired
           paste0('HUANG (', rewired_sizes['huang'], ')'),
           paste0('GUO (', rewired_sizes['guo'], ')'),
           paste0('DU (', rewired_sizes['du'], ')'),
           paste0('PAN (', rewired_sizes['pan'], ')'),
           paste0('RICHOUX-REGULAR (', rewired_sizes['richoux-regular'], ')'),
           paste0('RICHOUX-STRICT (', rewired_sizes['richoux-strict'], ')'),
           #partition both ->1
           paste0('HUANG (', partition_sizes['huang both'], ')'),
           paste0('GUO (', partition_sizes['guo both'], ')'),
           paste0('DU (', partition_sizes['du both'], ')'),
           paste0('PAN (', partition_sizes['pan both'], ')'),
           paste0('RICHOUX (', partition_sizes['richoux both'], ')'),
           #partition both -> 0
           paste0('HUANG (', partition_sizes['huang both'], ')'),
           paste0('GUO (', partition_sizes['guo both'], ')'),
           paste0('DU (', partition_sizes['du both'], ')'),
           paste0('PAN (', partition_sizes['pan both'], ')'),
           paste0('RICHOUX (', partition_sizes['richoux both'], ')'),
           #partition 0 -> 1
           paste0('HUANG (', partition_sizes['huang 0'], ')'),
           paste0('GUO (', partition_sizes['guo 0'], ')'),
           paste0('DU (', partition_sizes['du 0'], ')'),
           paste0('PAN (', partition_sizes['pan 0'], ')'),
           paste0('RICHOUX (', partition_sizes['richoux 0'], ')')
         ),
         labels_col = c('SPRINT', 'Richoux-\nFC', 'Richoux-\nLSTM', 'DeepFE', 'PIPR',
                        'RF-PCA', 'RF-MDS', 'SVM-MDS', 'RF-\nnode2vec', 'SVM-\nnode2vec')
)


# pheatmap(t(result_mat[, 2:7]),
#          #annotation_row = annotation_col[annotation_col$Test == 'Original', 'Dataset', drop=FALSE],
#          annotation_colors = list(Dataset = c('Huang'='#E69F00','Guo'='#56B4E9', 'Du'='#009E73',
#                               'Pan'='#F0E442','Richoux-Regular'='#0072B2','Richoux-Strict'='#CC79A7')),
#          cluster_rows = FALSE,
#          cluster_cols = FALSE,
#          gaps_col = 5,
#          display_numbers = TRUE,
#          legend = FALSE,
#          filename = 'plots/heatmap_results_original_quer.pdf',
#          width=6,
#          height=5.2,
#          fontsize = 11,
#          labels_row = c(
#            #paste0('Gold (', original_sizes['gold'], ')'),
#            paste0('HUANG\n(', original_sizes['huang'], ')'),
#                         paste0('GUO\n(', original_sizes['guo'], ')'),
#                         paste0('DU\n(', original_sizes['du'], ')'),
#                         paste0('PAN\n(', original_sizes['pan'], ')'),
#                         paste0('RICHOUX-\nREGULAR\n(', original_sizes['richoux-regular'], ')'),
#                         paste0('RICHOUX-\nSTRICT\n(', original_sizes['richoux-strict'], ')')
#          ),
#          labels_col = c('SPRINT',
#                         'Richoux-\nFC',
#                         'Richoux-\nLSTM',
#                         'DeepFE',
#                         'PIPR',
#                         'RF-PCA',
#                         'SVM-PCA',
#                         'RF-MDS',
#                         'SVM-MDS',
#                         'RF-\nnode2vec',
#                         'SVM-\nnode2vec')
# )
# 
# pheatmap(t(result_mat[, 8:13]),
#          #annotation_row = annotation_col[annotation_col$Test == 'Rewired', 'Dataset', drop=FALSE],
#          annotation_colors = list(Dataset = c('Huang'='#E69F00','Guo'='#56B4E9', 'Du'='#009E73',
#                                               'Pan'='#F0E442','Richoux-Regular'='#0072B2','Richoux-Strict'='#CC79A7')),
#          cluster_rows = FALSE,
#          cluster_cols = FALSE,
#          legend = FALSE,
#          gaps_col = 5,
#          display_numbers = TRUE,
#          filename = 'plots/heatmap_results_rewired_quer.pdf',
#          width=6,
#          height=4,
#          labels_row = c(paste0('HUANG\n(', rewired_sizes['huang'], ')'),
#                         paste0('GUO\n(', rewired_sizes['guo'], ')'),
#                         paste0('DU\n(', rewired_sizes['du'], ')'),
#                         paste0('PAN\n(', rewired_sizes['pan'], ')'),
#                         paste0('RICHOUX-REGULAR\n(', rewired_sizes['richoux-regular'], ')'),
#                         paste0('RICHOUX-STRICT\n(', rewired_sizes['richoux-strict'], ')')
#          ),
#          labels_col = c('SPRINT', 'Richoux-\nFC', 'Richoux-\nLSTM', 'DeepFE', 'PIPR',
#                         'RF-PCA', 'RF-MDS', 'SVM-MDS', 'RF-\nnode2vec', 'SVM-\nnode2vec')
# )
# 
# pheatmap(result_mat[, 14:ncol(result_mat)],
#          #annotation_col = annotation_col[!annotation_col$Test %in% c('Original', 'Rewired'), "Dataset"],
#          #annotation_colors = list(Dataset = c('HUANG'='#E69F00','GUO'='#56B4E9', 'DU'='#009E73',
#         #                                      'PAN'='#F0E442','RICHOUX'='#0072B2'),
#         #                          Test = c('Inter->Intra-1'='#888888', 'Inter->Intra-0'='#44AA99', 'Intra-0->Intra-1'='#661100')),
#         main = TeX('Train on $\\it{INTER}$, test on $\\it{INTRA}_1$        Train on $\\it{INTER}$, test on $\\it{INTRA}_0$        Train on $\\it{INTRA}_0$, test on $\\it{INTRA}_1$'),
#          cluster_rows = FALSE,
#          cluster_cols = FALSE,
#          legend = FALSE,
#          gaps_row = 5,
#          gaps_col = c(5, 10, 15),
#          display_numbers = TRUE,
#         fontsize = 10,
#         fontsize_number = 11,
#         fontsize_row = 11,
#         fontsize_col = 11,
#          filename = 'plots/heatmap_results_partitions.pdf',
#          width=10,
#          height=8,
#          labels_col = c(#partition both ->1
#            paste0('HUANG\n(', partition_sizes['huang both'], ')'),
#            paste0('GUO\n(', partition_sizes['guo both'], ')'),
#            paste0('DU\n(', partition_sizes['du both'], ')'),
#            paste0('PAN\n(', partition_sizes['pan both'], ')'),
#            paste0('RICHOUX\n(', partition_sizes['richoux both'], ')'),
#            #partition both -> 0
#            paste0('HUANG\n(', partition_sizes['huang both'], ')'),
#            paste0('GUO\n(', partition_sizes['guo both'], ')'),
#            paste0('DU\n(', partition_sizes['du both'], ')'),
#            paste0('PAN\n(', partition_sizes['pan both'], ')'),
#            paste0('RICHOUX\n(', partition_sizes['richoux both'], ')'),
#            #partition 0 -> 1
#            paste0('HUANG\n(', partition_sizes['huang 0'], ')'),
#            paste0('GUO\n(', partition_sizes['guo 0'], ')'),
#            paste0('DU\n(', partition_sizes['du 0'], ')'),
#            paste0('PAN\n(', partition_sizes['pan 0'], ')'),
#            paste0('RICHOUX\n(', partition_sizes['richoux 0'], ')')
#          ),
#          labels_row = c('SPRINT', 'Richoux-\nFC', 'Richoux-\nLSTM', 'DeepFE', 'PIPR',
#                         'RF-PCA', 'RF-MDS', 'SVM-MDS', 'RF-\nnode2vec', 'SVM-\nnode2vec')
# )
