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
                                                      'DeepFE', 'PIPR', 'D-SCRIPT', 'Topsy_Turvy', 
                                                      'RF_PCA','SVM_PCA', 'RF_MDS', 'SVM_MDS',
                                                      'RF_node2vec',  'SVM_node2vec', 'degree_hf', 'degree_cons'))]
all_results <- all_results[, Dataset := factor(Dataset, 
                                               levels = c('gold_standard', 'huang', 'guo', 'du', 'pan', 'richoux-regular', 'richoux-strict', 'dscript'))]

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
all_results[, Model := gsub('degree_hf', 'Harmonic Function', Model)]
all_results[, Model := gsub('degree_cons', 'Global and Local Consistency', Model)]
all_results[, Model := gsub('Topsy_Turvy', 'Topsy Turvy', Model)]
all_results <- all_results[, Model := factor(Model, 
                                             levels=c('SPRINT', 
                                                      'Richoux-FC', 'Richoux-LSTM',  
                                                      'DeepFE', 'PIPR', 'D-SCRIPT', 'Topsy Turvy', 'RF PCA','SVM PCA', 'RF MDS', 'SVM MDS',
                                                      'RF node2vec',  'SVM node2vec', 'Harmonic Function', 'Global and Local Consistency'))]

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
                              levels = c('Original', 'Rewired', 'Inter->Intra-0', 'Inter->Intra-1', 'Intra-0->Intra-1'))
annotation_col$Dataset <- tstrsplit(colnames(result_mat), '_', keep = 1)[[1]]
annotation_col$Dataset <- factor(annotation_col$Dataset, 
                                 levels = c('Gold', 'huang', 'guo', 'du', 'pan', 'richoux-regular', 'richoux-strict', 'richoux', 'dscript'))
annotation_col <- annotation_col[order(annotation_col$Test, annotation_col$Dataset), , drop = FALSE]
annotation_col$Dataset <- NULL
result_mat <- result_mat[, rownames(annotation_col)]

# training data sizes
get_sizes <- function(directory) {
  sprint_data_dir <- paste0('../algorithms/SPRINT/data/', directory, '/')
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
  names(train_sizes) <- tstrsplit(training_files_pos, '_', keep=1)[[1]]
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

get_sizes_dscript <- function(directory) {
  dscript_data_dir <- paste0('../algorithms/D-SCRIPT-main/data/', directory, '/')
  training_files <- list.files(path=dscript_data_dir, pattern = 'train')
  train_sizes <- sapply(paste0(dscript_data_dir, training_files), function(x){
    as.integer(system2("wc",
                       args = c("-l",
                                x,
                                " | awk '{print $1}'"),
                       stdout = TRUE)) 
  }
  )
  training_files[grepl('richoux', training_files, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', training_files[grepl('richoux', training_files, fixed=TRUE)])
  names(train_sizes) <- tstrsplit(training_files, '_', keep=1)[[1]]
  train_sizes <- prettyNum(train_sizes, big.mark = ',')
  return(train_sizes)
}
original_dscript_sizes <- get_sizes_dscript('original')
gold_size <- prettyNum(as.integer(system2("wc",
                                 args = c("-l",
                                          '../algorithms/D-SCRIPT-main/data/gold/Intra1.txt',
                                          " | awk '{print $1}'"),
                                 stdout = TRUE)), big.mark = ',')
names(gold_size) <- 'gold'
original_dscript_sizes <- c(original_dscript_sizes, gold_size)
rewired_dscript_sizes <- get_sizes_dscript('rewired')
dscript_data_dir <- '../algorithms/D-SCRIPT-main/data/partitions/'
training_files <- list.files(path=dscript_data_dir)
partition_dscript_sizes <- sapply(paste0(dscript_data_dir, training_files), function(x){
  as.integer(system2("wc",
                     args = c("-l",
                              x,
                              " | awk '{print $1}'"),
                     stdout = TRUE)) 
}
)
filenames <- tstrsplit(tstrsplit(training_files, '.txt', keep=1)[[1]], '_', keep=c(1,3))
names(partition_dscript_sizes) <- paste(filenames[[1]], filenames[[2]])
partition_dscript_sizes <- prettyNum(partition_dscript_sizes, big.mark = ',')

breaksList <- c(min(result_mat, na.rm = T), seq(0.5, 1.0, 0.005))

compute_number_colors <- function(mat) {
  breaks <- c(min(mat, na.rm = T), seq(0.5, 1.0, 0.0051))
  color <- colorRampPalette(rev(brewer.pal(n = 9, name =
                                             "RdYlBu")))(100)
  rgb_colors <- col2rgb(pheatmap:::scale_colours(as.matrix(mat), col = color, breaks = breaks, na_col = "#DDDDDD"))
  luminance <- rgb_colors * c(0.299, 0.587, 0.114)
  luminance <- luminance['red', ]+luminance['green', ] + luminance['blue', ]
  number_color <- ifelse(luminance < 125, "grey90", "grey30")
  return(number_color)
}

number_color <- compute_number_colors(t(result_mat))

pheatmap(t(result_mat),
         annotation_row = annotation_col,
         annotation_colors = list(
                                  Test = c('Original'='#AA4499', 'Rewired'='#DDCC77', 'Inter->Intra-0'='#44AA99', 'Inter->Intra-1'='#888888', 'Intra-0->Intra-1'='#661100')
                                  ),
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         gaps_col = 7,
         gaps_row = c(8,15,21,27),
         display_numbers = TRUE,
         number_color = number_color,
         legend = FALSE,
         filename = paste0('plots/heatmap_results_', measure, '.pdf'),
         width=10,
         height=10,
         cex = 1,
         breaks = breaksList,
         color = colorRampPalette(rev(brewer.pal(n = 9, name =
                                                   "RdYlBu")))(100),
         labels_row = c(
           paste0('GOLD STANDARD (', original_sizes['gold'], '/', original_dscript_sizes['gold'], ')'),
           paste0('HUANG (', original_sizes['huang'], '/', original_dscript_sizes['huang'], ')'),
           paste0('GUO (', original_sizes['guo'], '/', original_dscript_sizes['guo'], ')'),
           paste0('DU (', original_sizes['du'], '/', original_dscript_sizes['du'], ')'),
           paste0('PAN (', original_sizes['pan'], '/', original_dscript_sizes['pan'], ')'),
           paste0('RICHOUX-REGULAR (', original_sizes['richoux-regular'], '/', original_dscript_sizes['richoux-regular'], ')'),
           paste0('RICHOUX-STRICT (', original_sizes['richoux-strict'], '/', original_dscript_sizes['richoux-strict'], ')'),
           paste0('D-SCRIPT UNBALANCED (', original_sizes['dscript'], '/', original_dscript_sizes['dscript'], ')'),
           #rewired
           paste0('HUANG (', rewired_sizes['huang'], '/', rewired_dscript_sizes['huang'], ')'),
           paste0('GUO (', rewired_sizes['guo'], '/', rewired_dscript_sizes['guo'], ')'),
           paste0('DU (', rewired_sizes['du'], '/', rewired_dscript_sizes['du'], ')'),
           paste0('PAN (', rewired_sizes['pan'], '/', rewired_dscript_sizes['pan'], ')'),
           paste0('RICHOUX-REGULAR (', rewired_sizes['richoux-regular'], '/', rewired_dscript_sizes['richoux-regular'], ')'),
           paste0('RICHOUX-STRICT (', rewired_sizes['richoux-strict'], '/', rewired_dscript_sizes['richoux-strict'], ')'),
           paste0('D-SCRIPT UNBALANCED (', rewired_sizes['dscript'], '/', rewired_dscript_sizes['dscript'], ')'),
           #partition both -> 0
           paste0('HUANG (', partition_sizes['huang both'], '/', partition_dscript_sizes['huang both'], ')'),
           paste0('GUO (', partition_sizes['guo both'], '/', partition_dscript_sizes['guo both'], ')'),
           paste0('DU (', partition_sizes['du both'], '/', partition_dscript_sizes['du both'], ')'),
           paste0('PAN (', partition_sizes['pan both'], '/', partition_dscript_sizes['pan both'], ')'),
           paste0('RICHOUX-UNIPROT (', partition_sizes['richoux both'], '/', partition_dscript_sizes['richoux both'], ')'),
           paste0('D-SCRIPT UNBALANCED (', partition_sizes['dscript both'], '/', partition_dscript_sizes['dscript both'], ')'),
           #partition both ->1
           paste0('HUANG (', partition_sizes['huang both'], '/', partition_dscript_sizes['huang both'], ')'),
           paste0('GUO (', partition_sizes['guo both'], '/', partition_dscript_sizes['guo both'], ')'),
           paste0('DU (', partition_sizes['du both'], '/', partition_dscript_sizes['du both'], ')'),
           paste0('PAN (', partition_sizes['pan both'], '/', partition_dscript_sizes['pan both'], ')'),
           paste0('RICHOUX-UNIPROT (', partition_sizes['richoux both'], '/', partition_dscript_sizes['richoux both'], ')'),
           paste0('D-SCRIPT UNBALANCED (', partition_sizes['dscript both'], '/', partition_dscript_sizes['dscript both'], ')'),
           #partition 0 -> 1
           paste0('HUANG (', partition_sizes['huang 0'], '/', partition_dscript_sizes['huang 0'], ')'),
           paste0('GUO (', partition_sizes['guo 0'], '/', partition_dscript_sizes['guo 0'], ')'),
           paste0('DU (', partition_sizes['du 0'], '/', partition_dscript_sizes['du 0'], ')'),
           paste0('PAN (', partition_sizes['pan 0'], '/', partition_dscript_sizes['pan 0'], ')'),
           paste0('RICHOUX-UNIPROT (', partition_sizes['richoux 0'], '/', partition_dscript_sizes['richoux 0'], ')'),
           paste0('D-SCRIPT UNBALANCED (', partition_sizes['dscript 0'], '/', partition_dscript_sizes['dscript 0'], ')')
         ),
         labels_col = c('SPRINT\n(AUPR)', 'Richoux-\nFC', 'Richoux-\nLSTM', 'DeepFE', 'PIPR', 'D-SCRIPT', 'Topsy Turvy',
                        'RF-PCA', 'SVM-PCA', 'RF-MDS', 'SVM-MDS', 'RF-\nnode2vec', 'SVM-\nnode2vec', 
                        'Harmonic\nFunction', 'Global and\nLocal Cons.')
)

# library(ComplexHeatmap)
# library(circlize)
# svg('~/Downloads/heatmap_wide.svg', width=21, height=8.5)
# breaks <- c(min(result_mat, na.rm = T), seq(0.5, 1.0, 0.0051))
# Heatmap(result_mat,
#         row_split = factor(c(rep('Published Methods', 7), rep('Baseline ML', nrow(result_mat)-7)), levels = c('Published Methods', 'Baseline ML')),
#         column_split = factor(c(rep('Train: Original, Test: Original, 80/20', 8), rep('Train: Rewired, Test: Original, 80/20', 7), rep('Train: INTER, Test: INTRA-0', 6), rep('Train: INTER, Test: INTRA-1', 6), rep('Train: INTRA-0, Test: INTRA-1', 6)), levels = c('Train: Original, Test: Original, 80/20', 'Train: Rewired, Test: Original, 80/20', 'Train: INTER, Test: INTRA-0', 'Train: INTER, Test: INTRA-1', 'Train: INTRA-0, Test: INTRA-1')),
#         cluster_rows = FALSE,
#         cluster_columns = FALSE,
#         cluster_row_slices = FALSE,
#         cluster_column_slices = FALSE,
#         layer_fun = function(j, i, x, y, w, h, fill) {
#           grid.text(sprintf('%.2f', pindex(result_mat, i, j)),
#                     x = x, y = y, gp = gpar(col =
#                                               ifelse(
#                                                 (col2rgb(fill) * c(0.299, 0.587, 0.114))['red', ] + (col2rgb(fill) * c(0.299, 0.587, 0.114))['green', ] + (col2rgb(fill) * c(0.299, 0.587, 0.114))['blue', ] < 125,
#                                                 "grey90", "grey30"),
#                                             fontsize = 12))
#         },
#         show_heatmap_legend = FALSE,
#         col = colorRamp2(breaks, colorRampPalette(rev(brewer.pal(n = 9, name =
#                                                                        "RdYlBu")))(100)),
#         column_labels = c(
#            paste0('GOLD (', original_sizes['gold'], ')'),
#            paste0('HUANG(', original_sizes['huang'], ')'),
#            paste0('GUO(', original_sizes['guo'], ')'),
#            paste0('DU (', original_sizes['du'], ')'),
#            paste0('PAN (', original_sizes['pan'], ')'),
#            paste0('RICHOUX-REG.\n(', original_sizes['richoux-regular'], ')'),
#            paste0('RICHOUX-STR.\n(', original_sizes['richoux-strict'], ')'),
#            paste0('D-SCR. UNBAL. \n(', original_sizes['dscript'], ')'),
#            #rewired
#            paste0('HUANG (', rewired_sizes['huang'], ')'),
#            paste0('GUO (', rewired_sizes['guo'], ')'),
#            paste0('DU (', rewired_sizes['du'], ')'),
#            paste0('PAN (', rewired_sizes['pan'], ')'),
#            paste0('RICHOUX-REG.\n(', rewired_sizes['richoux-regular'], ')'),
#            paste0('RICHOUX-STR.\n(', rewired_sizes['richoux-strict'], ')'),
#            paste0('D-SCR. UNBAL.\n(', rewired_sizes['dscript'], ')'),
#            #partition both -> 0
#            paste0('HUANG (', partition_sizes['huang both'], ')'),
#            paste0('GUO (', partition_sizes['guo both'], ')'),
#            paste0('DU (', partition_sizes['du both'], ')'),
#            paste0('PAN (', partition_sizes['pan both'], ')'),
#            paste0('RICHOUX-UNI.\n(', partition_sizes['richoux both'], ')'),
#            paste0('D-SCR. UNBAL.\n(', partition_sizes['dscript both'], ')'),
#            #partition both ->1
#            paste0('HUANG (', partition_sizes['huang both'], ')'),
#            paste0('GUO (', partition_sizes['guo both'], ')'),
#            paste0('DU (', partition_sizes['du both'], ')'),
#            paste0('PAN (', partition_sizes['pan both'], ')'),
#            paste0('RICHOUX-UNI.\n(', partition_sizes['richoux both'], ')'),
#            paste0('D-SCR. UNBAL.\n(', partition_sizes['dscript both'], ')'),
#            #partition 0 -> 1
#            paste0('HUANG (', partition_sizes['huang 0'], ')'),
#            paste0('GUO (', partition_sizes['guo 0'], ')'),
#            paste0('DU (', partition_sizes['du 0'], ')'),
#            paste0('PAN (', partition_sizes['pan 0'], ')'),
#            paste0('RICHOUX-UNI.\n(', partition_sizes['richoux 0'], ')'),
#            paste0('D-SCR. UNBAL.\n(', partition_sizes['dscript 0'], ')')
#          ),
#         row_labels = c('SPRINT\n(AUPR)', 'Richoux-\nFC', 'Richoux-\nLSTM', 'DeepFE', 'PIPR', 'D-SCRIPT', 'Topsy Turvy',
#                         'RF-PCA', 'SVM-PCA', 'RF-MDS', 'SVM-MDS', 'RF-\nnode2vec', 'SVM-\nnode2vec',
#                         'Harmonic\nFunction', 'Global and\nLocal Cons.')
# )
# dev.off()
# number_color <- compute_number_colors(t(result_mat[, 2:8]))
# 
# pheatmap(t(result_mat[, 2:8]),
#          #annotation_row = annotation_col[annotation_col$Test == 'Original', 'Dataset', drop=FALSE],
#          annotation_colors = list(Dataset = c('Huang'='#E69F00','Guo'='#56B4E9', 'Du'='#009E73',
#                               'Pan'='#F0E442','Richoux-Regular'='#0072B2','Richoux-Strict'='#CC79A7')),
#          cluster_rows = FALSE,
#          cluster_cols = FALSE,
#          gaps_col = 7,
#          display_numbers = TRUE,
#          number_color = number_color,
#          legend = FALSE,
#          filename = 'plots/heatmap_results_original_quer.pdf',
#          width=8,
#          height=6.8,
#          fontsize = 11,
#          breaks = breaksList,
#          color = colorRampPalette(rev(brewer.pal(n = 9, name =
#                                                    "RdYlBu")))(100),
#          labels_row = c(
#            #paste0('Gold (', original_sizes['gold'], ')'),
#            paste0('HUANG\n(', original_sizes['huang'], '/', original_dscript_sizes['huang'], ')'),
#                         paste0('GUO\n(', original_sizes['guo'], '/', original_dscript_sizes['guo'], ')'),
#                         paste0('DU\n(', original_sizes['du'], '/', original_dscript_sizes['du'], ')'),
#                         paste0('PAN\n(', original_sizes['pan'], '/', original_dscript_sizes['pan'], ')'),
#                         paste0('RICHOUX-\nREGULAR\n(', original_sizes['richoux-regular'], '/', original_dscript_sizes['richoux-regular'], ')'),
#                         paste0('RICHOUX-\nSTRICT\n(', original_sizes['richoux-strict'], '/', original_dscript_sizes['richoux-strict'], ')'),
#            paste0('D-SCRIPT\nUNBALANCED\n(', original_sizes['dscript'], '/', original_dscript_sizes['dscript'], ')')
#          ),
#          labels_col = c('SPRINT\n(AUPR)',
#                         'Richoux-\nFC',
#                         'Richoux-\nLSTM',
#                         'DeepFE',
#                         'PIPR',
#                         'D-SCRIPT',
#                         'Topsy-Turvy',
#                         'RF-PCA',
#                         'SVM-PCA',
#                         'RF-MDS',
#                         'SVM-MDS',
#                         'RF-\nnode2vec',
#                         'SVM-\nnode2vec',
#                         'Harmonic\nFunction',
#                         'Global and\nLocal Cons.')
# )
# 
# number_color <- compute_number_colors(t(result_mat[, 9:15]))
# pheatmap(t(result_mat[, 9:15]),
#          #annotation_row = annotation_col[annotation_col$Test == 'Rewired', 'Dataset', drop=FALSE],
#          annotation_colors = list(Dataset = c('Huang'='#E69F00','Guo'='#56B4E9', 'Du'='#009E73',
#                                               'Pan'='#F0E442','Richoux-Regular'='#0072B2','Richoux-Strict'='#CC79A7')),
#          cluster_rows = FALSE,
#          cluster_cols = FALSE,
#          legend = FALSE,
#          gaps_col = 7,
#          display_numbers = TRUE,
#          number_color = number_color,
#          breaks = breaksList,
#          color = colorRampPalette(rev(brewer.pal(n = 9, name =
#                                                    "RdYlBu")))(100),
#          filename = 'plots/heatmap_results_rewired_quer.pdf',
#          width=7,
#          height=5,
#          labels_row = c(paste0('HUANG\n(', rewired_sizes['huang'], ')'),
#                         paste0('GUO\n(', rewired_sizes['guo'], ')'),
#                         paste0('DU\n(', rewired_sizes['du'], ')'),
#                         paste0('PAN\n(', rewired_sizes['pan'], ')'),
#                         paste0('RICHOUX-REGULAR\n(', rewired_sizes['richoux-regular'], ')'),
#                         paste0('RICHOUX-STRICT\n(', rewired_sizes['richoux-strict'], ')'),
#                         paste0('D-SCRIPT\nUNBALANCED\n(', rewired_sizes['dscript'], ')')
#          ),
#          labels_col = c('SPRINT\n(AUPR)',
#                         'Richoux-\nFC',
#                         'Richoux-\nLSTM',
#                         'DeepFE',
#                         'PIPR',
#                         'D-SCRIPT',
#                         'Topsy Turvy',
#                         'RF-PCA',
#                         'SVM-PCA',
#                         'RF-MDS',
#                         'SVM-MDS',
#                         'RF-\nnode2vec',
#                         'SVM-\nnode2vec',
#                         'Harmonic\nFunction',
#                         'Global and\nLocal Cons.')
# )
# 
# number_color <- compute_number_colors(result_mat[, 16:ncol(result_mat)])
# pheatmap(result_mat[, 16:ncol(result_mat)],
#          #annotation_col = annotation_col[!annotation_col$Test %in% c('Original', 'Rewired'), "Dataset"],
#          #annotation_colors = list(Dataset = c('HUANG'='#E69F00','GUO'='#56B4E9', 'DU'='#009E73',
#         #                                      'PAN'='#F0E442','RICHOUX'='#0072B2'),
#         #                          Test = c('Inter->Intra-1'='#888888', 'Inter->Intra-0'='#44AA99', 'Intra-0->Intra-1'='#661100')),
#         main = TeX('Train on $\\it{INTER}$, test on $\\it{INTRA}_0$                    Train on $\\it{INTER}$, test on $\\it{INTRA}_1$                    Train on $\\it{INTRA}_0$, test on $\\it{INTRA}_1$'),
#         cluster_rows = FALSE,
#         cluster_cols = FALSE,
#         legend = FALSE,
#         gaps_row = 7,
#         gaps_col = c(6, 12),
#         display_numbers = TRUE,
#         number_color = number_color,
#         breaks = breaksList,
#         color = colorRampPalette(rev(brewer.pal(n = 9, name =
#                                                   "RdYlBu")))(100),
#         fontsize = 10,
#         fontsize_number = 11,
#         fontsize_row = 11,
#         fontsize_col = 11,
#         filename = 'plots/heatmap_results_partitions.pdf',
#         width=12,
#         height=10,
#         labels_col = c(
#            #partition both -> 0
#            paste0('HUANG\n(', partition_sizes['huang both'], ')'),
#            paste0('GUO\n(', partition_sizes['guo both'], ')'),
#            paste0('DU\n(', partition_sizes['du both'], ')'),
#            paste0('PAN\n(', partition_sizes['pan both'], ')'),
#            paste0('RICHOUX-\nUNIPROT\n(', partition_sizes['richoux both'], ')'),
#            paste0('D-SCRIPT\nUNBALANCED\n(', partition_sizes['dscript both'], ')'),
#            #partition both ->1
#            paste0('HUANG\n(', partition_sizes['huang both'], ')'),
#            paste0('GUO\n(', partition_sizes['guo both'], ')'),
#            paste0('DU\n(', partition_sizes['du both'], ')'),
#            paste0('PAN\n(', partition_sizes['pan both'], ')'),
#            paste0('RICHOUX-\nUNIPROT\n(', partition_sizes['richoux both'], ')'),
#            paste0('D-SCRIPT\nUNBALANCED\n(', partition_sizes['dscript both'], ')'),
#            #partition 0 -> 1
#            paste0('HUANG\n(', partition_sizes['huang 0'], ')'),
#            paste0('GUO\n(', partition_sizes['guo 0'], ')'),
#            paste0('DU\n(', partition_sizes['du 0'], ')'),
#            paste0('PAN\n(', partition_sizes['pan 0'], ')'),
#            paste0('RICHOUX-\nUNIPROT\n(', partition_sizes['richoux 0'], ')'),
#            paste0('D-SCRIPT\nUNBALANCED\n(', partition_sizes['dscript 0'], ')')
#          ),
#          labels_row = c('SPRINT\n(AUPR)',
#                         'Richoux-\nFC',
#                         'Richoux-\nLSTM',
#                         'DeepFE',
#                         'PIPR',
#                         'D-SCRIPT',
#                         'Topsy Turvy',
#                         'RF-PCA',
#                         'SVM-PCA',
#                         'RF-MDS',
#                         'SVM-MDS',
#                         'RF-\nnode2vec',
#                         'SVM-\nnode2vec',
#                         'Harmonic\nFunction',
#                         'Global and\nLocal Cons.')
# )
# 
