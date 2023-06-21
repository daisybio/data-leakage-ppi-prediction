library(data.table)
library(ggplot2)
library(pheatmap)
library(RColorBrewer)

#### get sizes of datasets
get_all_sizes <- function(directory) {
  sprint_data_dir <- paste0('../algorithms/SPRINT/data/', directory, '/')
  
  train_files_pos <- list.files(path=sprint_data_dir, pattern = 'train_pos')
  train_files_neg <- list.files(path=sprint_data_dir, pattern = 'train_neg')
  test_files_pos <- list.files(path=sprint_data_dir, pattern = 'test_pos')
  test_files_neg <- list.files(path=sprint_data_dir, pattern = 'test_neg')
  
  train_sizes_pos <- get_sizes(sprint_data_dir, train_files_pos)
  train_sizes_neg <- get_sizes(sprint_data_dir, train_files_neg)
  test_sizes_pos <- get_sizes(sprint_data_dir, test_files_pos)
  test_sizes_neg <- get_sizes(sprint_data_dir, test_files_neg)
  
  sizes <- train_sizes_pos + train_sizes_neg + test_sizes_pos + test_sizes_neg
  train_files_pos[grepl('richoux', train_files_pos, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', train_files_pos[grepl('richoux', train_files_pos, fixed=TRUE)])
  names(sizes) <- tstrsplit(train_files_pos, '_', keep=1)[[1]]
  sizes <- sizes[order(sizes)]
  sizes <- prettyNum(sizes, big.mark = ',')
  return(sizes)
}

get_sizes_dscript <- function(directory) {
  dscript_data_dir <- paste0('../algorithms/D-SCRIPT-main/data/', directory, '/')
  training_files <- list.files(path=dscript_data_dir, pattern = 'train')
  test_files <- list.files(path=dscript_data_dir, pattern = 'test')
  train_sizes <- get_sizes(dscript_data_dir, training_files)
  test_sizes <- get_sizes(dscript_data_dir, test_files)

  sizes <- train_sizes + test_sizes
  training_files[grepl('richoux', training_files, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', training_files[grepl('richoux', training_files, fixed=TRUE)])
  names(sizes) <- tstrsplit(training_files, '_', keep=1)[[1]]
  sizes <- sizes[order(sizes)]
  sizes <- prettyNum(sizes, big.mark = ',')
  return(sizes)
}

get_sizes <- function(sprint_data_dir, file_list){
  sizes <- sapply(paste0(sprint_data_dir, file_list), function(x){
    as.integer(system2("wc",
                       args = c("-l",
                                x,
                                " | awk '{print $1}'"),
                       stdout = TRUE))
  }
  )
  return(sizes)
}

get_all_sizes('original')
get_sizes_dscript('original')

#### visualize degree ratios
test <- 'partition'
degree_ratios <- fread(paste0('../network_data/', test, '_degree_ratios.csv'))
if(test == 'original' | test == 'rewired'){
  degree_ratios[dataset == 'richoux_regular', dataset := 'RICHOUX-REGULAR']
  degree_ratios[dataset == 'richoux_strict', dataset := 'RICHOUX-STRICT']
}else{
  degree_ratios[, c("dataset", "partition") := tstrsplit(dataset, '_')]
  degree_ratios[dataset == 'richoux', dataset := 'RICHOUX-UNIPROT']
  degree_ratios[partition == 'both', partition := 'INTER']
  degree_ratios[partition == '0', partition := 'INTRA-0']
  degree_ratios[partition == '1', partition := 'INTRA-1']
}
degree_ratios[dataset == 'dscript', dataset := 'D-SCRIPT UNBALANCED']
if(test == 'original'){
  degree_ratios[dataset == 'gold_standard', dataset := 'GOLD-STANDARD']
  degree_ratios[, dataset := toupper(dataset)]
  degree_ratios[, dataset := factor(dataset, levels = c('HUANG', 'GUO', 'DU', 
                                                        'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBALANCED', 'GOLD-STANDARD'))]
}else{
  degree_ratios[, dataset := toupper(dataset)]
  if(test == 'rewired'){
    degree_ratios[, dataset := factor(dataset, levels = c('HUANG', 'GUO', 'DU', 
                                                        'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBALANCED'))]
  }else{
    degree_ratios[, dataset := factor(dataset, levels = c('HUANG', 'GUO', 'DU', 
                                                          'PAN', 'RICHOUX-UNIPROT', 'D-SCRIPT UNBALANCED'))]
  }
}
colorBlindBlack8  <- c('#000000', '#E69F00', '#56B4E9', '#009E73', 
                                '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#661100')

if(test == 'original' | test == 'rewired'){
  ggplot(degree_ratios, aes(degree_ratio, fill=dataset))+
    geom_histogram(binwidth = 0.05)+
    facet_wrap(~dataset, scales='free_y', ncol = 2)+
    scale_fill_manual(values = colorBlindBlack8)+
    xlab('Degree Ratio p')+
    theme_bw()+
    theme(legend.position = "none")
  ggsave(paste0('plots/degree_ratios_', test, '.pdf'), width=6, height=6)
}else{
  ggplot(degree_ratios, aes(degree_ratio, fill=dataset))+
    geom_histogram(binwidth = 0.05)+
    facet_wrap(partition~dataset, scales='free_y', nrow=3)+
    scale_fill_manual(values = colorBlindBlack8)+
    xlab('Degree Ratio p')+
    theme_bw()+
    theme(legend.position = "none")
  ggsave(paste0('plots/degree_ratios_', test, '.pdf'), width=12, height=4.5)
}


get_proportions <- function(name, test, degree_ratios, partition_train='INTRA-0', partition_test='1') {
  if(name == 'gold_standard' & test == 'original'){
    all_interactions <- fread('../algorithms/D-SCRIPT-main/data/gold/Intra1.txt')
  }else if(test == 'partition'){
    all_interactions <- fread(paste0('../algorithms/D-SCRIPT-main/data/partitions/', name, '_partition_', partition_test, '.txt'))
  }else{
    all_interactions <- fread(paste0('../algorithms/D-SCRIPT-main/data/', test, '/', name, '_test.txt'))
  }
  name <- toupper(name)
  name <- gsub('_', '-', name)
  if(name == 'DSCRIPT'){
    name <- 'D-SCRIPT UNBALANCED'
  }else if(name == 'RICHOUX'){
    name <- 'RICHOUX-UNIPROT'
  }
  if(test == 'partition'){
    deg_close_to_1 <- degree_ratios[dataset == name & partition == partition_train & degree_ratio >= 0.9, protein]
    deg_close_to_0 <- degree_ratios[dataset == name & partition == partition_train & degree_ratio <= 0.1, protein]
  }else{
    deg_close_to_1 <- degree_ratios[dataset == name & degree_ratio >= 0.9, protein]
    deg_close_to_0 <- degree_ratios[dataset == name & degree_ratio <= 0.1, protein]
  }
  all_interactions[, involved_pos := ifelse(V1 %in% deg_close_to_1 | V2 %in% deg_close_to_1, 1, 0)]
  all_interactions[, involved_neg := ifelse(V1 %in% deg_close_to_0 | V2 %in% deg_close_to_0, 1, 0)]
  proportion_high_deg_ratio_pos <- sum(all_interactions[V3 == 1, involved_pos])/nrow(all_interactions[V3 == 1])
  proportion_low_deg_ratio_pos <- sum(all_interactions[V3 == 1, involved_neg])/nrow(all_interactions[V3 == 1])
  proportion_high_deg_ratio_neg <- sum(all_interactions[V3 == 0, involved_pos])/nrow(all_interactions[V3 == 0])
  proportion_low_deg_ratio_neg <- sum(all_interactions[V3 == 0, involved_neg])/nrow(all_interactions[V3 == 0])
  if(test == 'partition'){
    part_test <- ifelse(partition_test == '0', 'INTRA-0', 'INTRA-1')
    return(data.table('%high p of pos.' = proportion_high_deg_ratio_pos,
                      '%low p of pos.' = proportion_low_deg_ratio_pos, 
                      '%high p of neg' = proportion_high_deg_ratio_neg, 
                      '%low p of neg' = proportion_low_deg_ratio_neg, 
                      name=name, 
                      partition_train=partition_train, 
                      partition_test=part_test))
  }else{
    return(data.table('%high p of pos.' = proportion_high_deg_ratio_pos,
                      '%low p of pos.' = proportion_low_deg_ratio_pos, 
                      '%high p of neg' = proportion_high_deg_ratio_neg, 
                      '%low p of neg' = proportion_low_deg_ratio_neg, 
                      name=name))
  }
}

if(test == 'partition'){
  df <- get_proportions('huang', 'partition', degree_ratios, 'INTER', '0')
  df <- rbind(df, get_proportions('huang', 'partition', degree_ratios, 'INTER', '1'))
  df <- rbind(df, get_proportions('huang', 'partition', degree_ratios, 'INTRA-0', '1'))
  
  df <- rbind(df, get_proportions('guo', 'partition', degree_ratios, 'INTER', '0'))
  df <- rbind(df, get_proportions('guo', 'partition', degree_ratios, 'INTER', '1'))
  df <- rbind(df, get_proportions('guo', 'partition', degree_ratios, 'INTRA-0', '1'))
  
  df <- rbind(df, get_proportions('du', 'partition', degree_ratios, 'INTER', '0'))
  df <- rbind(df, get_proportions('du', 'partition', degree_ratios, 'INTER', '1'))
  df <- rbind(df, get_proportions('du', 'partition', degree_ratios, 'INTRA-0', '1'))
  
  df <- rbind(df, get_proportions('pan', 'partition', degree_ratios, 'INTER', '0'))
  df <- rbind(df, get_proportions('pan', 'partition', degree_ratios, 'INTER', '1'))
  df <- rbind(df, get_proportions('pan', 'partition', degree_ratios, 'INTRA-0', '1'))
  
  df <- rbind(df, get_proportions('richoux', 'partition', degree_ratios, 'INTER', '0'))
  df <- rbind(df, get_proportions('richoux', 'partition', degree_ratios, 'INTER', '1'))
  df <- rbind(df, get_proportions('richoux', 'partition', degree_ratios, 'INTRA-0', '1'))
  
  df <- rbind(df, get_proportions('dscript', 'partition', degree_ratios, 'INTER', '0'))
  df <- rbind(df, get_proportions('dscript', 'partition', degree_ratios, 'INTER', '1'))
  df <- rbind(df, get_proportions('dscript', 'partition', degree_ratios, 'INTRA-0', '1'))
  
  df[, name := factor(name, levels = c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-UNIPROT', 'D-SCRIPT UNBALANCED'))]
  df[, setting := paste(partition_train, partition_test, sep='->')]
  df[, setting := factor(setting, levels = c('INTER->INTRA-0', 'INTER->INTRA-1', 'INTRA-0->INTRA-1'))]
  df <- df[order(setting, name)]
  mat <- as.matrix(df[, c(1:4)])
  rownames(mat) <- paste(df$name, df$setting, sep=': ')
}else{
  df <- get_proportions('huang', test, degree_ratios)
  df <- rbind(df, get_proportions('guo', test, degree_ratios))
  df <- rbind(df, get_proportions('du', test, degree_ratios))
  df <- rbind(df, get_proportions('pan', test, degree_ratios))
  df <- rbind(df, get_proportions('richoux_regular', test, degree_ratios))
  df <- rbind(df, get_proportions('richoux_strict', test, degree_ratios))
  df <- rbind(df, get_proportions('dscript', test, degree_ratios))
  if(test == 'original'){
    df <- rbind(df, get_proportions('gold_standard', test, degree_ratios))
    df[, name := factor(name, levels = c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBALANCED', 'GOLD-STANDARD'))]
  }else{
    df[, name := factor(name, levels = c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBALANCED'))]
  }
  df <- df[order(name)]
  mat <- as.matrix(df[, c(1:4)])
  rownames(mat) <- df$name
}

compute_number_colors <- function(mat) {
  breaks <- seq(0,1, 0.01)
  color <- colorRampPalette(rev(brewer.pal(n = 9, name =
                                             "RdYlBu")))(100)
  rgb_colors <- col2rgb(pheatmap:::scale_colours(as.matrix(mat), col = color, breaks = breaks, na_col = "#DDDDDD"))
  luminance <- rgb_colors * c(0.299, 0.587, 0.114)
  luminance <- luminance['red', ]+luminance['green', ] + luminance['blue', ]
  number_color <- ifelse(luminance < 125, "grey90", "grey30")
  return(number_color)
}

if(test == 'partition'){
  number_color <- matrix(compute_number_colors(mat), nrow=18)
  breaksList <- seq(0,1, 0.01)
  pheatmap(mat, 
           cluster_rows = FALSE,
           cluster_cols = FALSE,
           display_numbers = TRUE,
           number_color = number_color,
           color = colorRampPalette(rev(brewer.pal(n = 9, name =
                                                     "RdYlBu")))(100),
           breaks = breaksList,
           legend = FALSE,
           filename = paste0('plots/heatmap_degree_', test, '.pdf'),
           width=8,
           height=5
  )
}else{
  if(test == 'original'){
    number_color <- matrix(compute_number_colors(mat), nrow=8)
    number_color <- number_color[c(8, 1, 4, 3, 5, 7, 2, 6), ]
  }else{
    number_color <- matrix(compute_number_colors(mat), nrow=7)
    number_color <- number_color[c(1, 4, 3, 5, 7, 2, 6), ]
  }
  breaksList <- seq(0,1, 0.01)
  pheatmap(mat, 
           cluster_rows = TRUE,
           cluster_cols = FALSE,
           display_numbers = TRUE,
           number_color = number_color,
           color = colorRampPalette(rev(brewer.pal(n = 9, name =
                                                     "RdYlBu")))(100),
           breaks = breaksList,
           legend = FALSE,
           filename = paste0('plots/heatmap_degree_', test, '.pdf'),
           width=6,
           height=5
  )
}

# #### huang rewired scores investigation
# huang_pred_rew <- fread('../algorithms/D-SCRIPT-main/results_dscript/rewired/huang.txt.predictions.tsv')
# deg_close_to_1 <- degree_ratios[dataset == 'HUANG' & degree_ratio >= 0.9, protein]
# deg_close_to_0 <- degree_ratios[dataset == 'HUANG' & degree_ratio <= 0.1, protein]
# huang_pred_rew[, involved_pos := ifelse(V1 %in% deg_close_to_1 | V2 %in% deg_close_to_1, 1, 0)]
# huang_pred_rew[, involved_neg := ifelse(V1 %in% deg_close_to_0 | V2 %in% deg_close_to_0, 1, 0)]
# huang_pred_rew[, involved := ifelse(involved_pos == 1 & involved_neg == 0, 'involved_pos', 
#                                     ifelse(involved_neg == 1 & involved_pos == 0, 'involved_neg', NA))]
# huang_pred_rew[, involved := as.factor(involved)]
# huang_pred_rew <- merge(huang_pred_rew, degree_ratios[dataset == 'HUANG', c('protein', 'degree_ratio')], by.x = 'V1', by.y = 'protein')
# huang_pred_rew <- merge(huang_pred_rew, degree_ratios[dataset == 'HUANG', c('protein', 'degree_ratio')], by.x = 'V2', by.y = 'protein')
# colnames(huang_pred_rew) <- c('protein2', 'protein1', 'y_true', 'y_pred', 'involved_pos', 'involved_neg', 'involved', 'degree_ratio1', 'degree_ratio2')
# huang_pred_rew[, degree_ratio := ifelse((involved == 'involved_pos' & degree_ratio1 >= 0.9) | (involved == 'involved_neg' & degree_ratio1 <= 0.1), 
#                                         degree_ratio1, 
#                                     ifelse((involved == 'involved_pos' & degree_ratio2 >= 0.9)|(involved == 'involved_neg' & degree_ratio2 <= 0.1), 
#                                            degree_ratio2, NA))]
# huang_pred_rew[, deg_ratio_cutoff := as.factor(ifelse(degree_ratio > 0.5, 1, 0))]
# huang_pred_rew[, y_true := as.factor(y_true)]
# 
# ggplot(huang_pred_rew, aes(x = y_pred, color=y_true))+
#   geom_density()+
#   facet_wrap(~deg_ratio_cutoff)

