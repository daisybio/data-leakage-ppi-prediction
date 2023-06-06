library(data.table)
library(ggplot2)
library(pheatmap)


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

#### visualize degree rumaatios

degree_ratios <- fread('../network_data/original_degree_ratios.csv')
degree_ratios[dataset == 'richoux_regular', dataset := 'RICHOUX-REGULAR']
degree_ratios[dataset == 'richoux_strict', dataset := 'RICHOUX-STRICT']
degree_ratios[dataset == 'gold_standard', dataset := 'GOLD-STANDARD']
degree_ratios[dataset == 'dscript', dataset := 'D-SCRIPT UNBALANCED']
degree_ratios[, dataset := toupper(dataset)]
degree_ratios[, dataset := factor(dataset, levels = c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBALANCED', 'GOLD-STANDARD'))]
colorBlindBlack8  <- c('#000000', '#E69F00', '#56B4E9', '#009E73', 
                                '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#661100')


ggplot(degree_ratios, aes(degree_ratio, fill=dataset))+
  geom_histogram(binwidth = 0.05)+
  facet_wrap(~dataset, scales='free_y')+
  scale_fill_manual(values = colorBlindBlack8)+
  xlab('Degree Ratio p')+
  theme_bw()+
  theme(legend.position = "none")
ggsave('plots/degree_ratios.pdf', width=9, height=5)


get_proportions <- function(name) {
  if(name == 'gold_standard'){
    all_interactions <- fread('../algorithms/D-SCRIPT-main/data/gold/Intra1.txt')
  }else{
    all_interactions <- fread(paste0('../algorithms/D-SCRIPT-main/data/original/', name, '_test.txt'))
  }
  name <- toupper(name)
  name <- gsub('_', '-', name)
  if(name == 'DSCRIPT'){
    name <- 'D-SCRIPT UNBALANCED'
  }
  deg_close_to_1 <- degree_ratios[dataset == name & degree_ratio >= 0.9, protein]
  deg_close_to_0 <- degree_ratios[dataset == name & degree_ratio <= 0.1, protein]
  all_interactions[, involved_pos := ifelse(V1 %in% deg_close_to_1 | V2 %in% deg_close_to_1, 1, 0)]
  all_interactions[, involved_neg := ifelse(V1 %in% deg_close_to_0 | V2 %in% deg_close_to_0, 1, 0)]
  proportion_high_deg_ratio_pos <- sum(all_interactions[V3 == 1, involved_pos])/nrow(all_interactions[V3 == 1])
  proportion_low_deg_ratio_pos <- sum(all_interactions[V3 == 1, involved_neg])/nrow(all_interactions[V3 == 1])
  proportion_high_deg_ratio_neg <- sum(all_interactions[V3 == 0, involved_pos])/nrow(all_interactions[V3 == 0])
  proportion_low_deg_ratio_neg <- sum(all_interactions[V3 == 0, involved_neg])/nrow(all_interactions[V3 == 0])
  return(data.table('%high p of pos.' = proportion_high_deg_ratio_pos,
                    '%low p of pos.' = proportion_low_deg_ratio_pos, 
                    '%high p of neg' = proportion_high_deg_ratio_neg, 
                    '%low p of neg' = proportion_low_deg_ratio_neg, 
                    name=name))
}

df <- get_proportions('huang')
df <- rbind(df, get_proportions('guo'))
df <- rbind(df, get_proportions('du'))
df <- rbind(df, get_proportions('pan'))
df <- rbind(df, get_proportions('richoux_regular'))
df <- rbind(df, get_proportions('richoux_strict'))
df <- rbind(df, get_proportions('gold_standard'))
df <- rbind(df, get_proportions('dscript'))
df[, name := factor(name, levels = c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT', 'D-SCRIPT UNBALANCED', 'GOLD-STANDARD'))]
df <- df[order(name)]
mat <- as.matrix(df[, c(1:4)])
rownames(mat) <- df$name

pheatmap(mat, 
         cluster_rows = TRUE,
         cluster_cols = FALSE,
         display_numbers = TRUE, 
         legend = FALSE)



