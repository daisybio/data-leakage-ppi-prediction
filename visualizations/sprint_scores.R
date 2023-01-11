library(data.table)
library(ggplot2)
library(latex2exp)

sprint_dir <- '../algorithms/SPRINT/results/'
original_results <- lapply(list.files(paste0(sprint_dir, 'original/'), 
                                      pattern = '(du|guo|huang|pan|richoux_regular|richoux_strict)_results.txt$',
                                      full.names = TRUE
                                      ), fread)
names(original_results) <- tstrsplit(list.files(paste0(sprint_dir, 'original/'), 
                        pattern = '(du|guo|huang|pan|richoux_regular|richoux_strict)_results.txt$'),
                      '_results', keep=1)[[1]]
original_results <- rbindlist(original_results, idcol = 'dataset')
original_results$test <- 'Original'

rewired_results <- lapply(list.files(paste0(sprint_dir, 'rewired/'), 
                                      pattern = '(du|guo|huang|pan|richoux_regular|richoux_strict)_results.txt$',
                                      full.names = TRUE
), fread)
names(rewired_results) <- tstrsplit(list.files(paste0(sprint_dir, 'rewired/'), 
                                                pattern = '(du|guo|huang|pan|richoux_regular|richoux_strict)_results.txt$'),
                                     '_results', keep=1)[[1]]
rewired_results <- rbindlist(rewired_results, idcol = 'dataset')
rewired_results$test <- 'Rewired'

all_results <- rbind(original_results, rewired_results)
colnames(all_results) <- c('Dataset', 'Score', 'True Label', 'Test')
all_results$`True Label` <- as.factor(all_results$`True Label`)

partition_results <- lapply(list.files(paste0(sprint_dir, 'partitions/'), 
                                     pattern = '(du|guo|huang|pan|richoux)_train_(0|both)_test_(0|1).txt$',
                                     full.names = TRUE), fread)
names(partition_results) <- tstrsplit(list.files(paste0(sprint_dir, 'partitions/'), 
                                                 pattern = '(du|guo|huang|pan|richoux)_train_(0|both)_test_(0|1).txt$'),
                                    '.txt', keep=1)[[1]]
partition_results <- rbindlist(partition_results, idcol = 'dataset')
partition_results[, test := tstrsplit(dataset, '(du|guo|huang|pan|richoux)_', keep=2)]
partition_results[, dataset := tstrsplit(dataset, '_', keep=1)]
partition_results[dataset == 'richoux', dataset := 'richoux-uniprot']
partition_results$test <- gsub('train_both_', 'Inter->', partition_results$test)
partition_results$test <- gsub('train_0_', 'Intra-0->', partition_results$test)
partition_results$test <- gsub('test_0', 'Intra-0', partition_results$test)
partition_results$test <- gsub('test_1', 'Intra-1', partition_results$test)
colnames(partition_results) <- c('Dataset', 'Score', 'True Label', 'Test')
all_results <- rbind(all_results, partition_results)
all_results$Test <- factor(all_results$Test, levels=c('Original', 'Rewired', 'Inter->Intra-1', 'Inter->Intra-0', 'Intra-0->Intra-1'))
all_results[, Dataset := gsub('richoux_regular', 'richoux-regular', Dataset)]
all_results[, Dataset := gsub('richoux_strict', 'richoux-strict', Dataset)]
all_results[, Dataset := stringr::str_to_upper(Dataset)]
all_results$Dataset <- factor(all_results$Dataset, 
                              levels = c('HUANG', 'GUO', 'DU', 'RICHOUX-UNIPROT', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT'))

ggplot(all_results, aes(x = Test, y = log(Score), fill = `True Label`))+
  geom_boxplot()+
  facet_wrap(~Dataset)+
  theme_bw()+
  theme(text = element_text(size=15),axis.text.x = element_text(angle = 90, vjust = 1, hjust=1))+
  scale_x_discrete(labels = c(
    'Original' = 'Original', 
    'Rewired' = 'Rewired', 
    'Inter->Intra-1' = TeX('$\\it{INTER} \\rightarrow \\it{INTRA}_1$'), 
    'Inter->Intra-0' = TeX('$\\it{INTER} \\rightarrow \\it{INTRA}_0$'), 
    'Intra-0->Intra-1' = TeX('$\\it{INTRA}_0 \\rightarrow \\it{INTRA}_1$')
  ))
ggsave('plots/sprint_scores.pdf', height=8, width=10)



