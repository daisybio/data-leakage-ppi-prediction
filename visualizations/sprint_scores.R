library(data.table)
library(ggplot2)
sprint_dir <- '../algorithms/SPRINT/results/'
original_results <- lapply(list.files(paste0(sprint_dir, 'original/'), 
                                      pattern = '(du|guo|huang|pan|richoux_regular|richoux_strict)_results.txt$',
                                      full.names = TRUE
                                      ), fread)
names(original_results) <- tstrsplit(list.files(paste0(sprint_dir, 'original/'), 
                        pattern = '(du|guo|huang|pan|richoux_regular|richoux_strict)_results.txt$'),
                      '_results', keep=1)[[1]]
original_results <- rbindlist(original_results, idcol = 'dataset')
original_results$test <- 'original'

rewired_results <- lapply(list.files(paste0(sprint_dir, 'rewired/'), 
                                      pattern = '(du|guo|huang|pan|richoux_regular|richoux_strict)_results.txt$',
                                      full.names = TRUE
), fread)
names(rewired_results) <- tstrsplit(list.files(paste0(sprint_dir, 'rewired/'), 
                                                pattern = '(du|guo|huang|pan|richoux_regular|richoux_strict)_results.txt$'),
                                     '_results', keep=1)[[1]]
rewired_results <- rbindlist(rewired_results, idcol = 'dataset')
rewired_results$test <- 'rewired'

all_results <- rbind(original_results, rewired_results)
colnames(all_results) <- c('dataset', 'score', 'true label', 'test')
all_results$`true label` <- as.factor(all_results$`true label`)

partition_results <- lapply(list.files(paste0(sprint_dir, 'partitions/'), 
                                     pattern = '(du|guo|huang|pan|richoux)_train_(0|both)_test_(0|1).txt$',
                                     full.names = TRUE), fread)
names(partition_results) <- tstrsplit(list.files(paste0(sprint_dir, 'partitions/'), 
                                                 pattern = '(du|guo|huang|pan|richoux)_train_(0|both)_test_(0|1).txt$'),
                                    '.txt', keep=1)[[1]]
partition_results <- rbindlist(partition_results, idcol = 'dataset')
partition_results[, test := tstrsplit(dataset, '(du|guo|huang|pan|richoux)_', keep=2)]
partition_results[, dataset := tstrsplit(dataset, '_', keep=1)]
partition_results$test <- gsub('train_both_', 'inter->', partition_results$test)
partition_results$test <- gsub('train_0_', 'intra-0->', partition_results$test)
partition_results$test <- gsub('test_0', 'intra-0', partition_results$test)
partition_results$test <- gsub('test_1', 'intra-1', partition_results$test)
colnames(partition_results) <- c('dataset', 'score', 'true label', 'test')
all_results <- rbind(all_results, partition_results)
all_results$test <- factor(all_results$test, levels=c('original', 'rewired', 'inter->intra-1', 'inter->intra-0', 'intra-0->intra-1'))
all_results$dataset <- factor(all_results$dataset, 
                              levels = c('huang', 'guo', 'du', 'richoux', 'pan', 'richoux_regular', 'richoux_strict'))

ggplot(all_results, aes(x = test, y = log(score), fill = `true label`))+
  geom_boxplot()+
  facet_wrap(~dataset)+
  theme_bw()+
  theme(text = element_text(size=15),axis.text.x = element_text(angle = 90, vjust = 1, hjust=1))
ggsave('sprint_scores.png', height=8, width=10)



