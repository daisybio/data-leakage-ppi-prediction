library(data.table)
library(ggplot2)
library(ComplexUpset)


## Citation count
citations <- data.table(
  citations = c(623, 63, 147, 269, 121, 63, 8, 31, 143, 20, 141, 40, 9, 51, 41, 8, 200, 80, 951, 2, 30, 96, 50, 3, 128, 20),
  code_available = c(F, F, F, F, T, F, F, F, T, F, F, T, F, F, F, F, T, F, F, F, F, T, T, T, T, T)
)

ggplot(citations, aes(x = citations, color = code_available))+
  stat_ecdf()+
  theme_bw()

## unique proteins 
all_sets <- lapply(list.files(path='../algorithms/SPRINT/data/partitions', full.names = T, recursive = T), 
                   fread, header=F)
names(all_sets) <- tstrsplit(list.files(path='../algorithms/SPRINT/data/partitions', recursive = T), '.txt', keep=1)[[1]]
all_sets <- lapply(all_sets, function(x){
  return(unique(melt(x, measure.vars = c('V1', 'V2'), value.name = 'Protein')[, -'variable']))
})
all_sets <- rbindlist(all_sets, idcol='filename')
all_sets[, c('dataset', 'partition') := tstrsplit(filename, '_', keep=c(1,3))]
all_sets <- all_sets[, -'filename']
all_sets[, data_part := paste(dataset, partition, sep='_')]
all_sets[, both_0 := ifelse(partition %in% c('0', 'both'), T, F)]
all_sets[, both_1 := ifelse(partition %in% c('1', 'both'), T, F)]
all_sets[, p0_p1 := ifelse(partition %in% c('0', '1'), T, F)]
all_sets[, dataset := factor(dataset, levels=c('huang', 'guo', 'du', 'pan', 'richoux'))]
all_sets <- all_sets[order(dataset)]

both_0 <- data.table(unique(all_sets[both_0 == T, c('Protein', 'dataset')])[, table(dataset)])
both_0$train <- 'both'
both_0$test <- '0'
both_1 <- data.table(unique(all_sets[both_1 == T, c('Protein', 'dataset')])[, table(dataset)])
both_1$train <- 'both'
both_1$test <- '1'
p0_p1 <- data.table(unique(all_sets[p0_p1 == T, c('Protein', 'dataset')])[, table(dataset)])
p0_p1$train <- '0'
p0_p1$test <- '1'

overlaps <- rbind(both_0, both_1)
overlaps <- rbind(overlaps, p0_p1)
overlaps[, dataset := factor(dataset, levels=c('huang', 'guo', 'du', 'pan', 'richoux'))]
overlaps[, train := factor(train, levels=c('both', '0'))]
overlaps[, data_train := paste(dataset, train, sep='_')]
overlaps[, data_test := paste(dataset, test, sep='_')]

unique_prots_datasets <- data.table(unique(all_sets[, c('Protein', 'data_part')])[, table(data_part)])
overlaps <- merge(overlaps, unique_prots_datasets, all.x = T, by.x = 'data_train', by.y = 'data_part')
overlaps <- merge(overlaps, unique_prots_datasets, all.x = T, by.x = 'data_test', by.y = 'data_part')

get_overlap <- function(dataset, train, test) {
  data_train <- paste(dataset, train, sep='_')
  ds_train <- all_sets[data_part == data_train]
  data_test <- paste(dataset, test, sep='_')
  ds_test <- all_sets[data_part == data_test]
  length(intersect(ds_train$Protein, ds_test$Protein))
}

overlaps[, train_intersect_test := get_overlap(dataset, train, test), by = seq_len(nrow(overlaps))]
overlaps <- overlaps[order(dataset, train)]
overlaps[, Dataset := paste(dataset, train, test, sep='_')]
colnames(overlaps) <- c('Test','Train','Data', 'Unique whole data', 'train', 'test', 'unique_train', 'unique_test', 'train ∩ test', 'Dataset')
overlaps <- overlaps[, c('Dataset', 'Unique whole data', 'unique_train', 'unique_test', 'train ∩ test', 'Train', 'Test')]
overlaps[, perc_of_test := round(100 * `train ∩ test`/unique_train, 1)]

unique_prots_datasets <- setNames(unique_prots_datasets$N, unique_prots_datasets$data_part)

all_proteins <- data.table(Protein = unique(all_sets$Protein))
intersect_list <- list()

for(dataset_var in c('huang', 'guo', 'du', 'pan', 'richoux')){
  for(partition_var in c('both_0', 'both_1', 'p0_p1')){
    var_name <- paste(dataset_var, partition_var, sep = '_')
    if(partition_var != 'p0_p1'){
      train <- paste(dataset_var, tstrsplit(partition_var, '_', keep=1)[[1]], sep='_')
      test <- paste(dataset_var, tstrsplit(partition_var, '_', keep=2)[[1]], sep='_')
      tmp_list <- list(var_name, c(var_name, train), c(train, test))
    }else{
      tmp_list <- list(var_name, c(var_name, paste(dataset_var, '0', sep='_')))
    }
    intersect_list <- append(intersect_list, tmp_list)
    all_proteins[, eval(var_name) := Protein %in% unique(all_sets[dataset == dataset_var & 
                                                                    get(partition_var) == TRUE, Protein])]
  }
}

for(dataset_var in c('huang', 'guo', 'du', 'pan', 'richoux')){
  for(partition_var in c('0', '1', 'both')){
    var_name <- paste(dataset_var, partition_var, sep = '_')
    all_proteins[, eval(var_name) := Protein %in% unique(all_sets[dataset == dataset_var & 
                                                                   partition == partition_var, Protein])]
  }
}
upset(all_proteins, 
      colnames(all_proteins)[grepl('huang|du', colnames(all_proteins))],
      mode = 'intersect',
      width_ratio = 0.1,
      intersections = intersect_list[grepl('huang|du', intersect_list)]
      )

