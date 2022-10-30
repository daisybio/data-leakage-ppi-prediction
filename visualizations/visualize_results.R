library(data.table)
library(ggplot2)
library(RColorBrewer)
library(pheatmap)

original_results <- fread('results/original.csv')
original_results$Test <- 'Original'
rewired_results <- fread('results/rewired.csv')
rewired_results$Test <- 'Rewired'
partition_results <- fread('results/partition.csv')
colnames(partition_results) <- c('Model', 'Dataset', 'Accuracy', 'Test')

all_results <- rbind(original_results, rewired_results)
all_results <- all_results[, Model := factor(Model, 
                                             levels=c("SPRINT", 
                                                      "deepPPI_FC", "deepPPI_LSTM",  
                                                      "DeepFE", "PIPR", "RF_PCA","SVM_PCA", "RF_MDS", "SVM_MDS",
                                                      "RF_node2vec",  "SVM_node2vec"))]
all_results <- all_results[, Dataset := factor(Dataset, 
                                               levels = c("huang", "guo", "du", "pan", "richoux-regular", "richoux-strict"))]

ggplot(all_results, aes(x=Dataset, y = Accuracy, color=Test, group=Test))+
  geom_line(size=1, alpha=0.7)+
  geom_point(size=3)+
  facet_wrap(~Model)+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))+
  scale_x_discrete(labels=c("huang" = "Huang", "guo" = "Guo",
                            "du" = "Du","pan" = "Pan", "richoux-regular" = "Richoux regular",
                            "richoux-strict" = "Richoux strict"))
#ggsave("./original_vs_rewired.png",height=8, width=10)

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
                                             levels=c("SPRINT", 
                                                      "Richoux-FC", "Richoux-LSTM",  
                                                      "DeepFE", "PIPR", "RF PCA","SVM PCA", "RF MDS", "SVM MDS",
                                                      "RF node2vec",  "SVM node2vec"))]
ggplot(all_results, aes(x=Dataset, y = Accuracy, color=Test, group=Test))+
  geom_line(size=1, alpha=0.7)+
  geom_point(size=2)+
  facet_wrap(~Model)+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))+
  scale_x_discrete(labels=c("huang" = "Huang", "guo" = "Guo",
                            "du" = "Du","pan" = "Pan", "richoux-regular" = "Richoux regular",
                            "richoux-strict" = "Richoux strict"))+
  scale_color_manual(values = brewer.pal(5,"Set2")[c(2,3,1,5,4)])
#ggsave("./original_vs_rewired_vs_partitions.png",height=8, width=10)

ggplot(all_results, aes(x=Dataset, y = Accuracy, color=Model, group=Model))+
  geom_line(size=1, alpha=0.7)+
  geom_point(size=2)+
  facet_wrap(~Test)+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))+
  scale_x_discrete(labels=c("huang" = "Huang", "guo" = "Guo",
                            "du" = "Du","pan" = "Pan", "richoux-regular" = "Richoux regular",
                            "richoux-strict" = "Richoux strict"))+
  scale_color_manual(values = brewer.pal(12,"Paired")[-11])
#ggsave("./original_vs_rewired_vs_partitions_models.png",height=8, width=10)


colorBlindBlack8  <- c("#000000", "#E69F00", "#56B4E9", "#009E73", 
                       "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
result_mat <- as.matrix(dcast(all_results, Model ~ Dataset + Test, value.var = "Accuracy"))
rownames(result_mat) <- result_mat[, "Model"]
result_mat <- result_mat[, -1]
class(result_mat) <- "numeric"
annotation_col <- as.data.frame(tstrsplit(colnames(result_mat), '_'), col.names = c('Dataset', 'Test'))
rownames(annotation_col) <- colnames(result_mat)
annotation_col$Dataset <- stringr::str_to_title(annotation_col$Dataset)
annotation_col$Dataset <- factor(annotation_col$Dataset, 
                                 levels = c('Huang', 'Guo', 'Du', 'Pan', 'Richoux', 'Richoux-Regular', 'Richoux-Strict'))
annotation_col$Test <- gsub('both', 'Inter' ,annotation_col$Test)
annotation_col$Test <- gsub('0', 'Intra-0' ,annotation_col$Test)
annotation_col$Test <- gsub('1', 'Intra-1' ,annotation_col$Test)
annotation_col$Test <- factor(annotation_col$Test, 
                              levels = c('Original', 'Rewired', 'Inter->Intra-1', 'Inter->Intra-0', 'Intra-0->Intra-1'))
annotation_col <- annotation_col[order(annotation_col$Test, annotation_col$Dataset), ]
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


pheatmap(result_mat, 
         annotation_col = annotation_col, 
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         gaps_row = 5,
         gaps_col = c(6,12,17,22),
         display_numbers = TRUE,
         #filename = './heatmap_results.png',
         #width=10,
         #height=8,
         labels_col = c(paste0("Huang (", original_sizes["huang"], ")"),
           paste0("Guo (", original_sizes["guo"], ")"),
           paste0("Du (", original_sizes["du"], ")"),
           paste0("Pan (", original_sizes["pan"], ")"), 
           paste0("Richoux regular (", original_sizes["richoux-regular"], ")"),
           paste0("Richoux strict (", original_sizes["richoux-strict"], ")"), 
           #rewired
           paste0("Huang (", rewired_sizes["huang"], ")"),
           paste0("Guo (", rewired_sizes["guo"], ")"),
           paste0("Du (", rewired_sizes["du"], ")"),
           paste0("Pan (", rewired_sizes["pan"], ")"), 
           paste0("Richoux regular (", rewired_sizes["richoux-regular"], ")"),
           paste0("Richoux strict (", rewired_sizes["richoux-strict"], ")"),
           #partition both ->1
           paste0("Huang (", partition_sizes["huang both"], ")"),
           paste0("Guo (", partition_sizes["guo both"], ")"),
           paste0("Du (", partition_sizes["du both"], ")"),
           paste0("Pan (", partition_sizes["pan both"], ")"), 
           paste0("Richoux (", partition_sizes["richoux both"], ")"),
           #partition both -> 0
           paste0("Huang (", partition_sizes["huang both"], ")"),
           paste0("Guo (", partition_sizes["guo both"], ")"),
           paste0("Du (", partition_sizes["du both"], ")"),
           paste0("Pan (", partition_sizes["pan both"], ")"), 
           paste0("Richoux (", partition_sizes["richoux both"], ")"),
           #partition 0 -> 1
           paste0("Huang (", partition_sizes["huang 0"], ")"),
           paste0("Guo (", partition_sizes["guo 0"], ")"),
           paste0("Du (", partition_sizes["du 0"], ")"),
           paste0("Pan (", partition_sizes["pan 0"], ")"), 
           paste0("Richoux (", partition_sizes["richoux 0"], ")")
         )
)

pheatmap(t(result_mat[, 1:6]), 
         annotation_row = annotation_col[annotation_col$Test == 'Original', 'Dataset', drop=FALSE], 
         annotation_colors = list(Dataset = c("Huang"="#E69F00","Guo"="#56B4E9", "Du"="#009E73", 
                              "Pan"="#F0E442","Richoux-Regular"="#0072B2","Richoux-Strict"="#CC79A7")),
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         gaps_col = 5,
         display_numbers = TRUE,
         filename = './heatmap_results_original_quer.png',
         width=8,
         height=4,
         labels_row = c(paste0("Huang (", original_sizes["huang"], ")"),
                        paste0("Guo (", original_sizes["guo"], ")"),
                        paste0("Du (", original_sizes["du"], ")"),
                        paste0("Pan (", original_sizes["pan"], ")"), 
                        paste0("Richoux regular (", original_sizes["richoux-regular"], ")"),
                        paste0("Richoux strict (", original_sizes["richoux-strict"], ")")
         )
)

pheatmap(t(result_mat[, 7:12]), 
         annotation_row = annotation_col[annotation_col$Test == 'Rewired', 'Dataset', drop=FALSE], 
         annotation_colors = list(Dataset = c("Huang"="#E69F00","Guo"="#56B4E9", "Du"="#009E73", 
                                              "Pan"="#F0E442","Richoux-Regular"="#0072B2","Richoux-Strict"="#CC79A7")),
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         gaps_col = 5,
         display_numbers = TRUE,
         filename = './heatmap_results_rewired_quer.png',
         width=8,
         height=4,
         labels_row = c(paste0("Huang (", rewired_sizes["huang"], ")"),
                        paste0("Guo (", rewired_sizes["guo"], ")"),
                        paste0("Du (", rewired_sizes["du"], ")"),
                        paste0("Pan (", rewired_sizes["pan"], ")"), 
                        paste0("Richoux regular (", rewired_sizes["richoux-regular"], ")"),
                        paste0("Richoux strict (", rewired_sizes["richoux-strict"], ")")
         )
)

pheatmap(result_mat[, 13:ncol(result_mat)], 
         annotation_col = annotation_col[!annotation_col$Test %in% c('Original', 'Rewired'), ], 
         annotation_colors = list(Dataset = c("Huang"="#E69F00","Guo"="#56B4E9", "Du"="#009E73", 
                                              "Pan"="#F0E442","Richoux"="#0072B2"),
                                  Test = c("Inter->Intra-1"="#888888", "Inter->Intra-0"="#44AA99", "Intra-0->Intra-1"="#661100")),
         cluster_rows = FALSE,
         cluster_cols = FALSE,
         gaps_row = 5,
         display_numbers = TRUE,
         filename = './heatmap_results_partitions.png',
         width=8,
         height=5,
         labels_col = c(#partition both ->1
           paste0("Huang (", partition_sizes["huang both"], ")"),
           paste0("Guo (", partition_sizes["guo both"], ")"),
           paste0("Du (", partition_sizes["du both"], ")"),
           paste0("Pan (", partition_sizes["pan both"], ")"), 
           paste0("Richoux (", partition_sizes["richoux both"], ")"),
           #partition both -> 0
           paste0("Huang (", partition_sizes["huang both"], ")"),
           paste0("Guo (", partition_sizes["guo both"], ")"),
           paste0("Du (", partition_sizes["du both"], ")"),
           paste0("Pan (", partition_sizes["pan both"], ")"), 
           paste0("Richoux (", partition_sizes["richoux both"], ")"),
           #partition 0 -> 1
           paste0("Huang (", partition_sizes["huang 0"], ")"),
           paste0("Guo (", partition_sizes["guo 0"], ")"),
           paste0("Du (", partition_sizes["du 0"], ")"),
           paste0("Pan (", partition_sizes["pan 0"], ")"), 
           paste0("Richoux (", partition_sizes["richoux 0"], ")")
         )
)
