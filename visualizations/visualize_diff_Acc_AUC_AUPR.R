library(data.table)
library(ggplot2)
library(RColorBrewer)
library(pheatmap)

original_results_Acc <-
  fread(paste0('results/original_', 'Accuracy', '.csv'))
original_results_AUC <-
  fread(paste0('results/original_', 'AUC', '.csv'))
original_results_AUPR <-
  fread(paste0('results/original_', 'AUPR', '.csv'))

# merge dataframes
original_results <- merge(original_results_Acc, original_results_AUC, by = c('Model', 'Dataset'))
original_results <- merge(original_results, original_results_AUPR, by = c('Model', 'Dataset'))

original_results$diff_acc_auc <- original_results$Accuracy - original_results$AUC
original_results$diff_acc_aupr <- original_results$Accuracy - original_results$AUPR
original_results$diff_auc_aupr <- original_results$AUC - original_results$AUPR

original_results[Dataset == "dscript", Dataset := "D-SCRIPT"]

original_results[Model == "RF_MDS", Model := "RF MDS"]
original_results[Model == "SVM_MDS", Model := "SVM MDS"]
original_results[Model == "RF_PCA", Model := "RF PCA"]
original_results[Model == "SVM_PCA", Model := "SVM PCA"]
original_results[Model == "RF_node2vec", Model := "RF node2vec"]
original_results[Model == "SVM_node2vec", Model := "SVM node2vec"]
original_results[Model == "degree_cons", Model := "Global and Local\nConsistency"]
original_results[Model == "degree_hf", Model := "Harmonic Function"]
original_results[Model == "Topsy_Turvy", Model := "Topsy Turvy"]
original_results[Model == "deepPPI_FC", Model := "Richoux-FC"]
original_results[Model == "deepPPI_LSTM", Model := "Richoux-LSTM"]

original_results <- original_results[, Model := factor(
  Model,
  levels = c(
    'SPRINT',
    'Richoux-FC',
    'Richoux-LSTM',
    'DeepFE',
    'PIPR',
    'D-SCRIPT',
    'Topsy Turvy',
    'RF PCA',
    'SVM PCA',
    'RF MDS',
    'SVM MDS',
    'RF node2vec',
    'SVM node2vec',
    'Harmonic Function',
    'Global and Local\nConsistency'
  )
)]

# make 3 heatmaps for differences: x-axis: models, y-axis: datasets, values: differences
# 1. Accuracy - AUC
# make matrix
diff_acc_auc <- dcast(original_results, Dataset ~ Model, value.var = 'diff_acc_auc')
rownames(diff_acc_auc) <- paste(diff_acc_auc$Dataset, 'Acc.-AUC', sep = '_')
diff_acc_auc$Dataset <- NULL

# 2. Accuracy - AUPR
diff_acc_aupr <- dcast(original_results, Dataset ~ Model, value.var = 'diff_acc_aupr')
rownames(diff_acc_aupr) <- paste(diff_acc_aupr$Dataset, 'Acc.-AUPR', sep = '_')
diff_acc_aupr$Dataset <- NULL

# 3. AUC - AUPR
diff_auc_aupr <- dcast(original_results, Dataset ~ Model, value.var = 'diff_auc_aupr')
rownames(diff_auc_aupr) <- paste(diff_auc_aupr$Dataset, 'AUC-AUPR', sep = '_')
diff_auc_aupr$Dataset <- NULL

# concatenate the three matrices into one and make annotation row
diff_heatmap <- rbind(diff_acc_auc, diff_acc_aupr, diff_auc_aupr)
diff_heatmap <- as.matrix(diff_heatmap)
rownames(diff_heatmap) <- c(rownames(diff_acc_auc), rownames(diff_acc_aupr), rownames(diff_auc_aupr))

annotation_row <- data.table(names = rownames(diff_heatmap))
annotation_row[, c('Dataset', 'Difference') := tstrsplit(names, '_')]
annotation_row <- data.frame(annotation_row)
rownames(annotation_row) <- annotation_row$names
annotation_row$names <- NULL
annotation_row$Dataset <- NULL
annotation_row$Difference <- as.factor(annotation_row$Difference)

breaksList <- c(min(diff_heatmap, na.rm = T), seq(-0.5, 0.5, 0.0102))
color <- colorRampPalette(rev(brewer.pal(n = 9, name =
                                           "PiYG")))(100)
rgb_colors <-
  col2rgb(
    pheatmap:::scale_colours(
      as.matrix(diff_heatmap),
      col = color,
      breaks = breaksList,
      na_col = "#DDDDDD"
    )
  )
luminance <- rgb_colors * c(0.299, 0.587, 0.114)
luminance <-
  luminance['red',] + luminance['green',] + luminance['blue',]
number_color <- ifelse(luminance < 125, "grey90", "grey30")
colors <- c('#e6194b', '#f032e6', '#ffe119', '#4363d8', '#f58231', '#911eb4',
            '#3cb44b','#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324',
            '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075')

pheatmap(diff_heatmap,
         annotation_row = annotation_row,
         annotation_colors = list(
           Difference = c(
             'Acc.-AUC' = colors[8],
             'Acc.-AUPR' = colors[9],
             'AUC-AUPR' = colors[10]
           )
         ),
         filename = 'plots/differences_acc_auc_aupr.pdf',
         width = 10,
         height = 5,
         cluster_rows = FALSE,
         cluster_cols = FALSE, 
         legend = FALSE,
         gaps_col = c(1, 7),
         gaps_row = c(7, 14),
         breaks = breaksList,
         color = colorRampPalette(rev(brewer.pal(
           n = 9, name =
             "PiYG"
         )))(100),
         display_numbers = TRUE,
         number_color = number_color, 
         labels_row = rep(
           c('D-SCRIPT UNBAL.', 'DU', 'GUO', 'HUANG', 'PAN', 'RICHOUX-REGULAR', 'RICHOUX-STRICT'), 
           3)
)

# get score distributions

get_score_distribution <- function(path){
  scores <- lapply(
    list.files(path, pattern = '(dscript|huang|guo|du|pan|richoux_regular|richoux_strict).txt.predictions.tsv', full.names = TRUE), fread)
  names <- tstrsplit(list.files(path, pattern = '(dscript|huang|guo|du|pan|richoux_regular|richoux_strict).txt.predictions.tsv'), '.txt', keep=1)[[1]]
  names(scores) <- names
  scores <- rbindlist(scores, idcol='Dataset')
  colnames(scores) <- c('Dataset', 'P1', 'P2', 'y_true', 'y_pred')
  scores[, y_true := ifelse(y_true == '0', 'y_true=0', 'y_true=1')]
  scores[, Dataset := toupper(Dataset)]
  scores[Dataset == 'DSCRIPT', Dataset := 'D-SCRIPT UNBAL.']
  scores <- scores[, y_true := factor(y_true, levels = c('y_true=1', 'y_true=0'))]
  return(scores)
}
scores_dscript <- get_score_distribution('../algorithms/D-SCRIPT-main/results_dscript/original')
scores_dscript$Model <- 'D-SCRIPT'
scores_tt <- get_score_distribution('../algorithms/D-SCRIPT-main/results_topsyturvy/original')
scores_tt$Model <- 'Topsy-Turvy'

both_scores <- rbind(scores_dscript, scores_tt)

ggplot(both_scores, aes(x = y_true, y = y_pred)) +
  geom_boxplot()+
  facet_grid(Model~Dataset)+
  geom_hline(yintercept = 0.5, linetype = 'dashed', color = 'red')+
  theme_bw()
ggsave('~/Downloads/scores_ds_tt.png', height = 4, width=10)
