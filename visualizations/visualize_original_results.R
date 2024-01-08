library(data.table)
library(ggplot2)
library(RColorBrewer)

es <- TRUE
for (measure in c('Accuracy',
                  'Precision',
                  'Recall',
                  'Specificity',
                  'F1',
                  'MCC',
                  'AUC',
                  'AUPR')) {
  print(measure)
  
  #### result prefixes
  custom_res <- '../algorithms/Custom/results/'
  deepFE_res <- '../algorithms/DeepFE-PPI/result/custom/'
  deepPPI_res <- '../algorithms/DeepPPI/keras/results_custom/'
  seqppi_res <-
    '../algorithms/seq_ppi/binary/model/lasagna/results/'
  sprint_res <- '../algorithms/SPRINT/results/original/'
  dscript_res <-
    '../algorithms/D-SCRIPT-main/results_dscript/original/'
  tt_res <-
    '../algorithms/D-SCRIPT-main/results_topsyturvy/original/'
  
  # read in data
  all_results <-
    data.table(1)[, `:=` (c("Model", "Dataset", measure), NA)][, V1 := NULL][.0]
  
  # custom
  custom_results <-
    lapply(paste0(
      custom_res,
      list.files(custom_res, pattern = '^original_(du|guo|huang|pan|richoux|dscript).*.csv')
    ), fread)
  file_names <-
    tstrsplit(
      list.files(custom_res, pattern = '^original_(du|guo|huang|pan|richoux|dscript).*.csv'),
      '.csv',
      keep = 1
    )[[1]]
  file_names[grepl('richoux', file_names, fixed = TRUE)] <-
    gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed =
                                                     TRUE)])
  names(custom_results) <- file_names
  custom_results <- rbindlist(custom_results, idcol = 'filename')
  custom_results[, c('dataset', 'encoding', 'method') := tstrsplit(filename, '_', keep =
                                                                     c(2, 3, 4))]
  custom_results[is.na(method), method := 'degree']
  custom_results[, Model := paste(method, encoding, sep = '_')]
  if (measure == 'Recall') {
    custom_results <- custom_results[V1 == 'Sensitivity']
  } else{
    if (measure == 'Accuracy') {
      wide_df <-
        dcast(custom_results[dataset == 'dscript', ],
              filename + dataset + encoding + method + Model ~ V1,
              value.var = 'V2')
      wide_df[, Balanced_Accuracy := 0.5 * (Sensitivity + Specificity)]
      wide_df <- wide_df[order(filename)]
      custom_results <- custom_results[order(filename)]
      custom_results[dataset == 'dscript' &
                       V1 == 'Accuracy', V2 := wide_df$Balanced_Accuracy]
    }
    custom_results <- custom_results[V1 == measure]
  }
  colnames(custom_results) <-
    c('filename',
      'Measure',
      measure,
      'Dataset',
      'Encoding',
      'Method',
      'Model')
  
  all_results <-
    rbind(all_results, custom_results[, c('Model', 'Dataset', measure), with =
                                        FALSE])
  
  # deepFE
  if (es) {
    deepFE_results <-
      lapply(paste0(
        deepFE_res,
        list.files(deepFE_res, pattern = '^original_scores_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript)_es.csv', recursive = TRUE)
      ), fread)
    file_names <-
      tstrsplit(
        list.files(deepFE_res, pattern = '^original_scores_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript)_es.csv', recursive = TRUE),
        '/',
        keep = 1
      )[[1]]
  } else{
    deepFE_results <-
      lapply(paste0(
        deepFE_res,
        list.files(deepFE_res, pattern = '^original_scores_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript).csv', recursive = TRUE)
      ), fread)
    file_names <-
      tstrsplit(
        list.files(deepFE_res, pattern = '^original_scores_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript).csv', recursive = TRUE),
        '/',
        keep = 1
      )[[1]]
  }
  file_names[grepl('richoux', file_names, fixed = TRUE)] <-
    gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed =
                                                     TRUE)])
  names(deepFE_results) <- file_names
  deepFE_results <- rbindlist(deepFE_results, idcol = 'Dataset')
  if (measure == 'Accuracy') {
    balanced_accuracy <-
      0.5 * (deepFE_results[Dataset == 'dscript' &
                              V1 == 'Recall', Score] + deepFE_results[Dataset == 'dscript' &
                                                                        V1 == 'Specificity', Score])
    deepFE_results[Dataset == 'dscript' &
                     V1 == 'Accuracy', Score := balanced_accuracy]
  }
  deepFE_results <- deepFE_results[V1 == measure]
  colnames(deepFE_results) <- c('Dataset', 'Measure', measure)
  deepFE_results$Model <- 'DeepFE'
  
  all_results <-
    rbind(all_results, deepFE_results[, c('Model', 'Dataset', measure), with =
                                        FALSE])
  
  # deepPPI
  if (es) {
    deepPPI_results <-
      lapply(paste0(
        deepPPI_res,
        list.files(deepPPI_res, pattern = 'original_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript)_es.csv')
      ), fread)
    file_names <-
      tstrsplit(
        list.files(deepPPI_res, pattern = 'original_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript)_es.csv'),
        '.csv',
        keep = 1
      )[[1]]
  } else{
    deepPPI_results <-
      lapply(paste0(
        deepPPI_res,
        list.files(deepPPI_res, pattern = 'original_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript).csv')
      ), fread)
    file_names <-
      tstrsplit(
        list.files(deepPPI_res, pattern = 'original_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript).csv'),
        '.csv',
        keep = 1
      )[[1]]
  }
  file_names[grepl('richoux', file_names, fixed = TRUE)] <-
    gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed =
                                                     TRUE)])
  names(deepPPI_results) <- file_names
  deepPPI_results <- rbindlist(deepPPI_results, idcol = 'filename')
  deepPPI_results <-
    deepPPI_results[, c('Model', 'Dataset') := tstrsplit(filename, '_', keep = c(1, 3))]
  if (measure == 'Accuracy') {
    balanced_accuracy_FC <-
      0.5 * (deepPPI_results[Model == 'FC' &
                               Dataset == 'dscript' &
                               variable == 'Recall', value] + deepPPI_results[Model == 'FC' &
                                                                                Dataset == 'dscript' &
                                                                                variable == 'Specificity', value])
    balanced_accuracy_LSTM <-
      0.5 * (deepPPI_results[Model == 'LSTM' &
                               Dataset == 'dscript' &
                               variable == 'Recall', value] + deepPPI_results[Model == 'LSTM' &
                                                                                Dataset == 'dscript' &
                                                                                variable == 'Specificity', value])
    deepPPI_results[Model == 'FC' &
                      Dataset == 'dscript' &
                      variable == 'Accuracy', value := balanced_accuracy_FC]
    deepPPI_results[Model == 'LSTM' &
                      Dataset == 'dscript' &
                      variable == 'Accuracy', value := balanced_accuracy_LSTM]
  }
  deepPPI_results <- deepPPI_results[variable == measure]
  deepPPI_results[, Model := paste('deepPPI', Model, sep = '_')]
  colnames(deepPPI_results) <-
    c('filename', 'variable', measure, 'Model', 'Dataset')
  
  all_results <-
    rbind(all_results, deepPPI_results[, c('Model', 'Dataset', measure), with =
                                         FALSE])
  
  # PIPR
  if (es) {
    pipr_results <-
      lapply(paste0(
        seqppi_res,
        list.files(seqppi_res, pattern = '^original_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript)_es.csv')
      ), fread)
    file_names <-
      tstrsplit(
        list.files(seqppi_res, pattern = '^original_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript)_es.csv'),
        '.csv',
        keep = 1
      )[[1]]
  } else{
    pipr_results <-
      lapply(paste0(
        seqppi_res,
        list.files(seqppi_res, pattern = '^original_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript).csv')
      ), fread)
    file_names <-
      tstrsplit(
        list.files(seqppi_res, pattern = '^original_(du|guo|huang|pan|richoux_regular|richoux_strict|dscript).csv'),
        '.csv',
        keep = 1
      )[[1]]
  }
  file_names[grepl('richoux', file_names, fixed = TRUE)] <-
    gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed =
                                                     TRUE)])
  names(pipr_results) <- file_names
  pipr_results <- rbindlist(pipr_results, idcol = 'Filename')
  pipr_results <-
    pipr_results[, c('Test', 'Dataset') := tstrsplit(Filename, '_', keep =
                                                       c(1, 2))]
  if (measure == 'Accuracy') {
    balanced_accuracy <-
      0.5 * (pipr_results[Dataset == 'dscript' &
                            V1 == 'Recall', Score] + pipr_results[Dataset == 'dscript' &
                                                                    V1 == 'Specificity', Score])
    pipr_results[Dataset == 'dscript' &
                   V1 == 'Accuracy', Score := balanced_accuracy]
  }
  pipr_results <- pipr_results[V1 == measure]
  pipr_results$Model <- 'PIPR'
  colnames(pipr_results) <-
    c('Filename', 'Measure', measure, 'Test', 'Dataset', 'Model')
  
  all_results <-
    rbind(all_results, pipr_results[, c('Model', 'Dataset', measure), with =
                                      FALSE])
  
  # SPRINT
  sprint_results <- fread(paste0(sprint_res, 'all_results.tsv'))
  sprint_results$Model <- 'SPRINT'
  if(measure != 'AUC' & measure != 'AUPR'){
    # take the AUPR
    colnames(sprint_results) <- c('Dataset', 'AUC', measure, 'Model')
  }
  sprint_results$Dataset[grepl('richoux', sprint_results$Dataset, fixed =
                                 TRUE)] <-
    gsub('richoux_*', 'richoux-', sprint_results$Dataset[grepl('richoux', sprint_results$Dataset, fixed =
                                                                 TRUE)])
  sprint_results <- sprint_results[Dataset != 'gold_standard']
  all_results <-
    rbind(all_results, sprint_results[, c('Model', 'Dataset', measure), with =
                                        FALSE])
  
  # D-Script
  if (es) {
    dscript_results <- fread(paste0(dscript_res, 'all_results_es.tsv'))
  } else{
    dscript_results <- fread(paste0(dscript_res, 'all_results.tsv'))
  }
  
  if (measure == 'Accuracy') {
    balanced_accuracy <-
      0.5 * (dscript_results[Dataset == 'dscript' &
                               Metric == 'Recall', Value] + dscript_results[Dataset == 'dscript' &
                                                                              Metric == 'Specificity', Value])
    dscript_results[Dataset == 'dscript' &
                      Metric == 'Accuracy', Value := balanced_accuracy]
  }
  dscript_results <-
    dscript_results[Metric == measure & Dataset != 'gold']
  colnames(dscript_results) <-
    c('Model', 'Dataset', 'Metric', measure, 'Split')
  dscript_results$Model <- 'D-SCRIPT'
  dscript_results[Dataset == 'richoux_regular', Dataset := 'richoux-regular']
  dscript_results[Dataset == 'richoux_strict', Dataset := 'richoux-strict']
  all_results <-
    rbind(all_results, dscript_results[, c('Model', 'Dataset', measure), with =
                                         FALSE])
  
  # Topsy_Turvy
  if (es) {
    tt_results <- fread(paste0(tt_res, 'all_results_es.tsv'))
  } else{
    tt_results <- fread(paste0(tt_res, 'all_results.tsv'))
  }
  
  if (measure == 'Accuracy') {
    balanced_accuracy <-
      0.5 * (tt_results[Dataset == 'dscript' &
                          Metric == 'Recall', Value] + tt_results[Dataset == 'dscript' &
                                                                    Metric == 'Specificity', Value])
    tt_results[Dataset == 'dscript' &
                 Metric == 'Accuracy', Value := balanced_accuracy]
  }
  tt_results <- tt_results[Metric == measure & Dataset != 'gold']
  colnames(tt_results) <-
    c('Model', 'Dataset', 'Metric', measure, 'Split')
  tt_results$Model <- 'Topsy_Turvy'
  tt_results[Dataset == 'richoux_regular', Dataset := 'richoux-regular']
  tt_results[Dataset == 'richoux_strict', Dataset := 'richoux-strict']
  all_results <-
    rbind(all_results, tt_results[, c('Model', 'Dataset', measure), with =
                                    FALSE])
  
  # visualization
  all_results <- all_results[, Dataset := factor(
    Dataset,
    levels = c(
      "huang",
      "guo",
      "du",
      "pan",
      "richoux-regular",
      "richoux-strict",
      "dscript"
    )
  )]
  
  all_results <- all_results[, Model := factor(
    Model,
    levels = c(
      "RF_PCA",
      "SVM_PCA",
      "RF_MDS",
      "SVM_MDS",
      "RF_node2vec",
      "SVM_node2vec",
      "degree_cons",
      "degree_hf",
      "SPRINT",
      "deepPPI_FC",
      "deepPPI_LSTM",
      "DeepFE",
      "PIPR",
      "D-SCRIPT",
      "Topsy_Turvy"
    )
  )]
  if (es) {
    fwrite(all_results,
           file = paste0('results/original_', measure, '_es.csv'))
  } else{
    fwrite(all_results, file = paste0('results/original_', measure, '.csv'))
  }
}
