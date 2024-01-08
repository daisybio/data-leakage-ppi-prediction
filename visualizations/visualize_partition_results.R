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
  custom_res <- '../algorithms/Custom/results/partition_tests/'
  deepFE_res <- '../algorithms/DeepFE-PPI/result/custom/'
  deepPPI_res <- '../algorithms/DeepPPI/keras/results_custom/'
  seqppi_res <- '../algorithms/seq_ppi/binary/model/lasagna/results/'
  sprint_res <- '../algorithms/SPRINT/results/partitions/'
  dscript_res <-
    '../algorithms/D-SCRIPT-main/results_dscript/partitions/'
  tt_res <-
    '../algorithms/D-SCRIPT-main/results_topsyturvy/partitions/'
  
  # read in data
  all_results <-
    data.table(1)[, `:=` (c("Model", "Dataset", measure, "Partition"), NA)][, V1 := NULL][.0]
  
  # custom
  custom_results <-
    lapply(paste0(
      custom_res,
      list.files(custom_res, pattern = '^(du|guo|huang|pan|richoux|dscript)')
    ), fread)
  file_names <-
    tstrsplit(
      list.files(custom_res, pattern = '^(du|guo|huang|pan|richoux|dscript)'),
      '.csv',
      keep = 1
    )[[1]]
  names(custom_results) <- file_names
  custom_results <- rbindlist(custom_results, idcol = 'filename')
  custom_results[, c('dataset', 'encoding', 'model', 'train', 'test') := tstrsplit(filename, '_')]
  
  degree_results <-
    lapply(paste0(custom_res,  list.files(custom_res, pattern = '^partition')), fread)
  file_names <-
    tstrsplit(list.files(custom_res, pattern = '^partition'), '.csv', keep =
                1)[[1]]
  names(degree_results) <- file_names
  degree_results <- rbindlist(degree_results, idcol = 'filename')
  degree_results[, c('partition', 'dataset', 'train', 'test', 'model') := tstrsplit(filename, '_')]
  
  if (measure == 'Recall') {
    custom_results <- custom_results[V1 == 'Sensitivity']
    degree_results <- degree_results[V1 == 'Sensitivity']
  } else{
    if (measure == 'Accuracy') {
      wide_df <-
        dcast(custom_results[dataset == 'dscript', ],
              filename + dataset + encoding + model + train + test ~ V1,
              value.var = 'V2')
      wide_df[, Balanced_Accuracy := 0.5 * (Sensitivity + Specificity)]
      wide_df <- wide_df[order(filename)]
      custom_results <- custom_results[order(filename)]
      custom_results[dataset == 'dscript' &
                       V1 == 'Accuracy', V2 := wide_df$Balanced_Accuracy]
      
      wide_df <-
        dcast(
          degree_results[dataset == 'dscript', ],
          filename + partition + dataset + train + test + model ~ V1,
          value.var = 'V2'
        )
      wide_df[, Balanced_Accuracy := 0.5 * (Sensitivity + Specificity)]
      wide_df <- wide_df[order(filename)]
      degree_results <- degree_results[order(filename)]
      degree_results[dataset == 'dscript' &
                       V1 == 'Accuracy', V2 := wide_df$Balanced_Accuracy]
    }
    custom_results <- custom_results[V1 == measure]
    degree_results <- degree_results[V1 == measure]
  }
  custom_results[, Model := paste(model, encoding, sep = '_')]
  degree_results[, Model := paste('degree', model, sep = '_')]
  colnames(custom_results) <-
    c(
      'filename',
      'Measure',
      measure,
      'Dataset',
      'Encoding',
      'model',
      'Train',
      'Test',
      'Model'
    )
  colnames(degree_results) <-
    c(
      'filename',
      'Measure',
      measure,
      'partition',
      'Dataset',
      'Train',
      'Test',
      'model',
      'Model'
    )
  custom_results[, Partition := paste(Train, Test, sep = '->')]
  degree_results[, Partition := paste(Train, Test, sep = '->')]
  
  all_results <-
    rbind(all_results, custom_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])
  all_results <-
    rbind(all_results, degree_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])
  
  # deepFE
  if (es) {
    deepFE_results <-
      lapply(paste0(
        deepFE_res,
        list.files(deepFE_res, pattern = '^partition_scores_(du|guo|huang|pan|richoux|dscript)_(both|0)_(0|1)_es.csv', recursive = TRUE)
      ), fread)
    file_names <-
      tstrsplit(
        list.files(deepFE_res, pattern = '^partition_scores_(du|guo|huang|pan|richoux|dscript)_(both|0)_(0|1)_es.csv', recursive = TRUE),
        '/',
        keep = 2
      )[[1]]
  } else{
    deepFE_results <-
      lapply(paste0(
        deepFE_res,
        list.files(deepFE_res, pattern = '^partition_scores_(du|guo|huang|pan|richoux|dscript)_(both|0)_(0|1).csv', recursive = TRUE)
      ), fread)
    file_names <-
      tstrsplit(
        list.files(deepFE_res, pattern = '^partition_scores_(du|guo|huang|pan|richoux|dscript)_(both|0)_(0|1).csv', recursive = TRUE),
        '/',
        keep = 2
      )[[1]]
  }
  
  names(deepFE_results) <-
    tstrsplit(file_names, '.csv', keep = 1)[[1]]
  deepFE_results <- rbindlist(deepFE_results, idcol = 'filename')
  deepFE_results[, c('dataset', 'train', 'test') := tstrsplit(filename, '_', keep =
                                                                c(3, 4, 5))]
  if (measure == 'Accuracy') {
    wide_df <-
      dcast(deepFE_results[dataset == 'dscript', ],
            filename + dataset + train + test ~ V1,
            value.var = 'Score')
    wide_df[, Balanced_Accuracy := 0.5 * (Recall + Specificity)]
    wide_df <- wide_df[order(filename)]
    deepFE_results <- deepFE_results[order(filename)]
    deepFE_results[dataset == 'dscript' &
                     V1 == 'Accuracy', Score := wide_df$Balanced_Accuracy]
  }
  deepFE_results <- deepFE_results[V1 == measure]
  colnames(deepFE_results) <-
    c('filename', 'Measure', measure, 'Dataset', 'Train', 'Test')
  deepFE_results$Model <- 'DeepFE'
  deepFE_results[, Partition := paste(Train, Test, sep = '->')]
  
  all_results <-
    rbind(all_results, deepFE_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])
  
  # deepPPI
  if (es) {
    deepPPI_results <-
      lapply(paste0(
        deepPPI_res,
        list.files(deepPPI_res, pattern = '(FC|LSTM)_partition_(du|guo|huang|pan|richoux|dscript)_tr(both|0)_te(0|1)_es.csv')
      ), fread)
    file_names <-
      tstrsplit(
        list.files(deepPPI_res, pattern = '(FC|LSTM)_partition_(du|guo|huang|pan|richoux|dscript)_tr(both|0)_te(0|1)_es.csv'),
        '.csv',
        keep = 1
      )[[1]]
  } else{
    deepPPI_results <-
      lapply(paste0(
        deepPPI_res,
        list.files(deepPPI_res, pattern = '(FC|LSTM)_partition_(du|guo|huang|pan|richoux|dscript)_tr(both|0)_te(0|1).csv')
      ), fread)
    file_names <-
      tstrsplit(
        list.files(deepPPI_res, pattern = '(FC|LSTM)_partition_(du|guo|huang|pan|richoux|dscript)_tr(both|0)_te(0|1).csv'),
        '.csv',
        keep = 1
      )[[1]]
  }
  names(deepPPI_results) <- file_names
  deepPPI_results <- rbindlist(deepPPI_results, idcol = 'filename')
  deepPPI_results <-
    deepPPI_results[, c('Model', 'Dataset', 'Train', 'Test') := tstrsplit(filename, '_', keep = c(1, 3, 4, 5))]
  deepPPI_results[, Train := tstrsplit(Train, 'tr', keep = 2)]
  deepPPI_results[, Test := tstrsplit(Test, 'te', keep = 2)]
  deepPPI_results[, Partition := paste(Train, Test, sep = '->')]
  if (measure == 'Accuracy') {
    wide_df <-
      dcast(
        deepPPI_results[Dataset == 'dscript', ],
        filename + Model + Dataset + Train + Test + Partition ~ variable,
        value.var = 'value'
      )
    wide_df[, Balanced_Accuracy := 0.5 * (Recall + Specificity)]
    wide_df <- wide_df[order(filename)]
    deepPPI_results <- deepPPI_results[order(filename)]
    deepPPI_results[Dataset == 'dscript' &
                      variable == 'Accuracy', value := wide_df$Balanced_Accuracy]
  }
  n_train <-
    unique(deepPPI_results[variable %in% c('n_train'), c('Dataset', 'Partition', 'variable', 'value')])
  deepPPI_results <- deepPPI_results[variable == measure]
  deepPPI_results[, Model := paste('deepPPI', Model, sep = '_')]
  colnames(deepPPI_results) <-
    c('filename',
      'variable',
      measure,
      'Model',
      'Dataset',
      'Train',
      'Test',
      'Partition')
  
  all_results <-
    rbind(all_results, deepPPI_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])
  
  # PIPR
  if (es) {
    pipr_results <-
      lapply(paste0(
        seqppi_res,
        list.files(seqppi_res, pattern = '^partition_(du|guo|huang|pan|richoux|dscript)_(both|0)_(0|1)_es.csv')
      ), fread)
    file_names <-
      tstrsplit(
        list.files(seqppi_res, pattern = '^partition_(du|guo|huang|pan|richoux|dscript)_(both|0)_(0|1)_es.csv'),
        '.csv',
        keep = 1
      )[[1]]
  } else{
    pipr_results <-
      lapply(paste0(
        seqppi_res,
        list.files(seqppi_res, pattern = '^partition_(du|guo|huang|pan|richoux|dscript)_(both|0)_(0|1).csv')
      ), fread)
    file_names <-
      tstrsplit(
        list.files(seqppi_res, pattern = '^partition_(du|guo|huang|pan|richoux|dscript)_(both|0)_(0|1).csv'),
        '.csv',
        keep = 1
      )[[1]]
  }
  names(pipr_results) <- file_names
  pipr_results <- rbindlist(pipr_results, idcol = 'filename')
  pipr_results <-
    pipr_results[, c('Dataset', 'Train', 'Test') := tstrsplit(filename, '_', keep =
                                                                c(2, 3, 4))]
  if (measure == 'Accuracy') {
    wide_df <-
      dcast(pipr_results[Dataset == 'dscript', ],
            filename + Dataset + Train + Test ~ V1,
            value.var =
              'Score')
    wide_df[, Balanced_Accuracy := 0.5 * (Recall + Specificity)]
    wide_df <- wide_df[order(filename)]
    pipr_results <- pipr_results[order(filename)]
    pipr_results[Dataset == 'dscript' &
                   V1 == 'Accuracy', Score := wide_df$Balanced_Accuracy]
  }
  pipr_results <- pipr_results[V1 == measure]
  pipr_results$Model <- 'PIPR'
  colnames(pipr_results) <-
    c('filename',
      'Measure',
      measure,
      'Dataset',
      'Train',
      'Test',
      'Model')
  pipr_results[, Partition := paste(Train, Test, sep = '->')]
  
  all_results <-
    rbind(all_results, pipr_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])
  
  # SPRINT
  sprint_results <- fread(paste0(sprint_res, 'all_results.tsv'))
  sprint_results$Model <- 'SPRINT'
  if(measure != 'AUC' & measure != 'AUPR'){
    # take the AUPR
    colnames(sprint_results) <-
      c('Dataset', 'Train', 'Test', 'AUC', measure, 'Model')
  }
  sprint_results[, Partition := paste(Train, Test, sep = '->')]
  
  all_results <-
    rbind(all_results, sprint_results[, c('Model', 'Dataset', measure, 'Partition'), with = FALSE])
  
  # D-Script
  if (es) {
    dscript_results <- fread(paste0(dscript_res, 'all_results_es.tsv'))
  } else{
    dscript_results <- fread(paste0(dscript_res, 'all_results.tsv'))
  }
  
  if (measure == 'Accuracy') {
    wide_df <-
      dcast(dscript_results[Dataset == 'dscript', ], Model + Dataset + Split ~ Metric, value.var =
              'Value')
    wide_df[, Balanced_Accuracy := 0.5 * (Recall + Specificity)]
    wide_df <- wide_df[order(Model, Split)]
    dscript_results <- dscript_results[order(Model, Split)]
    dscript_results[Dataset == 'dscript' &
                      Metric == 'Accuracy', Value := wide_df$Balanced_Accuracy]
  }
  dscript_results <- dscript_results[Metric == measure]
  colnames(dscript_results) <-
    c('Model', 'Dataset', 'Metric', measure, 'Partition')
  dscript_results$Model <- 'D-SCRIPT'
  dscript_results[Dataset == 'richoux_regular', Dataset := 'richoux-regular']
  dscript_results[Dataset == 'richoux_strict', Dataset := 'richoux-strict']
  all_results <-
    rbind(all_results, dscript_results[, c('Model', 'Dataset', measure, 'Partition'), with =
                                         FALSE])
  
  # Topsy_Turvy
  if (es) {
    tt_results <- fread(paste0(tt_res, 'all_results_es.tsv'))
  } else{
    tt_results <- fread(paste0(tt_res, 'all_results.tsv'))
  }
  
  if (measure == 'Accuracy') {
    wide_df <-
      dcast(tt_results[Dataset == 'dscript', ], Model + Dataset + Split ~ Metric, value.var =
              'Value')
    wide_df[, Balanced_Accuracy := 0.5 * (Recall + Specificity)]
    wide_df <- wide_df[order(Model, Split)]
    tt_results <- tt_results[order(Model, Split)]
    tt_results[Dataset == 'dscript' &
                 Metric == 'Accuracy', Value := wide_df$Balanced_Accuracy]
  }
  tt_results <- tt_results[Metric == measure]
  colnames(tt_results) <-
    c('Model', 'Dataset', 'Metric', measure, 'Partition')
  tt_results$Model <- 'Topsy_Turvy'
  tt_results[Dataset == 'richoux_regular', Dataset := 'richoux-regular']
  tt_results[Dataset == 'richoux_strict', Dataset := 'richoux-strict']
  all_results <-
    rbind(all_results, tt_results[, c('Model', 'Dataset', measure, 'Partition'), with =
                                    FALSE])
  
  # visualization
  all_results <- all_results[, Dataset := factor(Dataset,
                                                 levels = c("huang", "guo", "du", "pan", "richoux", "dscript"))]
  
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
           file = paste0('results/partition_', measure, '_es.csv'))
  } else{
    fwrite(all_results, file = paste0('results/partition_', measure, '.csv'))
  }
}
