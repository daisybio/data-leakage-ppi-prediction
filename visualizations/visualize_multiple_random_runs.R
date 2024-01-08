library(data.table)
library(ggplot2)
library(xtable)

# Generate performance plots for 10 runs with different random split, original and rewired

# Custom
# read in data
folder <- '../algorithms/Custom/results/multiple_runs/'
# read all files in directory
files <- list.files(path=folder, pattern="*.csv", full.names=TRUE)
names_files <- tstrsplit(list.files(path=folder, pattern="*.csv"), '.csv', keep=1)[[1]]
# read all files in list
data <- lapply(files, fread)
names(data) <- names_files
# combine all data 
data <- rbindlist(data, idcol = "id")
# substitute all _RF and _SVM in id with -RF and -SVM
data[, id := gsub("_RF", "-RF", id)]
data[, id := gsub("_SVM", "-SVM", id)]
# split id column
data[, c('Test', 'Dataset', 'Model', 'Seed') := tstrsplit(id, '_')]
colnames(data) <- c('id', 'Performance Measure', 'Value', 'Test', 'Dataset', 'Model', 'Seed')
# rename Performance Measure Sensitivity to Recall
data[, `Performance Measure` := gsub("Sensitivity", "Recall", `Performance Measure`)]
# rename Model cons and hf to Global and Local Cons. and Harmonic Function
data[, Model := gsub("cons", "Global and Local Cons.", Model)]
data[, Model := gsub("hf", "Harmonic Function", Model)]


# Richoux_FC + Richoux_LSTM
# read in data
folder <- '../algorithms/DeepPPI/keras/results_custom'
# read all files in directory
files <- list.files(path=folder, pattern="*(17612|29715|30940|31191|42446|50495|60688|7413|75212|81645).csv", full.names=TRUE)
names_files <- tstrsplit(list.files(path=folder, pattern="*(17612|29715|30940|31191|42446|50495|60688|7413|75212|81645).csv"), '.csv', keep=1)[[1]]
# read all files in list
data2 <- lapply(files, fread)
names(data2) <- names_files
# combine all data
data2 <- rbindlist(data2, idcol = "id")
data2 <- data2[, c('Model', 'Test', 'Dataset', 'Seed') := tstrsplit(id, '_')]
data2[, Model := paste('Richoux', Model, sep = '_')]
data2 <- data2[variable %in% c('Accuracy', 'AUC', 'AUPR', 'F1', 'MCC', 'Precision', 'Recall', 'Specificity')]
colnames(data2) <- c('id', 'Performance Measure', 'Value', 'Model', 'Test', 'Dataset', 'Seed')

# SPRINT
# read in data
data3 <- fread('../algorithms/SPRINT/results/multiple_runs/all_results.tsv', sep = '\t')
# split dataset column
data3[, c('Test', 'Dataset', 'Seed') := tstrsplit(Dataset, '_')]
data3$Model <- 'SPRINT'
# wide to long: make AUC and AUPR columns into two columns: performance measure and value
data3 <- melt(data3, id.vars = c('Test', 'Dataset', 'Model', 'Seed'), measure.vars = c('AUC', 'AUPR'), variable.name = 'Performance Measure', value.name = 'Value')

# DeepFE
# read in data
folder <- '../algorithms/DeepFE-PPI/result/multiple_runs/'
# read all files in directory
files <- list.files(path=folder, pattern="*.csv", full.names=TRUE)
names_files <- tstrsplit(list.files(path=folder, pattern="*.csv"), '.csv', keep=1)[[1]]
# read all files in list
data4 <- lapply(files, fread)
names(data4) <- names_files
# combine all data
data4 <- rbindlist(data4, idcol = "id")
data4[, c('Test', 'scores', 'Dataset', 'Seed') := tstrsplit(id, '_')]
data4 <- data4[, -c('scores')]
data4$Model <- 'DeepFE'
colnames(data4) <- c('id', 'Performance Measure', 'Value', 'Test', 'Dataset', 'Seed', 'Model')

# PIPR
# read in data
folder <- '../algorithms/seq_ppi/binary/model/lasagna/results/multiple_runs/'
# read all files in directory
files <- list.files(path=folder, pattern="*.csv", full.names=TRUE)
names_files <- tstrsplit(list.files(path=folder, pattern="*.csv"), '.csv', keep=1)[[1]]
# read all files in list
data5 <- lapply(files, fread)
names(data5) <- names_files
# combine all data
data5 <- rbindlist(data5, idcol = "id")
data5[, c('Test', 'Dataset', 'Seed') := tstrsplit(id, '_')]
data5$Model <- 'PIPR'
colnames(data5) <- c('id', 'Performance Measure', 'Value', 'Test', 'Dataset', 'Seed', 'Model')

# D-SCRIPT
data6 <- fread('../algorithms/D-SCRIPT-main/results_dscript/multiple_runs/all_results.tsv', sep = '\t')
data6$Model <- 'D-SCRIPT'
data6 <- data6[!Metric %in% c('Sensitivity', 'TP', 'FP', 'TN', 'FN')]
colnames(data6) <- c('Model', 'Dataset', 'Seed', 'Performance Measure', 'Value', 'Test')

# Topsy-Turvy
data7 <- fread('../algorithms/D-SCRIPT-main/results_topsyturvy/multiple_runs/all_results.tsv', sep = '\t')
data7$Model <- 'Topsy-Turvy'
data7 <- data7[!Metric %in% c('Sensitivity', 'TP', 'FP', 'TN', 'FN')]
colnames(data7) <- c('Model', 'Dataset', 'Seed', 'Performance Measure', 'Value', 'Test')

# combine all data
all_results <- rbindlist(list(data[, -c("id")], data2[, -c("id")], data3, data4[, -c("id")], data5[, -c("id")], data6, data7), use.names=TRUE)
# make dataset names all cap
all_results[, Dataset := toupper(Dataset)]
# order model column
all_results[, Model := factor(Model, levels = c('PCA-RF', 'MDS-RF', 'node2vec-RF', 'PCA-SVM', 'MDS-SVM', 'node2vec-SVM', 'Global and Local Cons.', 'Harmonic Function', 'SPRINT', 'Richoux_FC', 'Richoux_LSTM', 'DeepFE', 'PIPR', 'D-SCRIPT', 'Topsy-Turvy'))]

colors <- c('#e6194b', '#f032e6', '#ffe119', '#4363d8', '#f58231', '#911eb4',
            '#3cb44b','#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#aaffc3', '#808000', '#000075')

# Visualize
ggplot(all_results[Test == 'original'], aes(x = Model, y = Value, color = Model)) +
  geom_point() +
  facet_grid(`Performance Measure` ~ Dataset, scales = 'free') +
  scale_color_manual(values = colors)+
  theme_bw()+
  # no labels on x axis ticks
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
#ggsave('plots/boxplot_multiple_runs_original.pdf', width=8, height=10)

# table with average performance measure and standard deviation per model and dataset
result_table <- all_results[`Performance Measure` == 'Accuracy' & Test == 'original', .(mean = round(mean(Value), 4), sd = round(sd(Value), 4)), by = .(Model, Dataset, `Performance Measure`)]
result_table <- result_table[order(Dataset, -mean)]
result_table[, `Performance Measure` := NULL]
# to latex table
print(xtable(result_table, caption = 'Mean and standard deviation over the ten accuracies obtained on the ten different random splits of the original datasets. Results are ranked by their mean, the best mean performance and largest standard deviation per dataset are indicated in bold.', label = 'tab:avg_acc_original_multiple_runs', digits = 4), include.rownames = F)

ggplot(all_results[Test == 'rewired'], aes(x = Model, y = Value, color = Model)) +
  geom_point() +
  facet_grid(`Performance Measure` ~ Dataset, scales = 'free') +
  scale_color_manual(values = colors)+
  theme_bw()+
  # no labels on x axis ticks
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
#ggsave('plots/boxplot_multiple_runs_rewired.pdf', width=8, height=10)

# table with average performance measure and standard deviation per model and dataset
result_table <- all_results[`Performance Measure` == 'Accuracy' & Test == 'rewired', .(mean = round(mean(Value), 4), sd = round(sd(Value), 4)), by = .(Model, Dataset, `Performance Measure`)]
result_table <- result_table[order(Dataset, -mean)]
result_table[, `Performance Measure` := NULL]
# to latex table
print(xtable(result_table, caption = 'Mean and standard deviation over the ten accuracies obtained on the ten different random splits of the rewired datasets. Results are ranked by their mean, the best mean performance and largest standard deviation per dataset are indicated in bold.', label = 'tab:avg_acc_rewired_multiple_runs', digits = 4), include.rownames = F)


# old robustness tests
robustness_tests <- fread('../algorithms/D-SCRIPT-main/results_dscript/robustness_results.tsv', sep = '\t')
robustness_tests[, Dataset := toupper(Dataset)]
robustness_tests[, Dataset := factor(Dataset, levels = c('HUANG', 'GUO', 'DU', 'PAN', 'RICHOUX_REGULAR'))]
ggplot(robustness_tests[!Metric %in% c('Sensitivity', 'TP', 'FP', 'TN', 'FN')], aes(x = Dataset, y = Value, color = Metric)) +
  geom_point()+
  scale_color_manual(values = colors)+
  facet_wrap(~Metric, nrow=2)+
  theme_bw()+
  theme(text = element_text(size=20), axis.text.x = element_text(angle = 45, hjust = 1))
