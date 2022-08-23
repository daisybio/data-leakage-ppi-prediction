library(data.table)
library(ggplot2)
library(RColorBrewer)

#### result prefixes
custom_res <- '../algorithms/Custom/results/'
deepFE_res <- '../algorithms/DeepFE-PPI/result/custom/'
deepPPI_res <- '../algorithms/DeepPPI/keras/results_custom/'
seqppi_res <- '../algorithms/seq_ppi/binary/model/lasagna/results/'
sprint_res <- '../algorithms/SPRINT/results/rewired/'

#### time files
all_times <- data.table(Model=character(), dataset=character(), Time=numeric())
# custom
custom_times <- lapply(paste0(custom_res, list.files(custom_res, pattern='time*')), fread)
file_names <- tstrsplit(list.files(custom_res, pattern='time*'), '.txt', keep=1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
file_names <- tstrsplit(file_names, '_', keep=c(3, 4))
names(custom_times) <- paste(file_names[[1]], file_names[[2]], sep='_')
custom_times <- rbindlist(custom_times, idcol = 'filename')
custom_times <- custom_times[, c('dataset', 'encoding') := tstrsplit(filename, '_')]
names(custom_times) <- c('filenames', 'algorithm', 'Time', 'dataset', 'encoding')
custom_times[, Model := paste(algorithm, encoding, sep='_')]

all_times <- rbind(all_times, custom_times[, c('Model', 'dataset', 'Time')])

# deepFE
deepFE_times <- lapply(paste0(deepFE_res, list.files(deepFE_res, pattern = 'time*', recursive = TRUE)), fread)
names(deepFE_times) <- list.files(deepFE_res)[-5]
deepFE_times <- rbindlist(deepFE_times, idcol='dataset')
deepFE_times$Model <- 'DeepFE'
colnames(deepFE_times) <- c('dataset', 'Time', 'Model') 
deepFE_times[dataset == 'richoux_regular', 'dataset'] <- 'richoux-regular'
deepFE_times[dataset == 'richoux_strict', 'dataset'] <- 'richoux-strict'

all_times <- rbind(all_times, deepFE_times[, c('Model', 'dataset', 'Time')])

# deepPPI
deepPPI_times <- lapply(paste0(deepPPI_res, list.files(deepPPI_res, pattern = 'time*')), fread)
file_names <- tstrsplit(list.files(deepPPI_res, pattern = 'time*'), '.txt', keep=1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
file_names <- tstrsplit(file_names, '_', keep=c(2, 4))
names(deepPPI_times) <- paste(file_names[[1]], file_names[[2]], sep='_')
deepPPI_times <- rbindlist(deepPPI_times, idcol = 'filename')
deepPPI_times <- deepPPI_times[, c('Model', 'dataset') := tstrsplit(filename, '_')]
names(deepPPI_times) <- c('filenames', 'Time', 'Model', 'dataset')
deepPPI_times <- deepPPI_times[, Model := paste('deepPPI', Model, sep='_')]

all_times <- rbind(all_times, deepPPI_times[, c('Model', 'dataset', 'Time')])

# seq_ppi
seq_ppi_times <-  lapply(paste0(seqppi_res, list.files(seqppi_res, pattern = 'time*')), fread)
file_names <- tstrsplit(list.files(seqppi_res, pattern = 'time*'), '.txt', keep=1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
names(seq_ppi_times) <- tstrsplit(file_names, '_', keep=3)[[1]]
seq_ppi_times <- rbindlist(seq_ppi_times, idcol = 'filename')
seq_ppi_times$Model <- 'PIPR'
colnames(seq_ppi_times) <- c('dataset', 'Time', 'Model') 

all_times <- rbind(all_times, seq_ppi_times[, c('Model', 'dataset', 'Time')])

# SPRINT
sprint_times <- lapply(paste0(sprint_res, list.files(sprint_res, pattern = '*time*')), function(x){
  tmp <- fread(x, header=FALSE)
  return(tmp[, ncol(tmp)-7, with=FALSE])
})
file_names <- tstrsplit(list.files(sprint_res, pattern = '*time*'), '.txt', keep=1)[[1]]
file_names[grepl('richoux', file_names, fixed=TRUE)] <- gsub('richoux_*', 'richoux-', file_names[grepl('richoux', file_names, fixed=TRUE)])
names(sprint_times) <- tstrsplit(file_names, '_', keep=1)[[1]]
sprint_times <- rbindlist(sprint_times, idcol = 'filename')
sprint_times$Model <- 'SPRINT'
colnames(sprint_times) <- c('dataset', 'Time', 'Model') 
sprint_times[, Time := as.numeric(tstrsplit(Time, 's', keep=1)[[1]])]

all_times <- rbind(all_times, sprint_times[, c('Model', 'dataset', 'Time')])
all_times <- all_times[, Model := factor(Model, 
                                         levels=c("RF_PCA","SVM_PCA", "RF_MDS", "SVM_MDS",
                                                  "RF_node2vec",  "SVM_node2vec", "SPRINT", 
                                                  "deepPPI_FC", "deepPPI_LSTM",  
                                                  "DeepFE", "PIPR"))]
all_times <- all_times[, dataset := factor(dataset, 
                                           levels = c("huang", "guo", "du", "richoux-regular", "richoux-strict", "pan"))]


# visualization
ggplot(all_times, aes(x=dataset, y = Time, color = Model, group=Model))+
  geom_line(size=2, alpha=0.5)+
  geom_point(size=3)+
  scale_x_discrete(labels=c("huang" = "Huang (4,242)", "guo" = "Guo (7,656)",
                            "du" = "Du (24,478)", "richoux-regular" = "Richoux regular (33,682)",
                            "richoux-strict" = "Richoux strict (34,026)", "pan" = "Pan (38,956)"),
                   )+
  labs(x = "Dataset (n training)", y = "Time [s]") +
  geom_hline(yintercept = 1800, color='red') +
  geom_hline(yintercept = 3600, color='red') +
  geom_hline(yintercept = 7200, color='red') +
  geom_text(aes(0, 1800, label = '30 min', vjust = -1, hjust=0), color='red') +
  geom_text(aes(0, 3600, label = '1 h', vjust = -1, hjust=0), color='red') +
  geom_text(aes(0, 7200, label = '2 h', vjust = -1, hjust=0), color='red') +
  scale_color_manual(values = brewer.pal(12, "Paired")[-11])+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))
ggsave("./all_times.png",height=8, width=12)

ggplot(all_times[Model != "PIPR"], aes(x=dataset, y = Time, color = Model, group=Model))+
  geom_line(size=2, alpha=0.5)+
  geom_point(size=3)+
  scale_x_discrete(labels=c("huang" = "Huang (4,242)", "guo" = "Guo (7,656)",
                            "du" = "Du (24,478)", "richoux-regular" = "Richoux regular (33,682)",
                            "richoux-strict" = "Richoux strict (34,026)", "pan" = "Pan (38,956)")
  )+
  labs(x = "Dataset (n training)", y = "Time [s]") +
  geom_hline(yintercept = 1800, color='red') +
  geom_text(aes(0, 1800, label = '30 min', vjust = -1, hjust=0), color='red') +
  scale_color_manual(values = brewer.pal(12, "Paired")[-11])+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))
ggsave("./all_times_wo_PIPR.png",height=8, width=12)

