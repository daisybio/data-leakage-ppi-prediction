library(data.table)
library(ggplot2)
library(RColorBrewer)

#### result prefixes
custom_res <- '../algorithms/Custom/results/'
deepFE_res <- '../algorithms/DeepFE-PPI/result/custom/'
deepPPI_res <- '../algorithms/DeepPPI/keras/results_custom/'
seqppi_res <- '../algorithms/seq_ppi/binary/model/lasagna/results/'
sprint_res <- '../algorithms/SPRINT/results/'

#### time files
all_times <- data.table(Test = character(), Model=character(), Dataset=character(), `Time [s]`=numeric())
# custom
custom_times <- lapply(paste0(custom_res, list.files(custom_res, pattern='time*')), fread)
file_names <- tstrsplit(list.files(custom_res, pattern='time*'), '.txt', keep=1)[[1]]
file_names[grepl('richoux_regular', file_names, fixed=TRUE)] <- gsub('richoux_regular', 'richoux-regular', file_names[grepl('richoux_regular', file_names, fixed=TRUE)])
file_names[grepl('richoux_strict', file_names, fixed=TRUE)] <- gsub('richoux_strict', 'richoux-strict', file_names[grepl('richoux_strict', file_names, fixed=TRUE)])
file_names <- tstrsplit(file_names, '_', keep=c(2, 3, 4))
names(custom_times) <- paste(file_names[[1]], file_names[[2]], file_names[[3]], sep='_')
custom_times_df <- rbindlist(custom_times[!grepl('partition', names(custom_times), fixed=TRUE)], idcol = 'filename')
partition_times_df <- rbindlist(custom_times[grepl('partition', names(custom_times), fixed=TRUE)], idcol = 'filename')
custom_times_df <- custom_times_df[, c('test', 'dataset', 'encoding') := tstrsplit(filename, '_')]
partition_times_df <- partition_times_df[, c('test', 'dataset', 'encoding') := tstrsplit(filename, '_')]
names(custom_times_df) <- c('Filename', 'Algorithm', 'Time [s]', 'Test', 'Dataset', 'Encoding')
names(partition_times_df) <- c('Filename', 'Part_Train', 'Part_Test', 'Algorithm', 'Time [s]', 'Test', 'Dataset', 'Encoding')
custom_times_df$Part_Train <- NA
custom_times_df$Part_Test <- NA
custom_times_df[, Model := paste(Algorithm, Encoding, sep='_')]
partition_times_df[, Model := paste(Algorithm, Encoding, Part_Train, Part_Test, sep='_')]
all_times_custom <- rbind(custom_times_df, partition_times_df)
fwrite(all_times_custom, '../algorithms/Custom/results/run_t.csv')

all_times <- rbind(all_times, all_times_custom[, c('Test', 'Model', 'Dataset','Time [s]')])

# deepFE
deepFE_times <- lapply(paste0(deepFE_res, list.files(deepFE_res, pattern = 'time*', recursive = TRUE)), fread)
file_names <- tstrsplit(basename(list.files(deepFE_res, pattern = 'time*', recursive = TRUE)), '.txt', keep=1)[[1]]
file_names[grepl('richoux_regular', file_names, fixed=TRUE)] <- gsub('richoux_regular', 'richoux-regular', file_names[grepl('richoux_regular', file_names, fixed=TRUE)])
file_names[grepl('richoux_strict', file_names, fixed=TRUE)] <- gsub('richoux_strict', 'richoux-strict', file_names[grepl('richoux_strict', file_names, fixed=TRUE)])
names(deepFE_times) <- file_names 
deepFE_times <- rbindlist(deepFE_times, idcol='Filename')
deepFE_times <- deepFE_times[, c('Test', 'Dataset', 'Part_Train', 'Part_Test') := tstrsplit(Filename, '_', keep=c(2,3,4,5), fill = NA)]
deepFE_times <- deepFE_times[, Model := ifelse(is.na(Part_Train),
                                               'DeepFE',
                                               paste('DeepFE', Part_Train, Part_Test, sep='_'))]
colnames(deepFE_times) <-c('Filename', 'Time [s]', 'Test', 'Dataset', 'Part_Train', 'Part_Test', 'Model')
fwrite(deepFE_times, '../algorithms/DeepFE-PPI/result/custom/run_t.csv')

all_times <- rbind(all_times, deepFE_times[, c('Test', 'Model', 'Dataset', 'Time [s]')])

# deepPPI
deepPPI_times <- fread(paste0(deepPPI_res, 'all_times.txt'))
colnames(deepPPI_times) <- c('Filename', 'Time [s]')
deepPPI_times$Filename[grepl('richoux_regular', deepPPI_times$Filename, fixed=TRUE)] <- gsub('richoux_regular', 'richoux-regular', deepPPI_times$Filename[grepl('richoux_regular', deepPPI_times$Filename, fixed=TRUE)])
deepPPI_times$Filename[grepl('richoux_strict', deepPPI_times$Filename, fixed=TRUE)] <- gsub('richoux_strict', 'richoux-strict', deepPPI_times$Filename[grepl('richoux_strict', deepPPI_times$Filename, fixed=TRUE)])
deepPPI_times <- deepPPI_times[, c('Model', 'Test', 'Dataset', 'Part_Train', 'Part_Test') := tstrsplit(Filename, '_', fill=NA)]
deepPPI_times <- deepPPI_times[, Part_Train := tstrsplit(Part_Train, 'tr', keep = 2)]
deepPPI_times <- deepPPI_times[, Part_Test := tstrsplit(Part_Test, 'te', keep = 2)]
deepPPI_times <- deepPPI_times[, Model := ifelse(is.na(Part_Train), 
                              paste0('DeepPPI_', Model),
                              paste('DeepPPI', Model, Part_Train, Part_Test, sep='_'))]

all_times <- rbind(all_times, deepPPI_times[,  c('Test', 'Model', 'Dataset', 'Time [s]')])

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
sprint_times <- lapply(paste0(sprint_res, list.files(sprint_res, pattern = '*time*', recursive = TRUE)), function(x){
  tmp <- fread(x, header=FALSE)
  return(tmp[1,2])
})
file_names <- tstrsplit(list.files(sprint_res, pattern = '*time*', recursive = TRUE), '_time.txt', keep=1)[[1]]
file_names[grepl('richoux_regular', file_names, fixed=TRUE)] <- gsub('richoux_regular', 'richoux-regular', file_names[grepl('richoux_regular', file_names, fixed=TRUE)])
file_names[grepl('richoux_strict', file_names, fixed=TRUE)] <- gsub('richoux_strict', 'richoux-strict', file_names[grepl('richoux_strict', file_names, fixed=TRUE)])
names(sprint_times) <- file_names
sprint_times <- rbindlist(sprint_times, idcol='Filename')
sprint_times[, c('min', 'Time [s]') := tstrsplit(V2, 'm')]
sprint_times[, `Time [s]` := tstrsplit(`Time [s]`, 's', keep=1)]
sprint_times[, min := as.numeric(min) * 60]
sprint_times[, `Time [s]` := as.numeric(`Time [s]`) + min]
sprint_times[, c('Test', 'Filename') := tstrsplit(Filename, '/')]
sprint_times[, c('Dataset', 'Part_Train', 'Part_Test') := tstrsplit(Filename, '_', keep=c(1,3,5), fill=NA)]
sprint_times$Test[sprint_times$Test == 'partitions'] <- 'partition' 
sprint_times <- sprint_times[, Model := ifelse(is.na(Part_Train),
                                               'SPRINT',
                                               paste('SPRINT', Part_Train, Part_Test, sep='_'))]
fwrite(sprint_times, '../algorithms/SPRINT/results/run_t.csv')
all_times <- rbind(all_times, sprint_times[,  c('Test', 'Model', 'Dataset', 'Time [s]')])

all_times <- all_times[, Dataset := factor(Dataset, 
                                           levels = c("huang", "guo", "du", "pan", "richoux-regular", "richoux-strict"))]


# visualization
ggplot(all_times[Test != 'partition'], aes(x=Dataset, y = `Time [s]`, color = Model, group=Model))+
  geom_point(size=3)+
  geom_line(size=2, alpha=0.5)+
  facet_wrap(~Test)+
  scale_x_discrete(labels=c("huang" = "Huang (4,242)", "guo" = "Guo (7,656)",
                            "du" = "Du (24,478)", "pan" = "Pan (38,956)",
                            "richoux-regular" = "Richoux regular (67,364)",
                            "richoux-strict" = "Richoux strict (68,052)"),
                   )+
  labs(x = "Dataset (n training)", y = "Time [s]") +
  geom_hline(yintercept = 1800, color='red') +
  geom_hline(yintercept = 3600, color='red') +
  #geom_hline(yintercept = 7200, color='red') +
  geom_text(aes(0, 1800, label = '30 min', vjust = -1, hjust=0), color='red') +
  geom_text(aes(0, 3600, label = '1 h', vjust = -1, hjust=0), color='red') +
  #geom_text(aes(0, 7200, label = '2 h', vjust = -1, hjust=0), color='red') +
  #scale_color_manual(values = brewer.pal(12, "Paired")[-11])+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))

ggplot(all_times[Test == 'partition'], aes(x=Dataset, y = `Time [s]`, color = Model, group=Model))+
  geom_point(size=3)+
  geom_line(size=2, alpha=0.5)+
  labs(x = "Dataset (n training)", y = "Time [s]") +
  geom_hline(yintercept = 1800, color='red') +
  geom_hline(yintercept = 3600, color='red') +
  #geom_hline(yintercept = 7200, color='red') +
  geom_text(aes(0, 1800, label = '30 min', vjust = -1, hjust=0), color='red') +
  geom_text(aes(0, 3600, label = '1 h', vjust = -1, hjust=0), color='red') +
  #geom_text(aes(0, 7200, label = '2 h', vjust = -1, hjust=0), color='red') +
  #scale_color_manual(values = brewer.pal(12, "Paired")[-11])+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))
ggsave("./all_times.png",height=8, width=12)

ggplot(all_times[Model != "PIPR"], aes(x=dataset, y = Time, color = Model, group=Model))+
  geom_line(size=2, alpha=0.5)+
  geom_point(size=3)+
  scale_x_discrete(labels=c("huang" = "Huang (4,242)", "guo" = "Guo (7,656)",
                            "du" = "Du (24,478)", "richoux-regular" = "Richoux regular (33,682)",
                            "richoux-strict" = "Richoux strict (34,026)", "pan" = "Pan (38,956)"),
  )+
  labs(x = "Dataset (n training)", y = "Time [s]") +
  geom_hline(yintercept = 1800, color='red') +
  geom_text(aes(0, 1800, label = '30 min', vjust = -1, hjust=0), color='red') +
  scale_color_manual(values = brewer.pal(12, "Paired")[-11])+
  theme_bw()+
  theme(text = element_text(size=20),axis.text.x = element_text(angle = 45, vjust = 0.5, hjust=0.5))
ggsave("./all_times_wo_PIPR.png",height=8, width=12)

