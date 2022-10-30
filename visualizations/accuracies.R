library(data.table)
library(ggplot2)

all_accuracies <- data.table(name=rep('guo_2008', 3), ml=rep('SVM', 3), strategy='classical_ML', accuracy=c(0.87, 0.58, 0.86), dataset=c('guo', 'own', 'own'))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('guo_2010', 6), ml=rep('SVM', 6), strategy='classical_ML', accuracy=c(0.89, 0.91, 0.93, 0.90, 0.978, 0.94), dataset=c('guo', 'pan', 'DIP_E', 'DIP_D', 'DIP_C', 'other')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('pan_2010', 1), ml='RF', strategy='classical_ML', accuracy=c(0.98), dataset='pan'))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('sun_2017', 7), ml=rep('Autoencoder', 7), strategy='Deep_Learning', accuracy=c(0.97, 0.5, 0.96, 0.98, 0.97, 0.9, 0.99), dataset=c('pan', 'martin', 'DIP_E', 'DIP_D', 'DIP_C', 'other', 'other')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('chen_2019', 1), ml=c('CNN + bi-GRU'), strategy='Deep_Learning', accuracy=rep(0.97), dataset=rep('guo', 1)))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('wang_2019', 2), ml=rep('CNN + Rotation Forest', 2), strategy='Deep_Learning', accuracy=c(0.98,0.89), dataset=c('guo', 'martin')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('xu_2020', 3), ml=rep('WSRC', 3), strategy='classical_ML', accuracy=c(1.0, 0.97, 0.99), dataset=c('guo', 'martin', 'huang')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('wang_2017', 6), ml=rep('FC', 6), strategy='Deep_Learning', accuracy=c(0.87, 0.95, 0.93, 0.94, 0.93, 0.93), dataset=c('martin', 'DIP_E', 'DIP_C', 'DIP_H', 'DIP_M', 'du')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('you_2015', 7), ml=rep('RF', 7), strategy='classical_ML', accuracy=c(0.95, 0.88, 0.89, 0.88, 0.94, 0.92, 0.91), dataset=c('guo', 'martin', 'DIP_E', 'DIP_C', 'DIP_H', 'DIP_M', 'DIP_P')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('hu_2015', 2), ml=rep('score', 2), strategy='scoring', accuracy=c(0.62, 0.68), dataset=c('guo', 'pan')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('du_2017', 9), ml=rep('FC', 9), strategy='Deep_Learning', accuracy=c(0.86, 0.92, 0.95, 0.94, 0.91, 0.98, 0.93, 0.79, 0.91), dataset=c('martin', 'DIP_E', 'DIP_C', 'DIP_H', 'DIP_M', 'huang', 'du', 'other', 'other')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('yao_2019', 8), ml=rep('FC', 8), strategy='Deep_Learning', accuracy=c(0.95, 1, 1, 1, 1, 0.99, 0.73, 0.94), dataset=c('guo', 'DIP_E', 'DIP_C', 'DIP_H', 'DIP_M', 'huang', 'other', 'other')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('jha_2020', 2), ml=rep('LSTM + Encoder/Decoder', 2), strategy='Deep_Learning', accuracy=c(0.97, 0.94), dataset=c('pan', 'du')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('saha_2014', 2), ml=rep('Ensemble: SVM, RF, DT, NB', 2), strategy='classical_ML', accuracy=c(0.68, 0.91), dataset=c('own', 'own')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('chen_20192', 8), ml=rep('Ensemble: RF, NB, NN, KNN, SVM', 8), strategy='classical_ML', accuracy=c(0.84, 0.98, 0.95, 0.98, 0.99, 0.92, 0.78, 0.94), dataset=c('guo', 'DIP_H', 'DIP_E', 'DIP_D', 'DIP_C', 'du', 'other', 'other')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('zhao_2020', 4), ml=rep('CNN + bi-GRU + Attention', 4), strategy='Deep_Learning', accuracy=c(0.97, 0.9, 0.88, 0.96), dataset=c('guo', 'du', 'other', 'other')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('hashemifar_2018', 5), ml=rep('CNN', 5), strategy='Deep_Learning', accuracy=c(0.95, 0.96, 0.97, 0.96, 0.96), dataset=c('guo', 'DIP_H', 'DIP_E', 'DIP_C', 'DIP_M')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('maetschke_2021', 3), ml=rep('RF', 3), strategy='classical_ML', accuracy=c(0.9, 0.73, 0.93), dataset=c('guo', 'other', 'other')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('shen_2007', 1), ml='SVM', strategy='classical_ML', accuracy=c(0.84), dataset='own'))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('mahapatra_2020', 1), ml='SVM', strategy='classical_ML', accuracy=c(0.99), dataset='own'))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('wang_2018', 6), ml=rep('Rotation Forest', 6), strategy='classical_ML', accuracy=c(0.97, 0.88, 0.92, 1.0, 0.91, 0.91), dataset=c('guo', 'martin', 'DIP_H', 'DIP_E', 'DIP_C', 'DIP_M')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('hamp_2015', 2), ml=rep('SVM', 2), strategy='classical_ML', accuracy=c(0.67, 0.87), dataset=c('own', 'own')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('li_2017', 4), ml=rep('score', 4), strategy='scoring', accuracy=c(0.74, 0.93, 0.61, 0.82), dataset=c('own', 'own', 'other', 'other')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('ieremie_2022', 2), ml=rep('Transformer', 2),strategy='Deep_Learning',  accuracy=c(0.91, 0.97), dataset=c('other', 'other')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('ding_2016', 7), ml=rep('RF', 7), strategy='classical_ML', accuracy=c(0.95, 0.88, 0.94, 0.93, 0.92, 0.96, 0.98), dataset=c('guo', 'martin', 'DIP_H', 'DIP_E', 'DIP_C', 'DIP_M', 'huang')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('richoux_FC_2019', 2), ml=c('FC'), strategy='Deep_Learning', accuracy=c(0.9, 0.76), dataset=c('richoux_reg', 'richoux_strict')))
all_accuracies <- rbind(all_accuracies, data.table(name = rep('richoux_LSTM_2019', 2), ml=c('LSTM'), strategy='Deep_Learning', accuracy=c(0.86, 0.78), dataset=c('richoux_reg', 'richoux_strict')))

ggplot(all_accuracies[strategy !='scoring' & !dataset %in% c('own', 'other', 'richoux_reg', 'richoux_strict', 'DIP_P')], aes(x = dataset, y = accuracy, fill=strategy))+
  geom_boxplot()+
  labs(x = 'Dataset', y = 'Accuracy')+
  scale_fill_manual(values = c('classical_ML' = '#E69F00', 'Deep_Learning' = '#56B4E9'), labels = c('Classical ML', 'Deep Learning'), name = 'Strategy')+
  theme(text = element_text(size=20))+
  theme_bw()
ggsave('accuracies_ml_vs_dl.png', height=3, width=8)

ggplot(all_accuracies[!dataset %in% c('own', 'other', 'richoux_reg', 'richoux_strict', 'DIP_P')], aes(x = dataset, y = accuracy, color=ml))+
  geom_point(size=2)+
  facet_wrap(~ml)+
  theme(text = element_text(size=20))+
  theme_bw()

