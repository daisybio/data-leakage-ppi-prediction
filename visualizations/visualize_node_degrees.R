library(data.table)
library(ggplot2)
library(RColorBrewer)

all_degrees <- fread('../algorithms/SPRINT/data/node_degrees.csv')
all_degrees$Test <- factor(all_degrees$Test, levels = c('original', 'rewired', 'partition'))
all_degrees$Split <- gsub('both', 'inter', all_degrees$Split)
all_degrees$Split <- gsub('0', 'intra-0', all_degrees$Split)
all_degrees$Split <- gsub('1', 'intra-1', all_degrees$Split)
all_degrees$Split <- gsub('2', 'intra-2', all_degrees$Split)

ggplot(all_degrees, aes(x=Degree, fill=Network))+
  geom_histogram(bins=30, position = 'dodge')+
  facet_wrap(Test~Dataset, scales = 'free', nrow = 3)+
  theme_bw()+
  xlim(0, 20)

ggsave('plots/node_degrees_pos_vs_neg.png', height = 6, width=12)

all_degrees$Split <- factor(all_degrees$Split, levels = c('train', 'test', 'inter', 'intra-0', 'intra-1', 'intra-2'))
ggplot(all_degrees, aes(x=Degree, fill=Split))+
  geom_histogram(bins=30, position = 'dodge')+
  facet_wrap(Test~Dataset, scales = 'free', nrow = 3)+
  theme_bw()+
  scale_fill_manual(values = brewer.pal(6, "Set1"))+
  xlim(0, 20)

ggsave('plots/node_degrees_train_vs_test.png', height = 6, width=12)

orig_vs_rewired <- merge(all_degrees[Test == 'original'], all_degrees[Test == 'rewired'], by=c('Node', 'Dataset', 'Split', 'Network'))
colnames(orig_vs_rewired) <- c('Node', 'Dataset', 'Split', 'Network', 'Degree original', 'Original', 'Degree rewired', 'Rewired')
ggplot(orig_vs_rewired, aes(x=`Degree original`, y=`Degree rewired`))+
  geom_point()+
  theme_bw() +
  geom_abline(intercept = 0, slope=1, color='red')+
  xlim(0,265)+
  ylim(0,265)

ggsave('plots/node_degrees_original_vs_rewired.png', height = 6, width=6)
