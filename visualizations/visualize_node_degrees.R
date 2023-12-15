library(data.table)
library(ggplot2)
library(RColorBrewer)
library(latex2exp)

all_degrees <- fread('../algorithms/SPRINT/data/node_degrees.csv')
all_degrees$Test <- factor(all_degrees$Test, levels = c('original', 'rewired', 'partition'))
all_degrees$Split <- gsub('both', 'inter', all_degrees$Split)
all_degrees$Split <- gsub('0', 'intra-0', all_degrees$Split)
all_degrees$Split <- gsub('1', 'intra-1', all_degrees$Split)
all_degrees$Split <- gsub('2', 'intra-2', all_degrees$Split)
all_degrees[, Dataset := stringr::str_to_upper(Dataset)]
all_degrees[Dataset == 'GOLD_STANDARD', Dataset := 'GOLD STANDARD']
all_degrees[Dataset == 'RICHOUX_REGULAR', Dataset := 'RICHOUX-REGULAR']
all_degrees[Dataset == 'RICHOUX_STRICT', Dataset := 'RICHOUX-STRICT']
all_degrees[Dataset == 'RICHOUX', Dataset := 'RICHOUX-UNIPROT']
all_degrees[Dataset == 'DSCRIPT', Dataset := 'D-SCRIPT UNBAL.']
all_degrees[, Test := stringr::str_to_title(Test)]
all_degrees[, Test := factor(Test, levels = c('Original', 'Rewired', 'Partition'))]
all_degrees[, Network := stringr::str_to_title(Network)]
all_degrees[, Split := stringr::str_to_title(Split)]

ggplot(all_degrees, aes(x=Degree, fill=Network))+
  geom_histogram(bins=30, position = 'dodge')+
  facet_wrap(Test~Dataset, scales = 'free', nrow = 3)+
  theme_bw()+
  xlim(0, 20)
ggsave('plots/node_degrees_pos_vs_neg.pdf', height = 6, width=14)

ggplot(all_degrees[Test == 'Original' & Dataset %in% c('HUANG', 'PAN') & Degree <= 15], aes(x=Degree, fill=Network))+
  geom_histogram(binwidth=1, position = 'dodge')+
  facet_wrap(~Dataset, scales = 'free', ncol = 1)+
  theme_bw()+
  theme(text = element_text(size=23))+
  labs(y = 'Count')+
  scale_x_reverse()+
  theme(legend.position='bottom')+
  coord_flip()
ggsave('plots/node_degrees_huang_pan_pos_vs_neg.pdf', height = 10, width=4.5)


all_degrees$Split <- factor(all_degrees$Split, levels = c('Train', 'Test', 'Inter', 'Intra-0', 'Intra-1', 'Intra-2'))
ggplot(all_degrees, aes(x=Degree, fill=Split))+
  geom_histogram(bins=30, position = 'dodge')+
  facet_wrap(Test~Dataset, scales = 'free', nrow = 3)+
  theme_bw()+
  scale_fill_manual(labels = c(
    'Train' = 'Train', 
    'Test' = 'Test', 
    'Inter' = TeX('$\\it{INTER}$'), 
    'Intra-0' = TeX('$\\it{INTRA}_0$'), 
    'Intra-1' = TeX('$\\it{INTRA}_1$'),
    'Intra-2' = TeX('$\\it{INTRA}_2$')
  ),
  values = brewer.pal(6, "Set1"))+
  xlim(0, 20)

ggsave('plots/node_degrees_train_vs_test.pdf', height = 6, width=14)

orig_vs_rewired <- merge(all_degrees[Test == 'Original'], all_degrees[Test == 'Rewired'], by=c('Node', 'Dataset', 'Split', 'Network'))
colnames(orig_vs_rewired) <- c('Node', 'Dataset', 'Split', 'Network', 'Degree original', 'Original', 'Degree rewired', 'Rewired')
ggplot(orig_vs_rewired, aes(x=`Degree original`, y=`Degree rewired`))+
  geom_point()+
  theme_bw() +
  geom_abline(intercept = 0, slope=1, color='red')+
  xlim(0,265)+
  ylim(0,265)

ggsave('plots/node_degrees_original_vs_rewired.pdf', height = 6, width=6)
