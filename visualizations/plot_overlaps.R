library(VennDiagram)
library(data.table)
library(RColorBrewer)
library(gridExtra)
library(ggplot2)
library(latex2exp)


for(run in c('original', 'rewired')){
  print(run)
  myCol <- brewer.pal(4, "Set2")[c(3,4)]

  all_sets <- lapply(list.files(path=paste0('../algorithms/SPRINT/data/', run), full.names = T, recursive = T), 
                     fread, header=F)
  names(all_sets) <- tstrsplit(list.files(path=paste0('../algorithms/SPRINT/data/', run), recursive = T), '.txt', keep=1)[[1]]
  all_sets <- lapply(all_sets, function(x){
    return(unique(melt(x, measure.vars = c('V1', 'V2'), value.name = 'Protein')[, -'variable']))
  })
  all_sets <- rbindlist(all_sets, idcol='filename')
  all_sets[, filename := gsub('richoux_', 'richoux-', filename)]
  all_sets[, c('dataset', 'set') := tstrsplit(filename, '_', keep=c(1, 2))]
  all_sets <- all_sets[, -'filename']
  all_sets[, dataset := factor(dataset, levels=c('huang', 'guo', 'du', 'pan', 'richoux-regular', 'richoux-strict', 'dscript'))]
  all_sets <- all_sets[order(dataset)]
  
  plot_list <- list()
  
  for (ds in c('huang', 'guo', 'du', 'pan', 'richoux-regular', 'richoux-strict', 'dscript')){
    train <- unique(all_sets[set == 'train' & dataset == ds, Protein])
    test <- unique(all_sets[set == 'test' & dataset == ds, Protein])
    if(ds == 'dscript'){
      ds <- 'd-script unbal.'
    }
    venn_plot <- venn.diagram(list(train, test), 
                              category.names = c('Training', 'Test'), 
                              main = stringr::str_to_upper(ds),
                              filename = NULL,
                              disable.logging = TRUE,
                              # Output features
                              imagetype="png" ,
                              height = 3, 
                              width = 3, 
                              units = 'cm',
                              resolution = 600,
                              compression = "lzw",
                              
                              # Circles
                              lwd = 2,
                              lty = 'blank',
                              fill = myCol,
                              
                              # Numbers
                              cex = 1.5,
                              fontface = "bold",
                              fontfamily = "sans",
                              
                              # Set names
                              main.cex = 1.5,
                              main.pos = c(0.5, 1.15),
                              main.fontfamily = "sans",
                              cat.cex = 1.5,
                              cat.default.pos = "outer",
                              cat.pos = c(330, 135),
                              cat.fontfamily = "sans")
    plot_list <- append(plot_list, setNames(list(venn_plot), ds))
  }
  
  g <- arrangeGrob(gTree(children=plot_list[['huang']]), 
                   gTree(children=plot_list[['guo']]),
                   gTree(children=plot_list[['du']]), 
                   gTree(children=plot_list[['pan']]), 
                   gTree(children=plot_list[['richoux-regular']]), 
                   gTree(children=plot_list[['richoux-strict']]), 
                   gTree(children=plot_list[['d-script unbal.']]), 
                   nrow = 1)
  ggsave(paste0('plots/venn_overlaps_', run, '.pdf'), g, height = 2, width = 18)
}


myCol <- brewer.pal(4, "Set2")[c(2, 3, 4)]
all_sets <- lapply(list.files(path='../algorithms/SPRINT/data/partitions', full.names = T, recursive = T), 
                   fread, header=F)
names(all_sets) <- tstrsplit(list.files(path='../algorithms/SPRINT/data/partitions', recursive = T), '.txt', keep=1)[[1]]
all_sets <- lapply(all_sets, function(x){
  return(unique(melt(x, measure.vars = c('V1', 'V2'), value.name = 'Protein')[, -'variable']))
})
all_sets <- rbindlist(all_sets, idcol='filename')
all_sets[, c('dataset', 'partition') := tstrsplit(filename, '_', keep=c(1,3))]
all_sets <- all_sets[, -'filename']
all_sets[, data_part := paste(dataset, partition, sep='_')]
all_sets[, both_0 := ifelse(partition %in% c('0', 'both'), T, F)]
all_sets[, both_1 := ifelse(partition %in% c('1', 'both'), T, F)]
all_sets[, p0_p1 := ifelse(partition %in% c('0', '1'), T, F)]
all_sets[, dataset := factor(dataset, levels=c('huang', 'guo', 'du', 'pan', 'richoux', 'dscript'))]
all_sets <- all_sets[order(dataset)]

plot_list <- list()

for (ds in c('huang', 'guo', 'du', 'pan', 'richoux')){
  p_0 <- unique(all_sets[partition == '0' & dataset == ds, Protein])
  p_1 <- unique(all_sets[partition == '1' & dataset == ds, Protein])
  p_both <- unique(all_sets[partition == 'both' & dataset == ds, Protein])
  if(ds == 'richoux')
    ds <- 'richoux-uniprot'
  venn_plot <- venn.diagram(x = list(p_0, p_1, p_both), 
               category.names = TeX(c('$\\it{INTRA}_0$', '$\\it{INTRA}_1$', '$\\it{INTER}$')), 
               main = stringr::str_to_upper(ds),
               filename = NULL,
               disable.logging = TRUE,
               # Output features
               imagetype="png" ,
               height = 3, 
               width = 3, 
               units = 'cm',
               resolution = 600,
               compression = "lzw",
               
               # Circles
               lwd = 2,
               lty = 'blank',
               fill = myCol,
               
               # Numbers
               cex = 1.5,
               fontface = "bold",
               fontfamily = "sans",
               
               # Set names
               main.cex = 1.5,
               main.pos = c(0.5, 1),
               main.fontfamily = "sans",
               cat.cex = 1.5,
               cat.default.pos = "outer",
               cat.pos = c(0, 0, 0),
               cat.fontfamily = "sans",
               rotation = 1)
  plot_list <- append(plot_list, setNames(list(venn_plot), ds))
}
p_0 <- unique(all_sets[partition == '0' & dataset == 'dscript', Protein])
p_1 <- unique(all_sets[partition == '1' & dataset == 'dscript', Protein])
p_both <- unique(all_sets[partition == 'both' & dataset == 'dscript', Protein])
venn_plot <- venn.diagram(x = list(p_0, p_1, p_both), 
                          category.names = TeX(c('$\\it{INTRA}_0$', '$\\it{INTRA}_1$', '$\\it{INTER}$')), 
                          main = stringr::str_to_upper('D-SCRIPT UNBALANCED'),
                          filename = NULL,
                          disable.logging = TRUE,
                          # Output features
                          imagetype="png" ,
                          height = 3, 
                          width = 3, 
                          units = 'cm',
                          resolution = 600,
                          compression = "lzw",
                          
                          # Circles
                          lwd = 2,
                          lty = 'blank',
                          fill = myCol,
                          
                          # Numbers
                          cex = 1.5,
                          fontface = "bold",
                          fontfamily = "sans",
                          
                          # Set names
                          main.cex = 1.5,
                          main.pos = c(0.5, 1),
                          main.fontfamily = "sans",
                          cat.cex = 1.5,
                          cat.default.pos = "outer",
                          cat.pos = c(320, 40, 180),
                          cat.fontfamily = "sans",
                          rotation = 1)
plot_list <- append(plot_list, setNames(list(venn_plot), 'dscript'))

all_sets <- lapply(list.files(path='../Datasets_PPIs/Hippiev2.3', full.names = T, pattern = '^Intra(0|1|2)_(pos|neg)_rr.txt'), 
                   fread, header=F)
names(all_sets) <- tstrsplit(list.files(path='../Datasets_PPIs/Hippiev2.3', pattern = '^Intra(0|1|2)_(pos|neg)_rr.txt'), '_rr', keep=1)[[1]]
all_sets <- lapply(all_sets, function(x){
  return(unique(melt(x, measure.vars = c('V1', 'V2'), value.name = 'Protein')[, -'variable']))
})
all_sets <- rbindlist(all_sets, idcol='filename')
all_sets[, dataset := tstrsplit(filename, '_', keep = 1)]
p_0 <- unique(all_sets[dataset == 'Intra0', Protein])
p_1 <- unique(all_sets[dataset == 'Intra1', Protein])
p_2 <- unique(all_sets[dataset == 'Intra2', Protein])
venn_plot <- venn.diagram(x = list(p_0, p_1, p_2), 
                          category.names = TeX(c('$\\it{INTRA}_0$', '$\\it{INTRA}_1$', '$\\it{INTRA}_2$')), 
                          main = 'GOLD STANDARD',
                          filename = NULL,
                          disable.logging = TRUE,
                          # Output features
                          imagetype="png" ,
                          height = 3, 
                          width = 3, 
                          units = 'cm',
                          resolution = 600,
                          compression = "lzw",
                          
                          # Circles
                          lwd = 2,
                          lty = 'blank',
                          fill = myCol,
                          
                          # Numbers
                          cex = 1.5,
                          fontface = "bold",
                          fontfamily = "sans",
                          
                          # Set names
                          main.cex = 1.5,
                          main.pos = c(0.5, 1.5),
                          main.fontfamily = "sans",
                          cat.cex = 1.5,
                          cat.default.pos = "outer",
                          cat.pos = c(0, 0, 0),
                          cat.fontfamily = "sans",
                          rotation = 1)
#plot_list <- append(plot_list, setNames(list(venn_plot), 'gold'))


g <- arrangeGrob(gTree(children=plot_list[['huang']]), 
             gTree(children=plot_list[['guo']]),
             gTree(children=plot_list[['du']]), 
             gTree(children=plot_list[['pan']]), 
             gTree(children=plot_list[['richoux-uniprot']]), 
             gTree(children=plot_list[['dscript']]), 
             ncol =2)
ggsave('~/Downloads/venn_overlaps_partitions.png', g, height = 12, width = 11)
#ggsave('plots/venn_overlaps_partitions.pdf', g, height = 5, width = 17)
