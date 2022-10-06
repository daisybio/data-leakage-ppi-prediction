library(data.table)
library(ggplot2)

citations <- data.table(
  citations = c(623, 63, 147, 269, 121, 63, 8, 31, 143, 20, 141, 40, 9, 51, 41, 8, 200, 80, 951, 2, 30, 96, 50, 3, 128, 20),
  code_available = c(F, F, F, F, T, F, F, F, T, F, F, T, F, F, F, F, T, F, F, F, F, T, T, T, T, T)
)

ggplot(citations, aes(x = citations, color = code_available))+
  stat_ecdf()+
  theme_bw()
