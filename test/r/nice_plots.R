options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("tanggle")

install_if_missing <- function(pkgs) {
  need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(need)) install.packages(need, quiet = TRUE)
}


install_if_missing(c("phangorn", "dplyr", "ggplot2", "ggforce", "ggnewscale", "ggpubr"))

# library(phangorn)
library(dplyr)
library(ggplot2)
library(tanggle)
library(ggforce)
library(ggnewscale)
library(ggpubr)
library(data.table)
# library(RColorBrewer)
devtools::load_all("fastnntr")

#--- 1. Function to make one plot from a file ---#
make_splitstree_plot <- function(title=NULL) {
  data <- fread("test/data/large/large_dist_matrix.csv", header=TRUE)
  # Load network
  Nnet <- run_neighbornet_networx(data, names(data), TRUE)
  
  # Prepare plotting data
  x <- data.frame(x = Nnet$.plot$vertices[,1],
                  y = Nnet$.plot$vertices[,2],
                  sample = rep(NA, nrow(Nnet$.plot$vertices)))
  
  x[Nnet$translate$node, "sample"] <- Nnet$translate$label
  
  # Create the plot
  p <- ggplot(Nnet, aes(x = x, y = y)) +
    geom_splitnet(layout = "slanted", size = 0.2) +
    # geom_point(data = x_tips, 
    #            aes(x = x, y = y), size = 1) +         
    geom_tiplab2(size = 3, hjust = -0.9, vjust=0.2) +
    theme_bw() +
    theme(
      legend.position = 'bottom',
      # legend.text = element_text(face = "italic"),
      legend.key.size = unit(0, 'lines')
    ) +
    coord_fixed() +
    labs(shape="Species", colour='Species', title=title)
  
  return(p)
}

#--- 2. Define input files ---#

plot1 <- make_splitstree_plot(title="Fast-NNT - Huson2023 Ordering")



#--- 4. Arrange together ---#

ggsave('test/fast_nnt_graph_R.png',plot1, 
       width=22, height=15, units='cm', bg='white')
