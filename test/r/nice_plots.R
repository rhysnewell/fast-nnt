options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", quiet = TRUE)

if (!requireNamespace("tanggle", quietly = TRUE))
    BiocManager::install("tanggle", ask=FALSE, update=FALSE)

install_if_missing <- function(pkgs) {
  need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(need)) install.packages(need, quiet = TRUE)
}


install_if_missing(c("dplyr", "ggplot2", "ggforce", "ggnewscale", "ggpubr", "data.table", "devtools"))
library(tanggle)
library(dplyr)
library(ggplot2)
library(tanggle)
library(ggforce)
library(ggnewscale)
library(ggpubr)
library(data.table)
devtools::load_all("fastnntr")

#--- 1. Function to make one plot from a file ---#
make_splitstree_plot <- function(title=NULL) {
  data <- fread("test/data/large/large_dist_matrix.csv", header=TRUE)
  # Load network
  
  Nnet <- run_neighbornet_networkx(data, flip_y=TRUE, labels=names(data), max_iterations=5000, ordering_method="splitstree4")
  # Nnet <- run_neighbornet_networkx(data)

  message("debug: edge class=", paste(class(Nnet$edge), collapse = ","), 
          " dim=", paste(dim(Nnet$edge), collapse = "x"), 
          " length=", length(Nnet$edge))
  message("debug: edge typeof=", typeof(Nnet$edge), 
          " is.matrix=", is.matrix(Nnet$edge), 
          " is.integer=", is.integer(Nnet$edge))
  message("debug: edge first rows:")
  print(utils::head(Nnet$edge))
  message("debug: names=", paste(names(Nnet), collapse = ","))
  message("debug: class(Nnet)=", paste(class(Nnet), collapse = ","))
  nb.tip <- length(Nnet$tip.label)
  neworder <- try(ape:::reorderRcpp(Nnet$edge, as.integer(nb.tip), as.integer(nb.tip + 1L), 2L), silent = TRUE)
  if (inherits(neworder, "try-error")) {
    message("debug: reorderRcpp error: ", neworder)
  } else {
    message("debug: reorderRcpp class=", paste(class(neworder), collapse = ","), 
            " dim=", paste(dim(neworder), collapse = "x"), 
            " length=", length(neworder))
    message("debug: reorderRcpp range=", paste(range(neworder), collapse = ".."),
            " anyNA=", any(is.na(neworder)),
            " anyOutOfRange=", any(neworder < 1 | neworder > nrow(Nnet$edge)))
    tmp <- Nnet$edge[neworder, ]
    message("debug: edge[neworder,] class=", paste(class(tmp), collapse = ","), 
            " dim=", paste(dim(tmp), collapse = "x"),
            " typeof=", typeof(tmp))
    tmp2 <- Nnet$edge[neworder, , drop = FALSE]
    message("debug: edge[neworder,,drop=FALSE] class=", paste(class(tmp2), collapse = ","), 
            " dim=", paste(dim(tmp2), collapse = "x"),
            " typeof=", typeof(tmp2))
  }
  reord <- try(reorder(Nnet, "postorder"), silent = TRUE)
  if (inherits(reord, "try-error")) {
    message("debug: reorder error: ", reord)
  } else {
    message("debug: reorder edge class=", paste(class(reord$edge), collapse = ","), 
            " dim=", paste(dim(reord$edge), collapse = "x"),
            " typeof=", typeof(reord$edge))
  }
  lad <- try(ape::ladderize(Nnet), silent = TRUE)
  if (inherits(lad, "try-error")) {
    message("debug: ladderize error: ", lad)
  } else {
    message("debug: ladderize edge class=", paste(class(lad$edge), collapse = ","), 
            " dim=", paste(dim(lad$edge), collapse = "x"))
  }
  
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

plot1 <- make_splitstree_plot(title="Fast-NNT - SplitsTree4 Ordering")



#--- 4. Arrange together ---#

ggsave('test/fast_nnt_graph_R.png',plot1, 
       width=22, height=15, units='cm', bg='white')
