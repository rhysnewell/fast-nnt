Sys.setenv(GGPLOT2_USE_S7 = "false")
options(repos = c(CRAN = "https://cloud.r-project.org"))
options(ggplot2.use_s7 = FALSE) # Avoid S7 theme crashes with older ggplot2 extensions.

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", quiet = TRUE)

if (!requireNamespace("tanggle", quietly = TRUE))
    BiocManager::install("tanggle", ask=FALSE, update=FALSE)

install_if_missing <- function(pkgs) {
  need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(need)) install.packages(need, quiet = TRUE)
}


install_if_missing(c("dplyr", "ggplot2", "ggforce", "ggnewscale", "ggpubr", "data.table", "devtools", "phangorn"))
library(tanggle)
library(dplyr)
library(ggplot2)
library(tanggle)
library(ggforce)
library(ggnewscale)
library(ggpubr)
library(data.table)
devtools::load_all("anonnntr")
set_anon_nnt_threads(8)

#--- 1. Function to make one plot from a file ---#
plot_networx <- function(Nnet, title=NULL) {
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
  
  # Prepare plotting data (fallback to computed coordinates if .plot is missing)
  coord <- if (!is.null(Nnet$.plot)) Nnet$.plot$vertices else phangorn::coords(Nnet, dim = "equal_angle")
  x <- data.frame(x = coord[,1],
                  y = coord[,2],
                  sample = rep(NA, nrow(coord)))
  
  if (!is.null(Nnet$translate)) {
    x[Nnet$translate$node, "sample"] <- Nnet$translate$label
  }
  
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

make_splitstree_plot <- function(title=NULL, ordering_method="splitstree4", inference_method="active-set") {
  data <- fread("test/data/large/large_dist_matrix.csv", header=TRUE)
  # Load network
  
  Nnet <- run_neighbornet_networkx(
    data,
    flip_y=TRUE,
    labels=names(data),
    max_iterations=5000,
    ordering_method=ordering_method,
    inference_method=inference_method
  )
  # Nnet <- run_neighbornet_networkx(data)
  plot_networx(Nnet, title=title)
}

make_nexus_plot <- function(path, title=NULL) {
  # Parse Nexus using phangorn's networx reader for large files
  Nnet <- phangorn::read.nexus.networx(path)
  if (inherits(Nnet, "splits")) {
    Nnet <- phangorn::as.networx(Nnet)
  }
  plot_networx(Nnet, title=title)
}

#--- 2. Define input files ---#

plot1 <- make_splitstree_plot(
  title="Anon-NNT - SplitsTree4 Ordering (Active-Set Inference)",
  ordering_method="splitstree4",
  inference_method="active-set"
)
plot2 <- make_splitstree_plot(
  title="Anon-NNT - SplitsTree4 Ordering (SplitsTree4 Inference)",
  ordering_method="splitstree4",
  inference_method="splitstree4"
)
plot3 <- make_splitstree_plot(
  title="Anon-NNT - Huson2023 Ordering (Active-Set Inference)",
  ordering_method="huson2023",
  inference_method="active-set"
)
plot4 <- make_splitstree_plot(
  title="Anon-NNT - Huson2023 Ordering (SplitsTree4 Inference)",
  ordering_method="huson2023",
  inference_method="splitstree4"
)
plot5 <- make_nexus_plot("test/data/large/euc_splitstree4.nex", title="NEXUS - SplitsTree4 Ordering")
plot6 <- make_nexus_plot("test/data/large/st6_huson2023.stree6", title="SplitsTree6 - Huson2023 Ordering")



#--- 4. Arrange together ---#

combined <- ggpubr::ggarrange(plot1, plot2, plot3, plot4, plot5, plot6, ncol=3, nrow=2, align="hv")

ggsave('test/r/anon_nnt_graph_R.png', combined,
       width=44, height=30, units='cm', bg='white')
