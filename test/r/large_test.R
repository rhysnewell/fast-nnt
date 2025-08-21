
# ---- deps ----
options(repos = c(CRAN = "https://cloud.r-project.org"))

install_if_missing <- function(pkgs) {
  need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(need)) install.packages(need, quiet = TRUE)
}

# CRAN deps you use in the script
install_if_missing(c("ggplot2", "ggrepel", "dplyr", "tidyr", "data.table", "svglite"))

# Ensure fastnntr is installed; if not, install from the local repo path
if (!requireNamespace("fastnntr", quietly = TRUE)) {
  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools", quiet = TRUE)
  # Infer repo root from this script's path (works with Rscript)
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub("^--file=", "", args[grep("^--file=", args)])
  if (!length(script_path)) {
    # fallback if run interactively or path not present
    script_path <- "test/r/large_test.R"
  }
  script_dir <- normalizePath(dirname(script_path), winslash = "/", mustWork = FALSE)
  repo_root  <- normalizePath(file.path(script_dir, "..", ".."), winslash = "/", mustWork = TRUE)
  pkg_path   <- file.path(repo_root, "fastnntr")
  devtools::install(pkg_path, upgrade = "never", quiet = TRUE)
}

# library(fastnntr)
library(ggplot2)
library(ggrepel)
library(dplyr)
library(tidyr)
library(data.table)
library(svglite)
devtools::load_all("fastnntr")


plot_fastnnt_from_matrix <- function(dist_mat,
                                     labels = NULL,            # character vector or NULL
                                     out_png = "test/plots/fast_nnt_graph_R.png",
                                     out_svg = "test/plots/fast_nnt_graph_R.svg",
                                     label_leaves_only = TRUE,
                                     width_by_weight = TRUE) {

  # 1) run NeighborNet (your Rust uses column names when labels = NULL)
  nx <- run_neighbour_net(dist_mat, labels, "")

  # 2) pull data via $ methods (no wrappers)
  tr    <- as.data.frame(nx$get_node_translations())     # id (1-based), label
  pos   <- as.data.frame(nx$get_node_positions())        # id, x, y
  edges <- as.data.frame(nx$get_graph_edges())           # edge_id, u, v, split_id, weight

  # 3) keep only edges with coords for both endpoints
  edges <- edges |>
    semi_join(pos, by = c("u" = "id")) |>
    semi_join(pos, by = c("v" = "id"))

  # 4) leaves = degree 1
  deg <- bind_rows(transmute(edges, id = u),
                   transmute(edges, id = v)) |>
    count(id, name = "deg")

  nodes <- pos |>
    left_join(tr,  by = "id") |>
    left_join(deg, by = "id") |>
    mutate(deg = replace_na(deg, 0L))

  leaves <- dplyr::filter(nodes, deg == 1L)

  # 5) edge coordinates for plotting
  edges_xy <- edges |>
    left_join(rename(pos, xu = x, yu = y), by = c("u" = "id")) |>
    left_join(rename(pos, xv = x, yv = y), by = c("v" = "id"))

  # 6) edge layer (optional width scaling by weight)
  edge_layer <-
    if (width_by_weight && nrow(edges_xy)) {
      rng <- range(edges_xy$weight, na.rm = TRUE)
      if (is.finite(rng[1]) && is.finite(rng[2]) && diff(rng) > 0) {
        edges_xy$width <- 0.3 + 1.7 * (edges_xy$weight - rng[1]) / diff(rng)
      } else {
        edges_xy$width <- 0.6
      }
      geom_segment(
        data = edges_xy,
        aes(x = xu, y = yu, xend = xv, yend = yv, linewidth = width),
        lineend = "round", alpha = 0.9, colour = "black", show.legend = FALSE
      )
    } else {
      geom_segment(
        data = edges_xy,
        aes(x = xu, y = yu, xend = xv, yend = yv),
        lineend = "round", linewidth = 0.01, alpha = 0.9, colour = "black"
      )
    }

  # 7) assemble plot (only leaf labels)
  p <- ggplot() +
    edge_layer +
    {
      if (label_leaves_only && nrow(leaves)) {
        list(
          geom_point(data = leaves, aes(x = x, y = y), size = 0.8, colour = "black"),
          ggrepel::geom_text_repel(
            data = leaves, aes(x = x, y = y, label = label),
            size = 2.2, box.padding = 0.05, point.padding = 0.05,
            segment.size = 0.1, max.overlaps = Inf
          )
        )
      } else NULL
    } +
    coord_equal() +
    theme_void() +
    ggtitle("Fast-NNT NeighborNet")

  dir.create(dirname(out_png), recursive = TRUE, showWarnings = FALSE)
  ggsave(out_png, p, width = 8, height = 8, dpi = 300)
  ggsave(out_svg, p, width = 8, height = 8)

  invisible(p)
}

# ---- example usage ----
# Build a toy symmetric distance matrix (labels taken from colnames)
set.seed(1)
# read in test/data/large/large_dist_matrix.csv
m <- data.table::fread("test/data/large/large_dist_matrix.csv", header = TRUE)

plot_fastnnt_from_matrix(m)  # saves PNG+SVG into ./plots/
