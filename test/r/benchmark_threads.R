Sys.setenv(GGPLOT2_USE_S7 = "false")
options(repos = c(CRAN = "https://cloud.r-project.org"))
options(ggplot2.use_s7 = FALSE)

install_if_missing <- function(pkgs) {
  need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(need)) install.packages(need, quiet = TRUE)
}

install_if_missing(c("data.table", "devtools", "ps", "peakRAM"))
library(data.table)
library(ps)
library(peakRAM)
devtools::load_all("anonnntr")

data <- fread("test/data/large/large_dist_matrix.csv", header = TRUE)
labels <- names(data)

thread_counts <- c(1, 2, 4, 8)
repeats <- 3

run_once <- function(threads) {
  if (requireNamespace("peakRAM", quietly = TRUE)) {
    bench <- peakRAM::peakRAM(
      {
        set_anon_nnt_threads(threads)
        run_neighbornet_networkx(
          data,
          flip_y = TRUE,
          labels = labels,
          max_iterations = 5000,
          ordering_method = "splitstree4"
        )
      }
    )
    if (!is.null(bench$Elapsed_Time_Seconds) && length(bench$Elapsed_Time_Seconds) >= 1) {
      return(data.frame(
        threads = threads,
        elapsed_sec = bench$Elapsed_Time_Seconds[1],
        cpu_sec = bench$CPU_Seconds_Used[1],
        peak_ram_mib = bench$Peak_RAM_Used_MiB[1],
        peak_vmem_mib = bench$Peak_Virtual_Memory_Used_MiB[1]
      ))
    }
  }

  get_rss_bytes <- function() {
    info <- ps::ps_memory_info(ps::ps_handle())
    if (is.list(info) && !is.null(info$rss)) {
      return(as.numeric(info$rss))
    }
    if (!is.null(names(info)) && "rss" %in% names(info)) {
      return(as.numeric(info[["rss"]]))
    }
    if (length(info) >= 1) {
      return(as.numeric(info[1]))
    }
    NA_real_
  }

  set_anon_nnt_threads(threads)
  t0 <- proc.time()
  rss_before <- get_rss_bytes()
  run_neighbornet_networkx(
    data,
    flip_y = TRUE,
    labels = labels,
    max_iterations = 5000,
    ordering_method = "splitstree4"
  )
  t1 <- proc.time() - t0
  rss_after <- get_rss_bytes()

  data.frame(
    threads = threads,
    elapsed_sec = as.numeric(t1["elapsed"]),
    cpu_sec = as.numeric(t1["user.self"] + t1["sys.self"]),
    peak_ram_mib = max(rss_before, rss_after, na.rm = TRUE) / (1024 * 1024),
    peak_vmem_mib = NA_real_
  )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) >= 2 && args[1] == "--single") {
  t <- as.integer(args[2])
  out_csv <- args[3]
  res <- run_once(t)
  write.csv(res, out_csv, row.names = FALSE)
  quit(save = "no")
}

results <- do.call(
  rbind,
  lapply(thread_counts, function(t) {
    do.call(rbind, lapply(seq_len(repeats), function(i) {
      out_csv <- tempfile(fileext = ".csv")
      status <- system2(
        "Rscript",
        c("test/r/benchmark_threads.R", "--single", as.character(t), out_csv),
        stdout = TRUE,
        stderr = TRUE
      )
      if (!is.null(attr(status, "status")) && attr(status, "status") != 0) {
        stop("Benchmark subprocess failed for threads=", t)
      }
      read.csv(out_csv, stringsAsFactors = FALSE)
    }))
  })
)

print(results)
write.csv(results, "test/r/anon_nnt_threads_benchmark.csv", row.names = FALSE)
