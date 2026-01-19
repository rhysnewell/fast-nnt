#' Set the global thread count for fastnnt (call once per session).
#'
#' @export
set_fastnnt_threads <- function(threads) {
    .Call(wrap__set_fastnnt_threads, threads)
}

#' Run NeighborNet and return a networkx-like list.
#'
#' @export
run_neighbornet_networkx <- function(
    x,
    flip_y = TRUE,
    labels = NULL,
    max_iterations = 5000,
    ordering_method = NULL
) {
    .Call(wrap__run_neighbornet_networkx, x, flip_y, labels, max_iterations, ordering_method)
}

#' Run NeighbourNet (alias) and return a networkx-like list.
#'
#' @export
run_neighbournet_networkx <- function(
    x,
    flip_y = TRUE,
    labels = NULL,
    max_iterations = 5000,
    ordering_method = NULL
) {
    .Call(wrap__run_neighbournet_networkx, x, flip_y, labels, max_iterations, ordering_method)
}
