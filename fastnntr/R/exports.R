#' @include extendr-wrappers.R
NULL

# Coerce supported inputs to a validated, symmetric numeric distance matrix.
# Accepts a `dist` object (the usual R representation of distances), a
# `data.frame`/`data.table`, or a `matrix`. Coercing a `dist` via `as.matrix()`
# both symmetrises the data and preserves the taxa labels in the dimnames.
.as_distance_matrix <- function(x) {
    if (inherits(x, "dist") || is.data.frame(x) || !is.matrix(x)) {
        x <- as.matrix(x)
    }
    if (!is.numeric(x)) {
        stop("'x' must be a numeric distance matrix or a 'dist' object", call. = FALSE)
    }
    if (nrow(x) != ncol(x)) {
        stop(sprintf("'x' must be square (got %d x %d)", nrow(x), ncol(x)), call. = FALSE)
    }
    if (!isSymmetric(unname(x))) {
        stop("'x' must be a symmetric distance matrix", call. = FALSE)
    }
    x
}

#' Compute a NeighbourNet split network
#'
#' Runs the NeighbourNet algorithm (Bryant & Moulton, 2004) on a distance
#' matrix and returns the resulting split network in a `networx`-compatible
#' list, ready to plot with \pkg{tanggle}/\pkg{ggplot2} or base \pkg{phangorn}.
#'
#' NeighbourNet produces an *implicit* (split) network: a planar diagram that
#' summarises conflicting signal in the data for exploratory analysis. Unlike
#' *explicit* networks, the boxes do not represent specific reticulation events
#' such as hybridisation or introgression.
#'
#' `run_neighbournet_networkx()` is an identical spelling alias of
#' `run_neighbornet_networkx()`.
#'
#' @param x A distance matrix. May be a `dist` object, a numeric `matrix`, or a
#'   `data.frame`/`data.table` of distances; non-matrix inputs are coerced with
#'   [as.matrix()]. The matrix must be square and symmetric.
#' @param flip_y Logical; if `TRUE` (default) the vertical axis of the computed
#'   layout is flipped, matching the orientation used by SplitsTree.
#' @param labels Optional character vector of taxon labels. If `NULL` (default)
#'   the column names of `x` are used (and, for a `dist` object, its labels).
#' @param max_iterations Integer; maximum NNLS iterations for split-weight
#'   estimation (default `5000`).
#' @param ordering_method Split ordering algorithm: `"multiway"` (default) or
#'   `"closest-pair"`. `NULL` uses the default.
#' @param inference_method Split-weight solver: `"active-set"` (default) or
#'   `"cg"`. `NULL` uses the default.
#'
#' @return A list of class `c("networx", "phylo")` containing `edge`,
#'   `tip.label`, `edge.length`, `Nnode`, `splitIndex`, `splits`, `translate`,
#'   and a `.plot` sublist whose `vertices` matrix holds 2-D node coordinates
#'   from the equal-angle layout. The structure is compatible with
#'   \pkg{phangorn}'s `networx` objects and \pkg{tanggle}'s ggplot2 layers.
#'
#' @seealso \pkg{tanggle} and \pkg{phangorn} for plotting `networx` objects;
#'   [stats::dist()] and `phangorn::dist.ml()` for computing distances.
#'
#' @examples
#' \donttest{
#' # A minimal workflow from raw distances to a plotted network.
#' d <- dist(matrix(rnorm(50), nrow = 5, dimnames = list(LETTERS[1:5], NULL)))
#' net <- run_neighbornet_networkx(d)
#' str(net$tip.label)
#'
#' # Plot with tanggle/ggplot2 (if installed):
#' if (requireNamespace("tanggle", quietly = TRUE)) {
#'   library(tanggle)
#'   ggplot2::ggplot(net) + geom_splitnet() + geom_tiplab2()
#' }
#' }
#'
#' @export
run_neighbornet_networkx <- function(
    x,
    flip_y = TRUE,
    labels = NULL,
    max_iterations = 5000,
    ordering_method = NULL,
    inference_method = NULL
) {
    x <- .as_distance_matrix(x)
    .Call(
        wrap__run_neighbornet_networkx,
        x,
        flip_y,
        labels,
        max_iterations,
        ordering_method,
        inference_method
    )
}

#' @rdname run_neighbornet_networkx
#' @export
run_neighbournet_networkx <- function(
    x,
    flip_y = TRUE,
    labels = NULL,
    max_iterations = 5000,
    ordering_method = NULL,
    inference_method = NULL
) {
    run_neighbornet_networkx(
        x,
        flip_y,
        labels,
        max_iterations,
        ordering_method,
        inference_method
    )
}

#' Set the global thread count for fastnnt
#'
#' Configures the size of the internal Rayon thread pool used for split-weight
#' estimation. The pool can only be initialised once per R session; subsequent
#' calls emit a warning and leave the existing pool unchanged.
#'
#' @param threads Integer (>= 1); number of worker threads to use.
#'
#' @return Invisibly `NULL`; called for its side effect.
#'
#' @examples
#' \donttest{
#' set_fastnnt_threads(4)
#' }
#'
#' @export
set_fastnnt_threads <- function(threads) {
    .Call(wrap__set_fastnnt_threads, threads)
}
