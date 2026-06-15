# Shared fixtures for the test suite. One deterministic 5-taxon distance matrix
# (the `small_5` reference) keeps every test reproducible and fast.

small_dist_matrix <- function() {
  m <- matrix(
    c(0,  5,  9,  9,  8,
      5,  0, 10, 10,  9,
      9, 10,  0,  8,  7,
      9, 10,  8,  0,  3,
      8,  9,  7,  3,  0),
    nrow = 5, byrow = TRUE
  )
  dimnames(m) <- list(letters[1:5], letters[1:5])
  m
}

example_net <- function(...) {
  run_neighbornet_networkx(small_dist_matrix(), ...)
}
