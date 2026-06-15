# Input coercion and validation performed by .as_distance_matrix().

test_that("matrix, dist, and data.frame inputs are accepted and agree", {
  m <- small_dist_matrix()
  from_mat  <- run_neighbornet_networkx(m)
  from_dist <- run_neighbornet_networkx(as.dist(m))
  from_df   <- run_neighbornet_networkx(as.data.frame(m))

  expect_equal(from_mat$tip.label, from_dist$tip.label)
  expect_equal(from_mat$tip.label, from_df$tip.label)
  expect_equal(length(from_mat$splits), length(from_dist$splits))
  expect_equal(length(from_mat$splits), length(from_df$splits))
})

test_that("explicit labels override the matrix dimnames", {
  net <- run_neighbornet_networkx(small_dist_matrix(), labels = LETTERS[1:5])
  expect_equal(net$tip.label, LETTERS[1:5])
  expect_equal(attr(net$splits, "labels"), LETTERS[1:5])
})

test_that("non-square and non-symmetric inputs are rejected", {
  expect_error(run_neighbornet_networkx(matrix(1:6, nrow = 2)), "square")

  ns <- small_dist_matrix()
  ns[1, 2] <- 99  # break symmetry
  expect_error(run_neighbornet_networkx(ns), "symmetric")
})
