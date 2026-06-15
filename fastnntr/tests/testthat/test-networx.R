# Structure of the returned networx object. These guard the contract that
# phangorn/tanggle rely on; the splits-object checks in particular are a
# regression test for the `write.nexus.networx()` crash (see test-nexus-io.R).

test_that("run_neighbornet_networkx returns a well-formed networx object", {
  net <- example_net()

  expect_s3_class(net, "networx")
  expect_s3_class(net, "phylo")
  expect_setequal(
    names(net),
    c("edge", "tip.label", "edge.length", "Nnode",
      "splitIndex", "splits", "translate", ".plot")
  )
  expect_equal(net$tip.label, letters[1:5])

  # edge is an n x 2 integer matrix aligned with edge.length and splitIndex
  expect_true(is.matrix(net$edge))
  expect_equal(ncol(net$edge), 2L)
  expect_equal(nrow(net$edge), length(net$edge.length))
  expect_equal(length(net$splitIndex), nrow(net$edge))

  # vertices: one (x, y) row per node
  expect_equal(ncol(net$.plot$vertices), 2L)
  expect_true(all(is.finite(net$.plot$vertices)))

  # translate maps node ids to the original labels
  expect_setequal(net$translate$label, net$tip.label)
})

test_that("$splits is a valid phangorn splits object", {
  net <- example_net()
  spl <- net$splits
  ntax <- length(net$tip.label)

  expect_s3_class(spl, "splits")
  expect_type(spl, "list")

  # The `labels` attribute drives phangorn's ONEwise(); its absence made
  # nTips = 0 and produced the "replacement has length zero" crash.
  expect_equal(attr(spl, "labels"), net$tip.label)

  # exactly one weight per split, all finite
  expect_equal(length(attr(spl, "weights")), length(spl))
  expect_true(all(is.finite(attr(spl, "weights"))))

  # cycle is a permutation of the taxa, not the all-zero vector the broken
  # graph.get_cycle() used to return
  expect_setequal(attr(spl, "cycle"), seq_len(ntax))

  # every split is a non-empty proper subset of valid 1-based taxon indices
  for (s in spl) {
    expect_true(length(s) >= 1 && length(s) < ntax)
    expect_true(all(s >= 1 & s <= ntax))
  }

  # splitIndex references valid positions in the splits list
  expect_true(all(net$splitIndex >= 1 & net$splitIndex <= length(spl)))
})

test_that("ordering and inference options all produce a network", {
  for (om in c("multiway", "closest-pair")) {
    for (im in c("active-set", "cg")) {
      net <- example_net(ordering_method = om, inference_method = im)
      expect_s3_class(net, "networx")
      expect_gt(length(net$splits), 0)
    }
  }
})

test_that("US/aliased spelling gives an equivalent network", {
  a <- example_net()
  b <- run_neighbournet_networkx(small_dist_matrix())
  expect_equal(a$tip.label, b$tip.label)
  expect_equal(length(a$splits), length(b$splits))
})
