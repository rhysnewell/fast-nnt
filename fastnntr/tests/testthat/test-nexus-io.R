# Regression tests for the reported bug: write.nexus.networx() failed with
# "replacement has length zero" because $splits was a bare matrix rather than a
# phangorn "splits" object, and the written CYCLE line was all zeros.

test_that("write.nexus.networx writes, and reads back, without error", {
  skip_if_not_installed("phangorn")
  net <- example_net()

  path <- tempfile(fileext = ".nexus")
  on.exit(unlink(path), add = TRUE)

  # The exact call the user reported; must not raise.
  expect_no_error(phangorn::write.nexus.networx(net, file = path))
  expect_true(file.exists(path))

  back <- phangorn::read.nexus.networx(path)
  expect_s3_class(back, "networx")
  expect_equal(length(back$splits), length(net$splits))
})

test_that("the written CYCLE line is a real permutation, not zeros", {
  skip_if_not_installed("phangorn")
  net <- example_net()

  path <- tempfile(fileext = ".nexus")
  on.exit(unlink(path), add = TRUE)
  phangorn::write.nexus.networx(net, file = path)

  cyc_line <- grep("CYCLE", readLines(path), value = TRUE)
  expect_length(cyc_line, 1)
  nums <- as.integer(strsplit(trimws(sub(".*CYCLE", "", cyc_line)), "[ \t;]+")[[1]])
  nums <- nums[!is.na(nums)]
  expect_setequal(nums, seq_along(net$tip.label))
})

test_that("write.nexus.splits accepts the $splits object directly", {
  skip_if_not_installed("phangorn")
  net <- example_net()

  path <- tempfile(fileext = ".nex")
  on.exit(unlink(path), add = TRUE)
  expect_no_error(phangorn::write.nexus.splits(net$splits, file = path))
  expect_true(file.exists(path))
})
