test_that("produces same classifications as policytree", {
 for (i in 1:10) {
    n <- 400
    p <- 4
    d <- 3

    # Classification task taken from policytree tests
    X <- round(matrix(rnorm(n * p), n, p),2)
    Y <- matrix(0, n, d)

    tree_1 <- exhaustive_tree(X,Y,2)
    tree_2 <- policy_tree(X,Y,2)

    expect_equal(predict(tree_1,X),predict(tree_2,X))

    # Classification task taken from policytree tests
    X <- matrix(as.numeric(sample(10:20, n * p, replace = TRUE)), n, p)
    Y <- matrix(0, n, d)

    tree_1 <- exhaustive_tree(X,Y,2)
    tree_2 <- policy_tree(X,Y,2)

    expect_equal(predict(tree_1,X),predict(tree_2,X))
 }
})
