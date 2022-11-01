test_that("produces same classifications as policytree for depth 1", {
 for (i in 1:30) {

    n <- 400
    p <- 4
    d <- 3
    depth <- 1

    # Classification task taken from policytree tests
    X <- round(matrix(rnorm(n * p), n, p),2)
    Y <- matrix(0, n, d)
    best.tree <- policytree:::make_tree(X, depth = depth, d = d)
    best.action <- policytree:::predict_test_tree(best.tree, X)
    Y[cbind(1:n, best.action)] <- 100 * runif(n)

    tree_1 <- parallel_policy_tree(X,Y,1)
    tree_2 <- policytree:::policy_tree(X,Y,1)

    expect_equal(predict(tree_1,X),predict(tree_2,X))

    # Classification task taken from policytree tests
    X <- matrix(sample(10:20, n * p, replace = TRUE), n, p)
    Y <- matrix(0, n, d)

    tree_1 <- parallel_policy_tree(X,Y,1)
    best.tree <- policytree:::make_tree(X, depth = depth, d = d)
    best.action <- policytree:::predict_test_tree(best.tree, X)
    Y[cbind(1:n, best.action)] <- 100 * runif(n)

    tree_1 <- parallel_policy_tree(X,Y,1)
    tree_2 <- policytree:::policy_tree(X,Y,1)

    expect_equal(predict(tree_1,X),predict(tree_2,X))
 }
})

test_that("produces same classifications as policytree for depth 2", {
 for (i in 1:30) {

    n <- 400
    p <- 4
    d <- 3
    depth <- 2

    # Classification task taken from policytree tests
    X <- round(matrix(rnorm(n * p), n, p),2)
    Y <- matrix(0, n, d)
    best.tree <- policytree:::make_tree(X, depth = depth, d = d)
    best.action <- policytree:::predict_test_tree(best.tree, X)
    Y[cbind(1:n, best.action)] <- 100 * runif(n)

    tree_1 <- parallel_policy_tree(X,Y,2)
    tree_2 <- policytree:::policy_tree(X,Y,2)

    expect_equal(predict(tree_1,X),predict(tree_2,X))

    # Classification task taken from policytree tests
    X <- matrix(sample(10:20, n * p, replace = TRUE), n, p)
    Y <- matrix(0, n, d)

    best.tree <- policytree:::make_tree(X, depth = depth, d = d)
    best.action <- policytree:::predict_test_tree(best.tree, X)
    Y[cbind(1:n, best.action)] <- 100 * runif(n)

    tree_1 <- parallel_policy_tree(X,Y,2)
    tree_2 <- policytree:::policy_tree(X,Y,2)

    expect_equal(predict(tree_1,X),predict(tree_2,X))
 }
})

test_that("produces same classifications as policytree for depth 3", {
 for (i in 1:20) {

    n <- 400
    p <- 2
    d <- 2
    depth <- 3

    # Classification task taken from policytree tests
    X <- round(matrix(rnorm(n * p), n, p),1)
    colnames(X) <- letters[1:ncol(X)]
    Y <- matrix(0, n, d)
    best.tree <- policytree:::make_tree(X, depth = depth, d = d)
    best.action <- policytree:::predict_test_tree(best.tree, X)
    Y[cbind(1:n, best.action)] <- 100 * runif(n)
    tree_1 <- parallel_policy_tree(X,Y,3)
    tree_2 <- policytree:::policy_tree(X,Y,3)

    expect_equal(predict(tree_1,X),predict(tree_2,X))

    # Classification task taken from policytree tests
    X <- matrix(sample(1:5, n * p, replace = TRUE), n, p)
    Y <- matrix(0, n, d)

    best.tree <- policytree:::make_tree(X, depth = depth, d = d)
    best.action <- policytree:::predict_test_tree(best.tree, X)
    Y[cbind(1:n, best.action)] <- 100 * runif(n)

    tree_1 <- parallel_policy_tree(X,Y,3)
    tree_2 <- policytree:::policy_tree(X,Y,3)

    expect_equal(predict(tree_1,X),predict(tree_2,X))
 }
})
