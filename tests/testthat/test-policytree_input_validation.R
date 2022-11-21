test_that("policytree validates that X and G have same number of rows", {
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

    expect_error(sparse_policy_tree(head(X),Y,1), "number of rows")

})

test_that("policytree validates that depth is positive", {

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



    expect_error(sparse_policy_tree(X,Y,-5),
                 "`depth` cannot be negative.")

})

test_that("policytree checks for no missing values in X", {

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
    X[1:10,1:2] <- NA



    expect_error(sparse_policy_tree(X,Y,1),
                 "Covariate matrix X contains missing values.")

})

test_that("policytree checks for no missing values in Gamma", {

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
    Y[1:10,1:2] <- NA



    expect_error(sparse_policy_tree(X,Y,1),
                 "Gamma matrix contains missing values.")

})
