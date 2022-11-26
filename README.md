
# Sparse Policy Tree

<img src='logo.png' align="right" height="120">

An R / Rust Package to implement an exhaustive tree search for
policy-learning. Aims to extend and speed up work done with the
`policytree` package.

# Usage

There’s just one function, `sparse_policy_tree()`. Use it just as you
would the `policy_tree()` function from the `policytree` package.

Trees aren’t guaranteed to be exactly the same as those produced by
`policytree` in that some leaves may be left unpruned (working on this),
but they should give the same predictions.

``` r

n <- 400
p <- 4
d <- 3
depth <- 2
# Classification task taken from policytree tests
# Continuous X
X <- round(matrix(rnorm(n * p), n, p),2)
colnames(X) <- letters[1:ncol(X)]
Y <- matrix(0, n, d)
best.tree <- policytree:::make_tree(X, depth = depth, d = d)
best.action <- policytree:::predict_test_tree(best.tree, X)
Y[cbind(1:n, best.action)] <- 100 * runif(n)
best.reward <- sum(Y[cbind(1:n, best.action)])

tree <- sparse_policy_tree(X,Y,2)

tree
#> policy_tree object 
#> Tree depth:  2 
#> Actions:  1 2 3 
#> Variable splits: 
#> (1) split_variable: a  split_value: -0.42 
#>   (2) split_variable: c  split_value: -0.5 
#>     (4) * action: 1 
#>     (5) * action: 2 
#>   (3) split_variable: a  split_value: 0.36 
#>     (6) * action: 2 
#>     (7) * action: 3
```

# Installation

## Installing Rust:

You must have [Rust](https://www.rust-lang.org/tools/install) installed
to compile this package. The rust website provides an excellent
installation script that has never caused me any issues.

On Linux, you can install Rust with:

``` sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

On Windows, I use the rust installation wizard, found
[here](https://forge.rust-lang.org/infra/other-installation-methods.html).

Once you install rust, you should be able to install the package with:

``` r
devtools::install_github("Yale-Medicaid/sparsepolicytree")
```

## Benchmarks:

| Number of Observations | Number of Distinct Predictor Values | Number of Predictors | Number of Treatments | Time   |
|------------------------|-------------------------------------|----------------------|----------------------|--------|
| 10^2                   | 2                                   | 30                   | 20                   | 0.003s |
| 10^3                   | 2                                   | 30                   | 20                   | 0.015s |
| 10^4                   | 2                                   | 30                   | 20                   | 0.15s  |
| 10^5                   | 2                                   | 30                   | 20                   | 1.57s  |
| 10^6                   | 2                                   | 30                   | 20                   | 17s    |
| 10^2                   | 30                                  | 30                   | 20                   | 0.052s |
| 10^3                   | 30                                  | 30                   | 20                   | 0.240s |
| 10^4                   | 30                                  | 30                   | 20                   | 3.041s |
| 10^5                   | 30                                  | 30                   | 20                   | 100s   |
| 10^6                   | 30                                  | 30                   | 20                   | 21 min |
| 10^2                   | 10^2                                | 30                   | 20                   | 0.41s  |
| 10^3                   | 10^3                                | 30                   | 20                   | 40s    |
| 10^4                   | 10^3                                | 30                   | 20                   | 70 min |
