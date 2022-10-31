
# Parallel Policy Tree

An R / Rust Package to implement a parallelized exhaustive tree search
for policy-learning. Aims to extend and speed up work done with the
`policytree` package.

# Usage

There’s just one function, `parallel_policy_tree()`. Use it just as you
would the `policy_tree()` function from the `policytree` package, and
enjoy the parallelized speed!

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

tree <- parallel_policy_tree(X,Y,2)

tree
#> policy_tree object 
#> Tree depth:  2 
#> Actions:  1 2 3 
#> Variable splits: 
#> (1) split_variable: b  split_value: 0.21 
#>   (2) split_variable: d  split_value: 0.09 
#>     (4) * action: 2 
#>     (5) * action: 1 
#>   (3) split_variable: c  split_value: 1.4 
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

Once you install rust, you should be able to install the package with
`devtools::install_github("Yale-Medicaid/parallel_policy_tree")`.
