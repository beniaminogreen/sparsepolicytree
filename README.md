
# Parallel Policy Tree

An R / Rust Package to implement a parallelized exhaustive tree search
for policy-learning. Aims to extend and speed up work done with the
`policytree` package.

# Usage

There’s just one function, `exhaustive_tree()`. Use it just as you would
the `policy_tree()` function from the `policytree` package, and enjoy
the parallelized speed!

Trees aren’t guaranteed to be exactly the same as those produced by
`policytree` in that some leaves may be left unpruned (working on this),
but they should give the same predictions.

``` r

n <- 400
p <- 4
d <- 3

# Classification task taken from policytree tests
X <- round(matrix(rnorm(n * p), n, p),2)
Y <- matrix(0, n, d)

tree <- exhaustive_tree(X,Y,2)

tree
#> policy_tree object 
#> Tree depth:  2 
#> Actions:  1 2 3 
#> Variable splits: 
#> (1) split_variable:   split_value: -2.6 
#>   (2) split_variable:   split_value: 1.44 
#>     (4) * action: 1 
#>     (5) * action: 1 
#>   (3) split_variable:   split_value: -3.36 
#>     (6) * action: 1 
#>     (7) * action: 1
```

# Installation

You must have [Rust](https://www.rust-lang.org/tools/install) installed
to compile this package.
