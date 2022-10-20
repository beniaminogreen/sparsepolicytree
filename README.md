
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

plot(tree)
#> PhantomJS not found. You can install it with webshot::install_phantomjs(). If it is installed, please make sure the phantomjs executable can be found via the PATH variable.
```

<div id="htmlwidget-acf3093ff9afb68b6e5d" style="width:100%;height:480px;" class="grViz html-widget"></div>
<script type="application/json" data-for="htmlwidget-acf3093ff9afb68b6e5d">{"x":{"diagram":"digraph nodes { \n node [shape=box] ;\n0 [label=\"  <= -3.08 \"] ;\n0 -> 1 [labeldistance=2.5, labelangle=45, headlabel=\"True\"];\n0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel=\"False\"]\n1 [label=\"  <= 1 \"] ;\n1 -> 3  ;\n1 -> 4  ;\n3  [shape=box,style=filled,color=\".7 .3 1.0\" , label=\" leaf node\n action = 1 \"];\n4  [shape=box,style=filled,color=\".7 .3 1.0\" , label=\" leaf node\n action = 1 \"];\n2 [label=\"  <= -2.58 \"] ;\n2 -> 5  ;\n2 -> 6  ;\n5  [shape=box,style=filled,color=\".7 .3 1.0\" , label=\" leaf node\n action = 1 \"];\n6  [shape=box,style=filled,color=\".7 .3 1.0\" , label=\" leaf node\n action = 1 \"];\n}","config":{"engine":"dot","options":null}},"evals":[],"jsHooks":[]}</script>
