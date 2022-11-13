#' Parallelized exhaustive policy tree search. Optimized for sparse Features
#'
#' @param X The covariates used. Dimension \eqn{N*p} where \eqn{p} is the number of features.
#' @param Gamma The rewards for each action. Dimension \eqn{N*d} where \eqn{d} is the number of actions.
#' @param depth The depth of the fitted tree. Default is 2.
#'
#' @references Athey, Susan, and Stefan Wager. "Policy Learning With Observational Data."
#'  Econometrica 89.1 (2021): 133-161.
#' @references Sverdrup, Erik, Ayush Kanodia, Zhengyuan Zhou, Susan Athey, and Stefan Wager.
#'  "policytree: Policy learning via doubly robust empirical welfare maximization over trees."
#'   Journal of Open Source Software 5, no. 50 (2020): 2232.
#' @references Zhou, Zhengyuan, Susan Athey, and Stefan Wager. "Offline multi-action policy learning:
#'  Generalization and optimization." Operations Research, forthcoming.
#'  @examples
#' \donttest{
#' # Fit a depth two tree on doubly robust treatment effect estimates from a causal forest.
#' # (Example taken from policytree documentation)
#' n <- 10000
#' p <- 10
#' # Discretizing continuous covariates decreases runtime.
#' X <- round(matrix(rnorm(n * p), n, p), 2)
#' colnames(X) <- make.names(1:p)
#' W <- rbinom(n, 1, 1 / (1 + exp(X[, 3])))
#' tau <- 1 / (1 + exp((X[, 1] + X[, 2]) / 2)) - 0.5
#' Y <- X[, 3] + W * tau + rnorm(n)
#' c.forest <- grf::causal_forest(X, Y, W)
#' dr.scores <- double_robust_scores(c.forest)
#'
#' tree <- parallel_policy_tree(X, dr.scores, 2)
#' tree
#'
#' # Predict treatment assignment.
#' predicted <- predict(tree, X)
#'
#' plot(X[, 1], X[, 2], col = predicted)
#' legend("topright", c("control", "treat"), col = c(1, 2), pch = 19)
#' abline(0, -1, lty = 2)
#'
#' # Predict the leaf assigned to each sample.
#' node.id <- predict(tree, X, type = "node.id")
#' # Can be reshaped to a list of samples per leaf node with `split`.
#' samples.per.leaf <- split(1:n, node.id)
#'
#' # The value of all arms (along with SEs) by each leaf node.
#' values <- aggregate(dr.scores, by = list(leaf.node = node.id),
#'                     FUN = function(x) c(mean = mean(x), se = sd(x) / sqrt(length(x))))
#' print(values, digits = 2)
#'
#' @export
parallel_policy_tree <- function(X, Gamma, depth=2){
    node_list <- rust_exhaustive_tree(X, Gamma, depth)

    tree_array <- matrix(0, nrow=length(node_list), 4)
    for (i in seq(node_list)){
        node <- node_list[[i]]
        if (node$is_leaf) {
            tree_array[i,1] <- -1
            tree_array[i,2] <- node$action
        } else {
            tree_array[i,1] <- node$split_variable
            tree_array[i,2] <- node$split_value
            tree_array[i,3] <- node$left_child
            tree_array[i,4] <- node$right_child
        }
    }

    output <- list(
                   nodes = node_list,
                   `_tree_array` = tree_array,
                   depth = depth,
                   n.actions = ncol(Gamma),
                   n.features = ncol(X),
                   action.names = colnames(Gamma),
                   columns = colnames(X)
    )
    class(output) <- "policy_tree"
    return(output)
}

