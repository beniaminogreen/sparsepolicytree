#' Return string `"Hello world!"` to R.
#' @export
sparse_policy_tree <- function(X, Gamma, depth) {
  n_obs <- nrow(X)
  valid_classes <- c("matrix")

  # Checks copied from `policytree` package
  # https://github.com/grf-labs/policytree/blob/master/r-package/policytree/R/policy_tree.R
  if (!inherits(X, valid_classes) || !inherits(Gamma, valid_classes)) {
    stop(paste(
      "Currently the only supported data input types are:",
      "`matrix`"
    ))
  }
  if (!is.numeric(as.matrix(X)) || any(dim(X) == 0)) {
    stop("The feature matrix X must be numeric")
  }
  if (!is.numeric(as.matrix(Gamma)) || any(dim(Gamma) == 0)) {
    stop("The reward matrix Gamma must be numeric")
  }
  if (anyNA(X)) {
    stop("Covariate matrix X contains missing values.")
  }
  if (anyNA(Gamma)) {
    stop("Gamma matrix contains missing values.")
  }
  if (depth < 0) {
    stop("`depth` cannot be negative.")
  }
  if (n_obs != nrow(Gamma)) {
    stop("X and Gamma does not have the same number of rows")
  }

  if (!is.double(X)) {
      class(X) <- "double"
  }
  if (!is.double(Gamma)) {
      class(Gamma) <- "double"
  }

  node_list <- rust_exhaustive_tree(X, Gamma, depth)

  tree_array <- matrix(0, nrow = length(node_list), 4)
  for (i in seq(node_list)) {
    node <- node_list[[i]]
    if (node$is_leaf) {
      tree_array[i, 1] <- -1
      tree_array[i, 2] <- node$action
    } else {
      tree_array[i, 1] <- node$split_variable
      tree_array[i, 2] <- node$split_value
      tree_array[i, 3] <- node$left_child
      tree_array[i, 4] <- node$right_child
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
