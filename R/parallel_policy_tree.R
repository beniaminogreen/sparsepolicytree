#' Return string `"Hello world!"` to R.
#' @export
parallel_policy_tree <- function(X, Y, depth){
    node_list <- rust_exhaustive_tree(X, Y, depth)

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
                   n.actions = ncol(Y),
                   n.features = ncol(X),
                   action.names = colnames(Y),
                   columns = colnames(X)
    )
    class(output) <- "policy_tree"
    return(output)
}

