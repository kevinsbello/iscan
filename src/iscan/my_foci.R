library(FOCI)

find_par <- function(X, shifted_nodes_m, order_m){
    shifted_nodes <- unlist(shifted_nodes_m)
    order <- unlist(order_m)
    d <- dim(X)[2]
    adj <- matrix(0, d, d)

    for (i in shifted_nodes) {
        pos = which(order==i)
        if (pos == 1) { next }
        colnames(X) <- 1:d
        fw <- order[1:(pos-1)]
        if (pos > 2) {
            foci_res <- foci(Y = X[,i], X = X[,fw], numCores = 1, standardize = "scale", stop = T)
            foci_select <- foci_res$selectedVar$names
            if (is.null(foci_select)) { next }
        } else {
            foci_res <- codec(Y = X[,i], Z = X[,fw])
            if (foci_res < 0) { next }
            else { foci_select <- fw }
        }
        par <- as.integer(foci_select)
        adj[par, i] <- 1
    }
    return (adj)
}