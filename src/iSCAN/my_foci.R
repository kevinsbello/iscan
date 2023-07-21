library(FOCI)

find_par <- function(X,shift_index_m,topo_order_m){
  shift_index <- unlist(shift_index_m)
  topo_order <- unlist(topo_order_m)
  #print(topo_order)
  
  d <- dim(X)[2]
  adj <- matrix(0,d,d)
  for (i in shift_index){
    if (which(topo_order==i)==1){next}
    colnames(X) <- 1:d
    fw <- topo_order[1:(which(topo_order==i)-1)]
    if (which(topo_order==i)>2){
      foci_res <- foci(Y=X[,i],X=X[,fw],numCores =1,standardize ="scale",stop = T)
      foci_select <- foci_res$selectedVar$names
      if (is.null(foci_select)){next}
    } else {
      foci_res <- codec(Y=X[,i],Z=X[,fw])
      if (foci_res < 0){next}
      else {foci_select <- fw}
    }
    par <- as.integer(foci_select)
    adj[par,i] <- 1
  }
  return (adj)
}