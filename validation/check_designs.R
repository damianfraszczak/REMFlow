# Independent construction of every risk-set covariate from events and raw data.
script <- sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1])
validation_dir <- dirname(normalizePath(script))
base <- file.path(validation_dir, 'results')
data_dir <- file.path(dirname(validation_dir), 'data', 'classroom_events')
models <- read.csv(file.path(base,'models.csv'))
out <- list()
for (ii in seq_len(nrow(models))) {
  z <- models[ii,]; n <- z$n; m <- z$m; p <- z$p
  ev <- as.matrix(read.csv(file.path(base,paste0(z$label,'_events.csv'))))
  con <- file(file.path(base,paste0(z$label,'.bin')),'rb')
  supplied <- array(readBin(con,'double',n=(m+1)*p*n*n,size=8,endian='little'),c(m+1,p,n,n)); close(con)
  independent <- array(0,c(m+1,p,n,n))
  simulation <- z$label == 'simulation'
  if (!simulation) {
    day <- substr(z$label,1,5); model <- substr(z$label,7,nchar(z$label))
    raw <- read.table(file.path(data_dir,paste0('class_interactions_',day,'.txt')),header=TRUE)
    raw <- raw[raw$to_all_col==0 & raw$from_all_col==0,]
    raw <- raw[order(raw$time_estimate_col),]
    stopifnot(isTRUE(all.equal(unname(as.matrix(raw[,c('time_estimate_col','send_col','receive_col')])),unname(ev),tolerance=1e-12)))
    a <- read.table(file.path(data_dir,'class_attributes.txt'),header=TRUE)
    seating <- matrix(0,n,n); friendship <- matrix(0,n,n)
    edges <- read.table(file.path(data_dir,paste0('class_seating_',day,'.txt')),header=TRUE)
    seating[as.matrix(edges)] <- 1; seating <- 1*((seating+t(seating))>0)
    edges <- read.table(file.path(data_dir,'class_edgelist_sem2.txt'),header=TRUE)
    friendship[as.matrix(edges)] <- 1
    terms <- switch(model,mod1='Intercept',mod2a=c('Intercept','Sender_male','Receiver_male'),mod3b=c('Intercept','Sender_male','Receiver_male','Seating','Friendship'),mod4f=c('Recency_ji','Recency_ij','Intercept','Sender_male','Sender_teacher','Receiver_male','Receiver_teacher','Seating','Friendship','PSAB_BA','PSAB_BY','PSAB_XA','PSAB_XB','PSAB_AY'))
  }
  counts <- matrix(0,n,n); last_send <- rep(NA_integer_,n)
  for (e in seq_len(m)) {
    for (s in seq_len(n)) for (r in seq_len(n)) if (s!=r) {
      A <- if(e>1) ev[e-1,2] else NA; B <- if(e>1) ev[e-1,3] else NA
      ba <- if(e>1) as.numeric(s==B && r==A) else 0
      if (simulation) {
        vals <- c(1,ba,if(is.na(last_send[s])) 0 else 1/(e-last_send[s]+1))
      } else {
        vals <- c(Intercept=1,Sender_male=as.numeric(a$gnd[s]==1),Receiver_male=as.numeric(a$gnd[r]==1),Sender_teacher=as.numeric(a$grd[s]==16),Receiver_teacher=as.numeric(a$grd[r]==16),Seating=seating[s,r],Friendship=friendship[s,r],Recency_ji=counts[r,s],Recency_ij=counts[s,r],PSAB_BA=ba,PSAB_BY=if(e>1) as.numeric(s==B && r!=A) else 0,PSAB_XA=if(e>1) as.numeric(s!=A && s!=B && r==A) else 0,PSAB_XB=if(e>1) as.numeric(s!=A && s!=B && r==B) else 0,PSAB_AY=if(e>1) as.numeric(s==A && r!=B) else 0)[terms]
      }
      independent[e,,s,r] <- vals
    }
    counts[ev[e,2],ev[e,3]] <- counts[ev[e,2],ev[e,3]]+1
    last_send[ev[e,2]] <- e
  }
  difference <- max(abs(independent-supplied))
  stopifnot(difference<1e-12)
  out[[ii]] <- data.frame(model=z$label,max_abs_design_difference=difference)
}
write.csv(do.call(rbind,out),file.path(base,'independent_design_checks.csv'),row.names=FALSE)
print(do.call(rbind,out))
