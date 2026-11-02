suppressPackageStartupMessages(library(relevent))
script <- sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1])
validation_dir <- dirname(normalizePath(script))
base <- file.path(validation_dir, 'results')
models <- read.csv(file.path(base,'models.csv'))
results <- list()
for (i in seq_len(nrow(models))) {
  z <- models[i,]
  ev <- as.matrix(read.csv(file.path(base,paste0(z$label,'_events.csv'))))
  # rem.dyad uses its final row for censoring; retain all actual events.
  final <- ev[nrow(ev),1]
  ev <- rbind(ev,c(final + max(abs(final),1)*.Machine$double.eps,NA,NA))
  con <- file(file.path(base,paste0(z$label,'.bin')),'rb')
  cov <- array(readBin(con,'double',n=(z$m+1)*z$p*z$n*z$n,size=8,endian='little'),c(z$m+1,z$p,z$n,z$n))
  close(con)
  fit <- rem.dyad(ev,n=z$n,effects='CovEvent',covar=list(CovEvent=cov),ordinal=FALSE,fit.method='MLE',conditioned.obs=0,hessian=TRUE,coef.seed=rep(0,z$p),optim.method='BFGS',optim.control=list(reltol=1e-12,maxit=if(z$label=='simulation') 200 else 10000),verbose=FALSE)
  saveRDS(fit,file.path(base,paste0(z$label,'_R.rds')))
  results[[i]] <- data.frame(label=z$label,term=seq_len(z$p),coef=fit$coef,se=sqrt(diag(fit$cov)),loglik=-fit$residual.deviance/2,bic=fit$BIC,convergence=fit$convergence)
  print(results[[i]])
  write.csv(do.call(rbind,results),file.path(base,'r_results.csv'),row.names=FALSE)
}
capture.output(sessionInfo(),file=file.path(base,'R_session.txt'))
