### This code fits a single-exposure DLNM to the chicagoNMMAPS data set
### included in the dlnm R package.
### First, it fits a quasi-Poisson DLNM using the dlnm package, then
### it fits a Negative Binomial DLNM as penalised GAM from mgcv,
### and lastly it fits the same penalised GAM using MCMC by utilising the
### function jagam and the R package nimble

# Load the necessary libraries
library(dlnm)
library(ggplot2)
library(viridis)
library(mgcv)
library(data.table)
library(splines)
library(plot3D)
library(readr)
library(nimble)
library(coda)

##############################################################
## 1. Use the dlnm package to fit a DLNM to the Chicago data. 
## Mortality counts vs temperature.
max_lag <- 15
cb1.temp <- crossbasis(chicagoNMMAPS$temp, lag=max_lag, argvar=list(df=10),
                       arglag=list(df=10))
summary(cb1.temp)
# fit the model
model1 <- glm(cvd ~ cb1.temp, family=quasipoisson(), chicagoNMMAPS)
# Compute the temperature effects for a temp-lag grid
temp_grid <- seq(-27,34,by=0.2)
lag_grid <- 0:max_lag
dlnm_grid <- data.table(expand.grid(temp_grid,lag_grid))
names(dlnm_grid) <- c("temp","lag")
# now compute the estimated relationship on the grid
pred1.temp <- crosspred(cb1.temp, model1, at=temp_grid, bylag=1, cumul=F, model.link = "log")
# note this excludes the intercept, uncomment if needed
dlnm_grid$effect <- exp( as.vector(pred1.temp$matfit) ) # + coef(model1)[1] )
x11()
persp3D(temp_grid, lag_grid, matrix(dlnm_grid$effect, length(temp_grid), length(lag_grid)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", 
        zlab = "RR", expand = 2/3, shade = 0.5)

###################################
###  2. Fit a NegBin DLNM in mgcv
# Set up the data in long format for gam()
dat <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp,date=chicagoNMMAPS$date)
n <- nrow(dat)
# create a new column for each lag
for(i in 1:max_lag){
  dat[,paste("temp",i,sep="") := c(rep(NA,i),temp0[1:(n-i)]) ]
}
# Need to set up matrices for the fit
tempDat <- na.omit( dat )
LAG <- matrix(0:max_lag,nrow(tempDat),length(0:max_lag),byrow=TRUE) 
index <- grep("temp", colnames(tempDat) )
TEMP <- as.matrix( tempDat[,..index] )
W <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
# fit gam with 10 knots for the lag dimension and 10 knots
# for temperature. 
gamDLNM <- gam(y~te(TEMP,LAG,by=W,k=c(10,10)),family=nb,method="REML",data=tempDat) 
## function to check if the maximum number of knots was adequate:
k.check(gamDLNM)
# More than enough. We gave it 99 parameters and it only needed about 41.
# Now 3D plot of "effects"
gam_grid <- copy(dlnm_grid)
# Names must match what went in the gam() call
names(gam_grid)[1:2] <- c("TEMP","LAG")
gam_grid$W <- 1
# Compute the estimated log-RR (as per paper) so excluding the intercept
logRR <- ( predict(gamDLNM,newdata=gam_grid,type="lpmatrix")[,-1] %*% coef(gamDLNM)[-1] )[,1]
gam_grid$RR <- exp( logRR ) #+ coef(gamDLNM)[1] )
x11()
persp3D(temp_grid, lag_grid, matrix(gam_grid$RR, length(temp_grid), length(lag_grid)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", 
        zlab = "RR", expand = 2/3, shade = 0.5,main="GAM DLNM")



########################################################################
## Now fit the Neg Bin GAM using MCMC in nimble. This uses the Bayesian 
## interpretation of penalised GAMs as per paper
# Function jagam is what translates the splines into priors, and produces 
# the BUGS-language code template (written in dum.jags, a text file)
# Note, the "family" argument is left at default, since we will manually 
# sort this later in the nimble code. All we need here are the model matric
# and the prior precision matrices which will be the same irrespective of the
# family.
jd <- jagam(y~te(TEMP,LAG,by=W,k=c(10,10)),file="dum.jags",data=tempDat) 
# The model matrix (excluding the intercept)
X <- jd$jags.data$X[,-1]
noCoefs <- ncol(X)  
noPenalties <- length(jd$pregam$sp)

# Now the nimble code to fit the model. Note the Negative Binomial 
# in nimble is not parameterised using the mean
Model <- nimbleCode({
  # linear predictor (log mu)
  eta[1:n] <- X[1:n,1:noCoefs] %*% b[1:noCoefs]
  # precision matrix of the prior for the coefficients (from dum.jags)
  K1[1:noCoefs,1:noCoefs] <- S1[1:noCoefs,1:noCoefs] * lambda[1] + S1[1:noCoefs,(noCoefs+1):(noCoefs*2)] * lambda[2] + S1[1:noCoefs,(noCoefs*2+1):(noCoefs*3)] * lambda[3]
  # The multivariate Normal prior
  b[1:noCoefs]  ~ dmnorm(zeros[1:noCoefs],K1[1:noCoefs,1:noCoefs]) 
  # Vague priors on the penalties
  for(k in 1:noPenalties){
    lambda[k] ~ dinvgamma(0.5,0.25)
  }
  # The conditional distribution
  for(i in 1:n){
    y[i] ~ dnegbin( p[i] , theta )
    log(mu[i]) <- beta0 + eta[i]
    p[i] <- theta/(theta+mu[i])
  }
  # The prior  size parameter of the Negatve Binomial
  theta ~ dexp(0.01)
  # and the intercept
  beta0 ~ dnorm(0,sd=50)
})
# Set up the lists of data and info for the nimble run
Consts <- list(n = jd$jags.data$n,noCoefs=noCoefs,noPenalties=noPenalties)
nimbleData <- list(y = jd$jags.data$y, X=X,S1=jd$jags.data$S1,
                   zeros=rep(0,noCoefs))
# Initial values (we use the fitted GAM estimates for quicker convergence)
inits <- list(b = gamDLNM$coef[-1],theta=1, lambda=rep(1,noPenalties),
              beta0=gamDLNM$coef[1])
nimbleModel <- nimbleModel(code=Model, name='nimbleModel', constants = Consts, 
                           data = nimbleData, inits = inits)
MCMCconfig <- configureMCMC(nimbleModel,monitors=c("b", "beta0","theta","lambda"))
# change the sampler of the coefficients to something more efficient
MCMCconfig$removeSamplers('b', print = FALSE)
MCMCconfig$addSampler(target = 'b', type = 'AF_slice')
modelMCMC <- buildMCMC(MCMCconfig)
compiled_model <- compileNimble(nimbleModel)
compiled_model_MCMC <- compileNimble(modelMCMC, project = nimbleModel)
# set up MCMC parameters
niter <- 10000
nburnin <- 5000
thin <- 5
# Do the MCMC. Note this takes a while -- over 1 hour per chain (need to 
# optimize..) The results object has been saved in the working folder so 
# no need to run.
results <- runMCMC(compiled_model_MCMC, niter = niter, nburnin = nburnin, nchains = 3, thin=thin,
                   inits=list(inits,inits,inits),setSeed = F, progressBar = T, samplesAsCodaMCMC = T)
# Trace plots of the MCMCM samples for each parameter
x11();plot(results,ask=T)
save(results,file="results.RData")
# Uncomment this if model was not run
# load("results.RData")
# get only the spline coefficients (excl. the intercept)
coefIndex <- grep( "b[", colnames( results[[1]] ),fixed=T)
betas <- results[,coefIndex]
betasMean <- apply(do.call(rbind,betas),2,mean)
# compute the relative risk and plot in 3D, as above
# Note, we use the fitted gam to produce the model matrix
logRR <- ( predict(gamDLNM,newdata=gam_grid,type="lpmatrix")[,-1] %*% betasMean )[,1]
gam_grid$RR_MCMC <- exp( logRR )
x11()
persp3D(temp_grid, lag_grid, matrix(gam_grid$RR_MCMC, length(temp_grid), length(lag_grid)), 
        theta = 200, phi = 20,ticktype = "detailed", xlab = "temp", ylab = "lag", 
        zlab = "RR", expand = 2/3, shade = 0.5,main="GAM DLNM using MCMC")


## Compare standard errors with the GAM from mgcv, to see what the effect is
## on the uncertainty from estimation of the penalty parameters.
betasMatrix <- do.call(rbind,betas)
betasGAM <- rmvn(3000,coef(gamDLNM),gamDLNM$Vp)[,-1]
logRR_MCMC <- tcrossprod( predict(gamDLNM,newdata=gam_grid,type="lpmatrix")[,-1], betasMatrix )
logRR_MCMC_std_err <- apply(logRR_MCMC,2,sd)
logRR_GAM <- tcrossprod( predict(gamDLNM,newdata=gam_grid,type="lpmatrix")[,-1], betasGAM )
logRR_GAM_std_err <- apply(logRR_GAM,2,sd)
# check the ratio between the standard errors 
summary( logRR_MCMC_std_err/logRR_GAM_std_err )
