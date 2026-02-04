### Fit DLNMs using mgcv and perform posterior predictive model checking.
### This reflects the models expansion of the Poisson model fitted to 
### the Thessaloniki data in the paper, but it does so to the ChicagoNMAPS 
### data set from the dlnm package.

## load R libraries
library(dlnm)
library(ggplot2)
library(viridis)
library(mgcv)
library(data.table)
library(plot3D)
library(readr)


## The data set
head(chicagoNMMAPS)
## get cleaner version
dat <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp,DoW=chicagoNMMAPS$dow,
                  Date=chicagoNMMAPS$date,timeStep=chicagoNMMAPS$time/max(chicagoNMMAPS$time),
                  DoY=chicagoNMMAPS$doy,year=chicagoNMMAPS$year)
dat
# need to create extra columns, one for each lag
n <- nrow(dat)
max_lag <- 20
for(i in 1:max_lag){
  dat[,paste("temp",i,sep="") := c(rep(NA,i),temp0[1:(n-i)]) ]
}
## set up the covariates in the right way
index <- grep("temp", colnames(dat) )
tempDat <- na.omit( dat )
TEMP <- as.matrix( tempDat[,..index] )
LAG <- matrix(0:max_lag,nrow(TEMP),length(0:max_lag),byrow=TRUE) 
W <- matrix(1,length(0:max_lag),nrow=nrow(LAG),ncol=ncol(LAG)) 
## fit gam with 10 knots for the lag dimension and 10 knots for Tapp. 
poissonDLNM <- gam(y~te(TEMP,LAG,by=W,k=c(10,10)),family=poisson,method="REML",data=tempDat) 


### 1. Check for overdispersion using the chi^2 test
1 - pchisq(poissonDLNM$deviance,poissonDLNM$df.residual)
# so model is overdispersed.

### 2. Check for overdispersion using summary statistics of the data and from
### posterior predictive samples
# First, generate posterior predictive samples from the Poisson model
n.sims <- 1000
b <- rmvn(n.sims,coef(poissonDLNM),poissonDLNM$Vp)
MM <- predict(poissonDLNM, type="lpmatrix")
MEAN <- exp( tcrossprod(MM,b) )
# now simulate from Poisson
PREDSpois <- apply(MEAN,2,function(x){rpois(length(x),lambda=x)})
# summary statistics 
statSamplePois <- data.table(
  Mean = apply(PREDSpois,2,mean),
  Median = apply(PREDSpois,2,median),
  Var = apply(PREDSpois,2,var),
  IQR = apply(PREDSpois,2,IQR),
  q1 = apply(PREDSpois,2,quantile,probs=0.01),
  q99 = apply(PREDSpois,2,quantile,probs=0.99)
)
# Now fit a Negative Binomial model and predict in the same way
negbinDLNM <- gam(y~te(TEMP,LAG,by=W,k=c(10,10)),family=nb,method="REML",data=tempDat) 
b <- rmvn(n.sims,coef(negbinDLNM),negbinDLNM$Vp)
myTheta <- negbinDLNM$family$getTheta(TRUE)
MM <- predict(negbinDLNM, type="lpmatrix")
MEAN <- exp( tcrossprod(MM,b) )
## now simulate from negbin
PREDSnb <- apply(MEAN,2,function(x){rnbinom(length(x),mu=x,size=myTheta)})
## summary statistics 
statSampleNegBin <- data.table(
  Mean = apply(PREDSnb,2,mean),
  Median = apply(PREDSnb,2,median),
  Var = apply(PREDSnb,2,var),
  IQR = apply(PREDSnb,2,IQR),
  q1 = apply(PREDSnb,2,quantile,probs=0.01),
  q99 = apply(PREDSnb,2,quantile,probs=0.99)
)
# summary statistics of the observed counts
mean(tempDat$y)
median(tempDat$y)
var(tempDat$y)
IQR(tempDat$y)
quantile(tempDat$y,probs=c(0.01,0.99))
# Copute posterior mean and 95% prediction interval of each 
# summary statistic, for each of the 2 models
# Poisson
apply(statSamplePois,2,mean)
apply(statSamplePois,2,quantile,probs=c(0.025,0.975))
# Negative Binomial
apply(statSampleNegBin,2,mean)
apply(statSampleNegBin,2,quantile,probs=c(0.025,0.975))

# Note that even the Negative Binomial underestimates the variance!
# Probably because of other structures in the model, e.g. temporal


### 4. Distributional assumption
# compute quantiles of the predicted "data sets" and compare against obs
myQuants <- seq(0,1,len=200)
predictedQuantiles <- apply(PREDSnb,2,quantile,probs=myQuants)
QQdatNB <- data.table(
  Mean = apply(predictedQuantiles,1,mean),
  Upper = apply(predictedQuantiles,1,quantile,probs=0.975),
  Lower = apply(predictedQuantiles,1,quantile,probs=0.025),
  Obs = quantile(tempDat$y,probs=myQuants)
)
QQplotNB <- ggplot(data=QQdatNB) + geom_ribbon(aes(x=Obs,ymin=Lower,ymax=Upper),fill="grey70") + 
  geom_point(aes(x=Obs,y=Mean)) + geom_abline(slope=1,intercept = 0) + 
  theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) +
  ylab("Predicted quantiles") + xlab("Observed quantiles")
QQplotNB
## Not so great at lower and upper extremes (note the max. value of 312 is very extreme)



### 3. Check temporal structure
# start with comparing sample autocorrelation with autocorrelation from
# predictions of the negative binomial model
maxACFlag <- 25
obsACF <- acf(tempDat$y,plot = F,lag.max = maxACFlag)$acf[-1]
predsACF <- apply(PREDSnb,2,function(x){acf(x,lag.max=maxACFlag,plot=F)$acf[-1]})
ACFdat <- data.table(obs=obsACF,mean=apply(predsACF,1,mean),lower=apply(predsACF,1,quantile,probs=0.025),
                     upper=apply(predsACF,1,quantile,probs=0.975),Lag=1:maxACFlag)
ACFbaseline <- ggplot(data=ACFdat,aes(x=Lag)) + geom_point(aes(y=obs),pch=4) + geom_point(aes(y=mean)) + 
  geom_errorbar(aes(ymin = lower, ymax=  upper), width=0.5) + ylim(0,0.5) + 
  theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) + 
  xlab("Lag (days)") + ylab("Auto-correlation") 
ACFbaseline
# so strong underestimation of the ACF
# Some plots to investigate temporal structure in the observed counts
# day of year:
boxplot(tempDat$y~tempDat$DoY) # clear seasonality. Note freaky extremes!
boxplot(tempDat$y~tempDat$DoY,ylim=c(0,100)) # exclude extremes from plot
# day-of-week:
boxplot(tempDat$y~tempDat$DoW) # not clear
# exclude extremes from plot
boxplot(tempDat$y~tempDat$DoW,ylim=c(0,100)) # weak signal
# daily time step
ts.plot(tempDat$y,ylim=c(0,100))
# yearly time
plot(tempDat$year,tempDat$y,ylim=c(0,100)) # slight decrease?
# Fit a negative binomial model with temporal structures
# Note the 3rd argument in m() controls the power parameter of the power
# exponential Gaussian process, so can be used to tune the amount of 
# autocorrelation that is captured.
negbinDLNMtemporal <- gam(y~te(TEMP,LAG,by=W,k=c(10,10)) + s(DoY,bs="cc",k=50) + s(DoW,bs="re") +
                      s(timeStep,k=200,bs="gp",m=c(2,-1,0.01)) + s(year,k=9),
                  family=nb,method="REML", data=tempDat,knots=list(DoY=c(0, 366))) 
# simulate from the posterior predictive distribution
b <- rmvn(n.sims,coef(negbinDLNMtemporal),negbinDLNMtemporal$Vp)
myTheta <- negbinDLNMtemporal$family$getTheta(TRUE)
MM <- predict(negbinDLNMtemporal, type="lpmatrix")
MEAN <- exp( tcrossprod(MM,b) )
## now simulate from negbin
PREDSnb_temporal <- apply(MEAN,2,function(x){rnbinom(length(x),mu=x,size=myTheta)})
## check ACF 
predsACF <- apply(PREDSnb_temporal,2,function(x){acf(x,lag.max=maxACFlag,plot=F)$acf[-1]})
ACFdat2 <- data.table(obs=obsACF,mean=apply(predsACF,1,mean),lower=apply(predsACF,1,quantile,probs=0.025),
                     upper=apply(predsACF,1,quantile,probs=0.975),Lag=1:maxACFlag)
# and plot together
ACFdat$model <- "Neg. Bin. DLNM"
ACFdat2$model <- "Neg. Bin. DLNM with temporal structures"
ACFplotDat <- rbind(ACFdat,ACFdat2)
ACFplot <- ggplot(data=ACFplotDat,aes(x=Lag)) + geom_point(aes(y=obs),pch=4) + geom_point(aes(y=mean)) + 
  geom_errorbar(aes(ymin = lower, ymax=  upper), width=0.5) + ylim(0,0.5) + 
  theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold")) + 
  xlab("Lag (days)") + ylab("Auto-correlation") + facet_wrap(~model)
ACFplot
# much better now, except for fist lag.. might be improved further if
# we increase the knots of the timeStep spline
summary(negbinDLNMtemporal)

