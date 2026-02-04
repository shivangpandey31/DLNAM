## Fit a DLNM as a GAM to study the synergy between temperature and PM10

library(data.table)
library(mgcv)
library(ggplot2)
library(viridis)
library(dlnm)
library(readr)

## The data set
head(chicagoNMMAPS)
## get cleaner version
dat <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp,PM10_0=chicagoNMMAPS$pm10)
dat

# compute the lagged covariates -- be careful with this, make sure it is 
# doing the right thing
max_lag <- 20 # probably too high for PM10 but OK to exemplify
n <- nrow(dat)
for(i in 1:max_lag){
  dat[,paste("temp",i,sep="") := c(rep(NA,i),temp0[1:(n-i)]) ]
  dat[,paste("PM10_",i,sep="") := c(rep(NA,i),PM10_0[1:(n-i)]) ]
}
# check
dat

# Set up covariate matrices
tempDat <- na.omit( dat )
LAG <- matrix(0:max_lag,nrow(tempDat),length(0:max_lag),byrow=TRUE) 
TmeanIndex <- grep("temp", colnames(tempDat) )
TEMP <- as.matrix( tempDat[,..TmeanIndex] )
PM10Index <- grep("PM10", colnames(tempDat) )
PM10 <- as.matrix( tempDat[,..PM10Index] )
L <- matrix(1,nrow=nrow(LAG),ncol=ncol(LAG))
# and fit model
model <- gam(y~te(TEMP,PM10,LAG,by=L,k=10),family=nb,data=tempDat) 
# Faster with but careful with nthreads, make sure there are enough
#model <- bam(y~te(TEMP,PM10,LAG,by=L,k=10),family=nb,data=tempDat,method="fREML",discrete=T,nthreads = 12) 

# Compare the intercept
exp(coef(model)[1] )
## with the mean number of deaths per day
mean(tempDat$y )
## which is the "baseline" for relative risk


## 3D plot of "effects"
range(tempDat$temp0)
range(tempDat$PM10_0)
quantile(tempDat$PM10_0,probs=c(0.05,0.25,0.5,0.75,0.95))
pm10Grid <- 9:71
tempGrid <- -25:33
dlnm_grid <- data.table(expand.grid(tempGrid,pm10Grid,0:max_lag))
names(dlnm_grid) <- c("TEMP","PM10","LAG")
dlnm_grid$L <- 1

# plot temp-lag relative risk for different quantiles of PM10
myPM10 <- round( quantile(tempDat$PM10_0,probs=c(0.05,0.25,0.5,0.75,0.95)),0)
plotGrid <- dlnm_grid[PM10 %in% myPM10]
plotGrid[,PM10name:=paste("PM10 = ",PM10,sep="")]
modelMatrix <- predict(model,newdata=plotGrid,type="lpmatrix")[,-1]
logRR <- ( modelMatrix %*% coef(model)[-1] )[,1]
plotGrid$RR <- exp( logRR )
# uncertainty
n.sims <- 1000
b <- rmvn(n.sims,coef(model),model$Vp)[,-1]
X <- predict(model,newdata=plotGrid,type="lpmatrix")[,-1]
LP <- tcrossprod(X ,b)
Lower <- apply(LP,1,quantile,probs=0.025)
Upper <- apply(LP,1,quantile,probs=0.975)
myDF <- data.frame(Upper,Lower)
test <- as.numeric(with(myDF,Upper>=0 & Lower<=0)) # is zero in the 95% CrI?
plotGrid$sign <- test
Sign <- plotGrid[sign==0]
# ggplot raster plot
range(plotGrid$RR)
myBrks <- seq(0.88,1.42,by=0.02) 
EffectSurface <- ggplot(data=plotGrid) + geom_raster(aes(x=TEMP, y = LAG, fill=RR)) + 
  scale_fill_gradient2(limits=range(myBrks),breaks=myBrks,midpoint=1,high="darkred",low="darkblue") + 
  ylab("Lag (days)") + xlab("Temperature (\u00B0C)")  + geom_point(data=Sign,aes(x=TEMP,y=LAG),alpha=0.05) + 
  facet_wrap(~PM10name,ncol=3) + theme(legend.key.height = unit(1.5, 'cm')) + theme_bw() +
  theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold"),legend.text=element_text(size=12),
        legend.key.height = unit(2.2, 'cm'))
EffectSurface
# A spike at high temperatures at high PM10


### Cumulative risk surface
modelMatrix <- predict(model,newdata=dlnm_grid,type="lpmatrix")[,-1]
b <- rmvn(1000,model$coefficients,model$Vp)
logRR <- data.table( tcrossprod(modelMatrix , b[,-1] ) )
logRR[,TEMP:=dlnm_grid$TEMP]
logRR[,PM10:=dlnm_grid$PM10]
CR <- logRR[,lapply(.SD,function(x){exp(sum(x))}),by=c("TEMP","PM10")]
plotDat <- data.table(
  lower = apply( CR[,-c("TEMP","PM10")],1,quantile,probs=0.025 ),
  upper = apply( CR[,-c("TEMP","PM10")],1,quantile,probs=0.975 ) ,
  CR = apply( CR[,-c("TEMP","PM10")],1,mean ),
  CR[,c("TEMP","PM10")]
)
## significance
test <- as.numeric(with(plotDat[,c("upper","lower")],upper>=1 & lower<=1)) # is 1 in the 95% CrI?
plotDat$sign <- test
Sign <- plotDat[sign==0]
# ggplot surface plot
range(plotDat$CR)
myBrks <- seq(0.5,10,by=0.5) ## relative risk 
EffectSurface <- ggplot(data=plotDat) + geom_raster(aes(x=TEMP, y = PM10, fill=CR)) + 
  scale_fill_gradient2(limits=range(myBrks),breaks=myBrks,midpoint=1,high="darkred",low="darkblue") + xlab("temperature") +
  ylab("PM10") + geom_point(data=Sign,aes(x=TEMP,y=PM10),alpha=0.05) +  theme_bw() +
  theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold"),legend.text=element_text(size=12),
        legend.key.height = unit(2, 'cm'))
EffectSurface


################################################################################
### Compute attributable fraction and number
### We will do this for given values of PM10 (see below)

################################################################################
### AF for non-optimum Tmax ranges
datHere <- data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp,PM10_0=chicagoNMMAPS$pm10,date=chicagoNMMAPS$date)
for(i in 1:max_lag){
  datHere[,paste("temp",i,sep="") :=temp0 ]
  datHere[,paste("PM10",i,sep="") :=PM10_0 ]
}
datHere <- na.omit(datHere) # since PM10 has missing values
# now put in long format for gam() prediction
longDat <- melt(datHere,id.vars=c("date"),measure.vars = list(grep("temp", colnames(datHere) ),grep("PM10", colnames(datHere) )))
# create the lag variable
names(longDat)[which(names(longDat)=="variable")] <- "LAG"
longDat[,LAG := as.numeric(LAG)-1]
names(longDat)[which(names(longDat)=="value1")] <- "TEMP"
names(longDat)[which(names(longDat)=="value2")] <- "PM10"
longDat$L <- 1
X <- predict(model,newdata=longDat,type="lpmatrix") 
b <- model$coefficients
logRRaf <- X[,-1] %*% b[-1] 
## now center on optimum temperature-PM10
dum <- copy(longDat)
OTPM10 <- plotDat[which.min(CR),c("TEMP","PM10")]
dum[,TEMP:=OTPM10$TEMP]
dum[,PM10:=OTPM10$PM10]
X_OTPM10 <- predict(model,newdata=dum,type="lpmatrix") 
logRR_OTPM10 <- X_OTPM10[,-1] %*% b[-1]
logRRcen <- logRRaf - logRR_OTPM10
longDat <- cbind( longDat, logRRcen)
## now compute eq 6 of Gasp. and Leone (2014):
fAF <- longDat[,1-exp(-sum(V1)) ,by=c("date","PM10","TEMP")]
## and merge with original data.table
datOrig <- na.omit( data.table(y=chicagoNMMAPS$cvd,temp0=chicagoNMMAPS$temp,PM10_0=chicagoNMMAPS$pm10,date=chicagoNMMAPS$date) )
datOrig <- merge(datOrig,fAF,by=c("date") )
## now compute fAN
datOrig[,fAN:=V1*y]
# now totals (eq. 8a and 8b)
ANtot <- sum(datOrig$fAN)
AFtot <- ANtot/sum(datOrig$y)
####################################################
### AF again but for various temp-PM10 regions
tempTholds <- c(-1000, quantile(datOrig$temp0,probs=c(0.05,0.25,0.75,0.95)), 1000 )
PM10tholds <- c(0,quantile(datOrig$PM10,probs=c(0.25,0.75)),1000 )
## compute Attr. number but keep Tmax and RH columns
tempNames <- c("Extreme Cold","Mild Cold","Mild Heat","Extreme Heat")
pm10Names <- c("Low","Medium","High")
results2 <- data.table(AF=runif(12),Temperature=rep(tempNames,length(pm10Names)),
                       PM10condition=rep(pm10Names,each=length(tempNames)))
for(i in 1:length(pm10Names)){
  for(j in 1:length(tempNames)){
    index <- datOrig$temp0>=tempTholds[j] & datOrig$temp0<tempTholds[j+1] & datOrig$PM10_0>=PM10tholds[i] & datOrig$PM10_0<PM10tholds[i+1]
    fAN_subset <- datOrig[index,fAN]
    ANtot <- sum(fAN_subset)
    AFtot <- ANtot/sum(datOrig[index]$y)
    results2[Temperature==tempNames[j] & PM10condition==pm10Names[i],AF:=AFtot]
  }
}
## Barplot
results2[,Temperature:=factor(Temperature,levels=tempNames)]
results2[,PM10condition:=factor(PM10condition,levels=pm10Names)]
barPlot <- ggplot(results2, aes(fill=Temperature, y=AF, x=PM10condition)) + 
  geom_bar(position="stack", stat="identity") +
  scale_fill_viridis(discrete = T) +  ggtitle("") + xlab("PM10 Level") + ylab("Attributable fraction")+
  theme_bw() + 
  theme(axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold"),legend.text=element_text(size=12),legend.key.height = unit(1.5, 'cm'))
barPlot




