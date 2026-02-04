rm(list = ls())

####################### Loading libraries
library(dplyr)
library(gnm)
library(dlnm)
library(ggplot2)
library(lubridate)
library(data.table)
library(mgcv)
library(plot3D)
library(viridis)
library(plot3D)
library(readr)
library(peakRAM)

df <- read.csv("case_time_series.csv")
# select only first 100 ids to reduce computation time
#df <- df %>% filter(id <= 200)
head(df)

temp_knots <- c(-2.510472, 16.020317, 19.799403)  # 10%, 75%, 90%
temp_b_knots <- c(-8.547488, 25.588451)           # 1%, 99%
cen <- 6.914142  

day_of_week <- factor(wday(df$date))

df$stratum <- factor(paste(df$id, month(df$date), sep = "-"))

peakRAM({
  cbtmean <- crossbasis(df$tmean, lag = 7,
                        argvar = list(fun = "ns", 
                                      knots = temp_knots, 
                                      Boundary.knots = temp_b_knots),
                        arglag = list(fun = "ns", 
                                      knots = logknots(7, nk = 3)),
                        group = df$id)

  # Model
  



mod <- gnm(outcome ~ cbtmean + day_of_week,
        eliminate = df$stratum,
             data = df,
            family = poisson)
})
summary(mod)

cptmean <- crosspred(cbtmean, mod, cen = cen, by = 1.5, cumul = TRUE)

png("DLNM_DLNMcumulativeIRR.png", width=800, height=600)
plot(cptmean, "overall", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", xlab = "Average Temperature (°C)", ci.arg = list(col = adjustcolor(1, alpha.f = 0.2)), ylim = c(0.5, 4))
dev.off()





# GAM MODEL
library(mgcv)
library(data.table)
library(lubridate)

dat <- as.data.table(df)
dat[, date := as.Date(date)]
dat[, stratum := factor(paste(id, month(date), sep = "-"))]
dat[, day_of_week := factor(wday(date))]

setorder(dat, id, date)

max_lag <- 7   # match your original crossbasis lag=7

dat[, temp0 := tmean]
for (l in 1:max_lag) {
  dat[, (paste0("temp", l)) := shift(temp0, n = l, type = "lag"), by = id]
}

index <- grep("^temp\\d+$|^temp0$", names(dat), value=TRUE)
tempDat <- na.omit(dat)

TEMP <- as.matrix(tempDat[, ..index])
LAG  <- matrix(0:max_lag, nrow=nrow(TEMP), ncol=ncol(TEMP), byrow=TRUE)
W    <- matrix(1, nrow=nrow(TEMP), ncol=ncol(TEMP))


library(parallel)

nc <- max(1L, parallel::detectCores() - 1L)

peakRAM({
  m_gam <- bam(
    outcome ~ te(TEMP, LAG, by=W, k=c(4, 4)) + day_of_week, # + stratum,
    family = poisson(),
    method = "fREML",
    data = tempDat,
    discrete = TRUE,
    nthreads = nc
    #cluster = cl
  )
})




k.check(m_gam)

###################################
# Compute cumulative IRR curve from your GAM
####################################
library(MASS)      # for mvrnorm
library(ggplot2)

cen <- 6.914142
temp_grid <- seq(min(tempDat$temp0, na.rm=TRUE),
                 max(tempDat$temp0, na.rm=TRUE),
                 by = 0.1)
lag_grid <- 0:max_lag

# prediction grid
grid <- as.data.table(expand.grid(TEMP=temp_grid, LAG=lag_grid))
grid[, W := 1]
# baseline grid at cen (same lags)
grid0 <- copy(grid)
grid0[, TEMP := cen]

grid[, day_of_week := levels(tempDat$day_of_week)[1]]
grid[, stratum := levels(tempDat$stratum)[1]]

grid0 <- copy(grid)
grid0[, TEMP := cen]

# linear predictor matrices (drop intercept)
Xp  <- predict(m_gam, newdata=grid,  type="lpmatrix")[, -1, drop=FALSE]
X0p <- predict(m_gam, newdata=grid0, type="lpmatrix")[, -1, drop=FALSE]

beta <- coef(m_gam)[-1]
grid[, logRR := as.numeric(Xp %*% beta - X0p %*% beta)]
grid[, RR := exp(logRR)]

# cumulative across lags
cum <- grid[, .(cumlogRR = sum(logRR)), by=TEMP]
cum[, IRR := exp(cumlogRR)]


############
# Add uncertainty (posterior simulation)
#############
nsim <- 100

V <- m_gam$Vp[-1, -1, drop=FALSE]  # drop intercept
b_sims <- MASS::mvrnorm(nsim, mu=beta, Sigma=V)

LP  <- Xp  %*% t(b_sims)
LP0 <- X0p %*% t(b_sims)
LPc <- LP - LP0                    # centered logRR sims at each grid point

# attach sims into grid (careful with memory; this is moderate for lag<=21 and temp grid ~500)
LPc_dt <- as.data.table(LPc)
LPc_dt[, TEMP := grid$TEMP]
LPc_dt[, LAG  := grid$LAG]

# sum across lags within TEMP for each sim column
sim_cols <- names(LPc_dt)[!(names(LPc_dt) %in% c("TEMP","LAG"))]
cum_sims <- LPc_dt[, lapply(.SD, sum), by=TEMP, .SDcols=sim_cols]

cum[, lower := apply(exp(as.matrix(cum_sims[, ..sim_cols])), 1, quantile, probs=0.05)]
cum[, upper := apply(exp(as.matrix(cum_sims[, ..sim_cols])), 1, quantile, probs=0.95)]

png("GAM_DLNMcumulativeIRR.png", width=800, height=600)
ggplot(cum, aes(x=TEMP, y=IRR)) +
  geom_ribbon(aes(ymin=lower, ymax=upper), alpha=0.2) +
  geom_line(linewidth=1) +
  geom_hline(yintercept=1, linetype=2) +
  ylim(c(0.5, 4)) +
  labs(x="Average Temperature (°C)", y="IRR",
       title="Overall cumulative effect (GAM-DLNM), centered at cen") +
  theme_bw()
dev.off()
