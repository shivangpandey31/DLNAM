#########################################################
################ DLNM with Interaction ##################
#########################################################

rm(list = ls())

####################### Loading libraries
library(dplyr)
library(gnm)
library(dlnm)
library(ggplot2)
library(lubridate)

df <- read.csv("case_time_series.csv")

head(df)

temp_knots <- c(-2.510472, 16.020317, 19.799403)  # 10%, 75%, 90%
temp_b_knots <- c(-8.547488, 25.588451)           # 1%, 99%
cen <- 6.914142  

df$stratum <- factor(paste(df$id, month(df$date), sep = "-"))

cbtmean <- crossbasis(df$tmean, lag = 21,
                      argvar = list(fun = "ns", 
                                    knots = temp_knots, 
                                    Boundary.knots = temp_b_knots),
                      arglag = list(fun = "ns", 
                                    knots = logknots(21, nk = 3)),
                      group = df$id)

# Model
day_of_week <- factor(wday(df$date))


#### INTERACTIONS

### TIME INVARIANT EFFECT MODIFIERS

# There are two ways to include effect modifers: 
# 1) the way I showed you here
# 2) another one that is easier to implement
# Both ways yield exactly the same results

# In this document I show you both ways for binary and categorical effect modifiers with more than two categories

# - BINARY EFFECT MODIFIERS

# 1) 1st way

# the reference category will be 0 (not smoking)
# the variable is already coded as dummy variable

# If you do cbtmean*smoking in the gnm() function, then it will include also the main effect of smoking
# Since smoking is not time-varying, the interaction terms need to be created before

cb_int <- cbtmean*df$smoking

mod <- gnm(outcome ~ cbtmean + cb_int + day_of_week,
           eliminate = df$stratum, data = df,
           family = poisson)

summary(mod)


# the parameters associated to cbtmean are the effect for non-smoking people
# so if you use crosspred() on cbtmean this will give you the curve for non-smokers 

cptmean_not_smok <- crosspred(cbtmean, mod, cen = cen, by = 1.5, cumul = TRUE)

# the effect for smokers can be found by summing the parameters of cbtmean + cb_int

# parameters cbtmean
b1 <- coef(mod)[1:ncol(cbtmean)]
b1

# parameters cb_int
b2 <- coef(mod)[21:(ncol(cbtmean)*2)]
b2

# take into account that the indeces of the parameters may be different depending
# on your cross-basis. So, please, adjust accordingly. The same is valid for variance-covariance matrices.

# taking the sum
sum_b1_b2 <- b1 + b2

# Finding variance-covariance matrix of the sum
V_b1 <- vcov(mod)[1:ncol(cbtmean), 1:ncol(cbtmean)]
V_b1

V_b2 <- vcov(mod)[21:(ncol(cbtmean)*2), 21:(ncol(cbtmean)*2)]
V_b2

V_b1_b2 <- vcov(mod)[1:ncol(cbtmean), 21:(ncol(cbtmean)*2)]
V_b1_b2

V_tot <- V_b1 + V_b2 + V_b1_b2 + t(V_b1_b2)

rownames(V_tot) <- c()
colnames(V_tot) <- c()

cptmean_smok <- crosspred(cbtmean, coef = sum_b1_b2, vcov = V_tot, cen = cen, model.link = "log", by = 1.5, cumul = TRUE)

plot(cptmean_not_smok, "overall", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", xlab = "Average Temperature (°C)", ci.arg = list(col = adjustcolor(1, alpha.f = 0.2)), ylim = c(0.5, 4))
lines(cptmean_smok, "overall", ci = "area", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", col = 2, ci.arg = list(col = adjustcolor(2, alpha.f = 0.2)))
legend("topleft", paste("Smoking =", c("No", "Yes")), col = 1:2, lwd = 1.5)

### Test for interaction
# comparing model with and without interaction term
mod0 <- gnm(outcome ~ cbtmean + day_of_week,
            eliminate = df$stratum, data = df,
            family = poisson)

anova(mod0, mod, test = "Chisq")
AIC(mod0)
AIC(mod)

# Interaction is significant in this case, so there is actually a statistically significant
# difference between smokers and not-smokers.

# 2) 2nd way

# This second way is actually easier to implement
# We can change the parametrisation of the binary variable, so that we can use the original
# crosspred function and we don't need to sum anything.
# So, what we need to do is to basically change the reference category each time

# effect for non-smokers
# the reference category is already non-smokers (coded as 0)

cb_int <- cbtmean*df$smoking

mod1 <- gnm(outcome ~ cbtmean + cb_int + day_of_week,
           eliminate = df$stratum, data = df,
           family = poisson)

cptmean_not_smok <- crosspred(cbtmean, mod1, cen = cen, by = 1.5, cumul = TRUE)

# to calculate effect for smokers, we can code smokers as the reference category

df$smoking2 <- (df$smoking == 0)*1

# now smokers are coded as 0 and not smokers as 1

cb_int <- cbtmean*df$smoking2

mod2 <- gnm(outcome ~ cbtmean + cb_int + day_of_week,
           eliminate = df$stratum, data = df,
           family = poisson)

cptmean_smok <- crosspred(cbtmean, mod2, cen = cen, by = 1.5, cumul = TRUE)

plot(cptmean_not_smok, "overall", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", xlab = "Average Temperature (°C)", ci.arg = list(col = adjustcolor(1, alpha.f = 0.2)), ylim = c(0.5, 4))
lines(cptmean_smok, "overall", ci = "area", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", col = 2, ci.arg = list(col = adjustcolor(2, alpha.f = 0.2)))
legend("topleft", paste("Smoking =", c("No", "Yes")), col = 1:2, lwd = 1.5)

# This is because the function crosspred will predict the effect of the reference category

anova(mod0, mod1, test = "Chisq")

anova(mod0, mod2, test = "Chisq")

# Results from the test will be the same. It is the same model, but different reference category.



# - CATEGORICAL EFFECT MODIFIERS WITH MORE THAN TWO CATEGORIES

# 1) 1st way

# Creating first a random categorical variable with four categories: "south", "centre", "north"
# You will not see any effect on this variable

rand <- sample(c("south", "centre", "north", "shivang"), size = length(unique(df$id)), replace = TRUE)

df$categ <- rep(rand, table(df$id))

head(df)

# let's assume that "south" is our reference category
# if a category is the reference, we don't create a dummy for it

dummy1 <- (df$categ == "centre")*1
dummy2 <- (df$categ == "north")*1
dummy3 <- (df$categ == "shivang")*1

cbint_dummy1 <- cbtmean*dummy1
cbint_dummy2 <- cbtmean*dummy2
cbint_dummy3 <- cbtmean*dummy3

mod <- gnm(outcome ~ cbtmean + cbint_dummy1 + cbint_dummy2 + cbint_dummy3 + day_of_week,
           eliminate = df$stratum, data = df,
           family = poisson)

# since "south" is our reference category we can use standard crosspred
cptmean_south <- crosspred(cbtmean, mod, cen = cen, by = 1.5, cumul = TRUE)


# the effect for centre can be found by summing the parameters of cbtmean + cbint_dummy1

# parameters cbtmean
b1 <- coef(mod)[1:ncol(cbtmean)]
b1

# parameters cbint_dummy1
b2 <- coef(mod)[21:(ncol(cbtmean)*2)]
b2

# take into account that the indeces of the parameters may be different depending
# on your cross-basis. So, please, adjust accordingly. The same is valid for variance-covariance matrices.

# taking the sum
sum_b1_b2 <- b1 + b2

# Finding variance-covariance matrix of the sum
V_b1 <- vcov(mod)[1:ncol(cbtmean), 1:ncol(cbtmean)]
V_b1

V_b2 <- vcov(mod)[21:(ncol(cbtmean)*2), 21:(ncol(cbtmean)*2)]
V_b2

V_b1_b2 <- vcov(mod)[1:ncol(cbtmean), 21:(ncol(cbtmean)*2)]
V_b1_b2

V_tot <- V_b1 + V_b2 + V_b1_b2 + t(V_b1_b2)

rownames(V_tot) <- c()
colnames(V_tot) <- c()

cptmean_centre <- crosspred(cbtmean, coef = sum_b1_b2, vcov = V_tot, cen = cen, model.link = "log", by = 1.5, cumul = TRUE)


# the effect for north can be found by summing the parameters of cbtmean + cbint_dummy2

# parameters cbtmean
b1 <- coef(mod)[1:ncol(cbtmean)]
b1

# parameters cbint_dummy2
b2 <- coef(mod)[41:(ncol(cbtmean)*3)]
b2

# take into account that the indeces of the parameters may be different depending
# on your cross-basis. So, please, adjust accordingly. The same is valid for variance-covariance matrices.

# taking the sum
sum_b1_b2 <- b1 + b2

# Finding variance-covariance matrix of the sum
V_b1 <- vcov(mod)[1:ncol(cbtmean), 1:ncol(cbtmean)]
V_b1

V_b2 <- vcov(mod)[41:(ncol(cbtmean)*3), 41:(ncol(cbtmean)*3)]
V_b2

V_b1_b2 <- vcov(mod)[1:ncol(cbtmean), 41:(ncol(cbtmean)*3)]
V_b1_b2

V_tot <- V_b1 + V_b2 + V_b1_b2 + t(V_b1_b2)

rownames(V_tot) <- c()
colnames(V_tot) <- c()

cptmean_north <- crosspred(cbtmean, coef = sum_b1_b2, vcov = V_tot, cen = cen, model.link = "log", by = 1.5, cumul = TRUE)


# the effect for shivang can be found by summing the parameters of cbtmean + cbint_dummy3

# parameters cbtmean
b1 <- coef(mod)[1:ncol(cbtmean)]
b1

# parameters cbint_dummy2
b2 <- coef(mod)[61:(ncol(cbtmean)*4)]
b2

# take into account that the indeces of the parameters may be different depending
# on your cross-basis. So, please, adjust accordingly. The same is valid for variance-covariance matrices.

# taking the sum
sum_b1_b2 <- b1 + b2

# Finding variance-covariance matrix of the sum
V_b1 <- vcov(mod)[1:ncol(cbtmean), 1:ncol(cbtmean)]
V_b1

V_b2 <- vcov(mod)[61:(ncol(cbtmean)*4), 61:(ncol(cbtmean)*4)]
V_b2

V_b1_b2 <- vcov(mod)[1:ncol(cbtmean), 61:(ncol(cbtmean)*4)]
V_b1_b2

V_tot <- V_b1 + V_b2 + V_b1_b2 + t(V_b1_b2)

rownames(V_tot) <- c()
colnames(V_tot) <- c()

cptmean_shivang <- crosspred(cbtmean, coef = sum_b1_b2, vcov = V_tot, cen = cen, model.link = "log", by = 1.5, cumul = TRUE)


plot(cptmean_south, "overall", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", xlab = "Average Temperature (°C)", ci.arg = list(col = adjustcolor(1, alpha.f = 0.2)), ylim = c(0.5, 4))
lines(cptmean_centre, "overall", ci = "area", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", col = 2, ci.arg = list(col = adjustcolor(2, alpha.f = 0.2)))
lines(cptmean_north, "overall", ci = "area", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", col = 3, ci.arg = list(col = adjustcolor(3, alpha.f = 0.2)))
lines(cptmean_shivang, "overall", ci = "area", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", col = 4, ci.arg = list(col = adjustcolor(4, alpha.f = 0.2)))
legend("topleft", paste("Region =", c("South", "Centre", "North", "Shivang")), col = 1:4, lwd = 1.5)

# Interaction test
anova(mod0, mod, test = "Chisq")

# Test not significant, as expected as variable is created randomly



# 2) 2nd way

# The second way is kind of the same as the previous approach, just changing the reference category every time

# let's assume that "south" is our reference category

dummy1 <- (df$categ == "centre")*1
dummy2 <- (df$categ == "north")*1
dummy3 <- (df$categ == "shivang")*1

cbint_dummy1 <- cbtmean*dummy1
cbint_dummy2 <- cbtmean*dummy2
cbint_dummy3 <- cbtmean*dummy3

mod <- gnm(outcome ~ cbtmean + cbint_dummy1 + cbint_dummy2 + cbint_dummy3 + day_of_week,
           eliminate = df$stratum, data = df,
           family = poisson)

# since "south" is our reference category we can use standard crosspred
cptmean_south <- crosspred(cbtmean, mod, cen = cen, by = 1.5, cumul = TRUE)

# let's change the reference to centre
dummy1 <- (df$categ == "south")*1
dummy2 <- (df$categ == "north")*1
dummy3 <- (df$categ == "shivang")*1

cbint_dummy1 <- cbtmean*dummy1
cbint_dummy2 <- cbtmean*dummy2
cbint_dummy3 <- cbtmean*dummy3

mod <- gnm(outcome ~ cbtmean + cbint_dummy1 + cbint_dummy2 + cbint_dummy3 + day_of_week,
           eliminate = df$stratum, data = df,
           family = poisson)

# since "centre" is our reference category we can use standard crosspred
cptmean_centre <- crosspred(cbtmean, mod, cen = cen, by = 1.5, cumul = TRUE)

# let's change the reference to north
dummy1 <- (df$categ == "south")*1
dummy2 <- (df$categ == "centre")*1
dummy3 <- (df$categ == "shivang")*1

cbint_dummy1 <- cbtmean*dummy1
cbint_dummy2 <- cbtmean*dummy2
cbint_dummy3 <- cbtmean*dummy3

mod <- gnm(outcome ~ cbtmean + cbint_dummy1 + cbint_dummy2 + cbint_dummy3 + day_of_week,
           eliminate = df$stratum, data = df,
           family = poisson)

# since "north" is our reference category we can use standard crosspred
cptmean_north <- crosspred(cbtmean, mod, cen = cen, by = 1.5, cumul = TRUE)

# let's change the reference to shivang
dummy1 <- (df$categ == "south")*1
dummy2 <- (df$categ == "centre")*1
dummy3 <- (df$categ == "north")*1

cbint_dummy1 <- cbtmean*dummy1
cbint_dummy2 <- cbtmean*dummy2
cbint_dummy3 <- cbtmean*dummy3

mod <- gnm(outcome ~ cbtmean + cbint_dummy1 + cbint_dummy2 + cbint_dummy3 + day_of_week,
           eliminate = df$stratum, data = df,
           family = poisson)

# since "shivang" is our reference category we can use standard crosspred
cptmean_shivang <- crosspred(cbtmean, mod, cen = cen, by = 1.5, cumul = TRUE)



plot(cptmean_south, "overall", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", xlab = "Average Temperature (°C)", ci.arg = list(col = adjustcolor(1, alpha.f = 0.2)), ylim = c(0.5, 4))
lines(cptmean_centre, "overall", ci = "area", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", col = 2, ci.arg = list(col = adjustcolor(2, alpha.f = 0.2)))
lines(cptmean_north, "overall", ci = "area", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", col = 3, ci.arg = list(col = adjustcolor(3, alpha.f = 0.2)))
lines(cptmean_shivang, "overall", ci = "area", lwd = 1.5, main = "Overall effect of average temperature", ylab = "IRR", col = 4, ci.arg = list(col = adjustcolor(4, alpha.f = 0.2)))
legend("topleft", paste("Region =", c("South", "Centre", "North", "Shivang")), col = 1:4, lwd = 1.5)

# The interaction test will be the same as before, not reported here.

# Both approaches can be extended to more than four categories following the same logic.

# The advantage of the first approach is that you need to fit just one model, while it is more tedious and less obvious to implement.

# The second approach requires fitting a model for each category, but it is conceptually easier.










