#!/usr/bin/env Rscript
# fit_malaria.R -- reference malaria DLNM fits.
#
# Usage: Rscript fit_malaria.R <bench_dir>
#
# The Python runner writes <bench_dir>/config.json and malaria_data.csv. This
# script fits one focal exposure-lag surface at a time. Other climate exposures
# enter as row means over the same lag window, matching the reference modelling
# convention used in the thesis-era analysis.

suppressMessages({
  library(dlnm)
  library(glmmTMB)
  library(jsonlite)
  library(splines)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript fit_malaria.R <bench_dir>")
bench <- args[1]
cfg <- fromJSON(file.path(bench, "config.json"), simplifyVector = TRUE)
dat <- read.csv(file.path(bench, cfg$data))

write_json(list(
  r_version = R.version.string,
  platform = R.version$platform,
  packages = list(
    dlnm = as.character(utils::packageVersion("dlnm")),
    glmmTMB = as.character(utils::packageVersion("glmmTMB")),
    jsonlite = as.character(utils::packageVersion("jsonlite")),
    splines = "base"
  )
), file.path(bench, "r_environment.json"), auto_unbox = TRUE, pretty = TRUE)

exposures <- as.character(cfg$exposures)
lag_count <- as.integer(cfg$lag_count)
ci <- as.numeric(cfg$ci_level)
value_df <- as.integer(cfg$value_df)
lag_df <- as.integer(cfg$lag_df)

log_lag_ns <- function(df, lag) {
  df <- as.integer(df)
  if (df < 2L) stop("natural-spline lag df must be at least 2")
  if (df == 2L) return(list(fun = "ns", df = df))
  list(
    fun = "ns",
    knots = logknots(lag, fun = "ns", df = df, intercept = TRUE)
  )
}

p <- function(exposure, suffix) {
  file.path(bench, paste0("ref_", exposure, "_", suffix))
}

lag_cols <- function(exposure) {
  paste0(exposure, "_lag", seq_len(lag_count))
}

control_for <- function(exposure) {
  spec <- cfg$control_spec[[exposure]]
  if (is.null(spec)) character(0) else as.character(spec)
}

write_outputs <- function(exposure, pred, grid) {
  cum <- data.frame(
    value = as.numeric(grid),
    fit = as.numeric(pred$allRRfit),
    lo = as.numeric(pred$allRRlow),
    hi = as.numeric(pred$allRRhigh)
  )
  write.csv(cum, p(exposure, "cum.csv"), row.names = FALSE)

  mat <- pred$matRRfit
  surf <- data.frame(
    value = rep(as.numeric(grid), times = ncol(mat)),
    lag = rep(seq_len(ncol(mat)), each = nrow(mat)),
    rr = as.numeric(mat)
  )
  write.csv(surf, p(exposure, "surf.csv"), row.names = FALSE)
}

fit_exposure <- function(exposure) {
  x_lag <- as.matrix(dat[, lag_cols(exposure), drop = FALSE])
  ref <- as.numeric(cfg$reference[[exposure]])
  grid <- as.numeric(cfg$grid[[exposure]])

  mf <- data.frame(
    y = as.integer(dat[[cfg$target_col]]),
    month = dat$month,
    year = dat$year,
    unique_cluster = factor(dat$unique_cluster),
    Country = factor(dat$Country)
  )

  controls <- control_for(exposure)
  control_terms <- character(0)
  for (ctrl in controls) {
    nm <- paste0("ctrl_", ctrl)
    mf[[nm]] <- rowMeans(dat[, lag_cols(ctrl), drop = FALSE], na.rm = TRUE)
    control_terms <- c(control_terms, nm)
  }

  cb <- crossbasis(
    x_lag,
    lag = c(1, lag_count),
    argvar = list(fun = "ns", df = value_df),
    arglag = log_lag_ns(lag_df, c(1, lag_count))
  )

  rhs <- c("cb", control_terms, "ns(month, df = 4)", "year",
           "(1 | unique_cluster)", "(1 | Country)")
  form <- as.formula(paste("y ~", paste(rhs, collapse = " + ")))

  fit <- glmmTMB(
    form,
    data = mf,
    family = binomial(link = "logit"),
    control = glmmTMBControl(optCtrl = list(iter.max = 1000, eval.max = 1000))
  )

  n_cb <- ncol(cb)
  beta <- summary(fit)$coef$cond[, 1]
  vc <- as.matrix(vcov(fit)$cond)
  cb_idx <- seq_len(n_cb) + 1L
  pred <- crosspred(
    cb,
    coef = beta[cb_idx],
    vcov = vc[cb_idx, cb_idx],
    cen = ref,
    model.link = "logit",
    at = grid,
    bylag = 1,
    cumul = TRUE,
    ci.level = ci
  )
  write_outputs(exposure, pred, grid)
  cat(sprintf("  %-10s DLNM saved (df %d x %d, controls: %s)\n",
              exposure, value_df, lag_df,
              ifelse(length(controls), paste(controls, collapse = ","), "none")))
}

cat("Malaria DLNM fits\n")
for (exposure in exposures) fit_exposure(exposure)
