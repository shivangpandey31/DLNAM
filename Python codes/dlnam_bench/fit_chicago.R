#!/usr/bin/env Rscript
# fit_chicago.R -- real-data DLNM on Chicago NMMAPS under four approaches
# (QAIC, QBIC, penalised, TDLNM), each adjusting for the SAME covariates the DLNAM uses:
# smooth confounders (dew point, ozone, PM10), a seasonal/long-term time trend,
# and day-of-week. Only the temperature exposure-response is exported (cumulative
# + lag surface), centered at the reference, matching the DLNAM outputs.
#
# Usage:  Rscript fit_chicago.R <bench_dir> [methods]
#   bench_dir must contain config.json (written by run_real_chicago.py) and the
#   data csv it points to. methods: subset of {qaic,qbic,pen,tdlnm}; default all.
#
# Outputs (prefix: "" QAIC, "qbic_" QBIC, "pen_" penalised, "tree_" TDLNM):
#   <prefix>chicago_temp_cum.csv   value, fit, lo, hi   (cumulative RR vs temp)
#   <prefix>chicago_temp_surf.csv  value, lag, rr       (lag surface)
#
# Requires: dlnm, splines, mgcv, jsonlite; dlmtree for tdlnm.

suppressMessages({ library(dlnm); library(splines); library(mgcv); library(jsonlite) })
write_environment <- function(path, methods, tdlnm_settings) {
  pkgs <- c("dlnm", "splines", "mgcv", "jsonlite")
  if ("tdlnm" %in% methods) pkgs <- c(pkgs, "dlmtree")
  versions <- setNames(as.list(vapply(pkgs, function(p) {
    as.character(utils::packageVersion(p))
  }, character(1))), pkgs)
  write_json(list(
    r_version = R.version.string,
    platform = R.version$platform,
    methods = methods,
    tdlnm = tdlnm_settings,
    packages = versions
  ), path, auto_unbox = TRUE, pretty = TRUE)
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript fit_chicago.R <bench_dir> [methods]")
bench   <- args[1]
methods <- if (length(args) >= 2) trimws(strsplit(args[2], ",")[[1]]) else c("qaic","qbic","pen","tdlnm")
valid_methods <- c("qaic", "qbic", "pen", "tdlnm")
bad_methods <- setdiff(methods, valid_methods)
if (length(bad_methods)) stop(sprintf("unknown method(s): %s", paste(bad_methods, collapse = ",")))
if ("tdlnm" %in% methods && !requireNamespace("dlmtree", quietly = TRUE)) {
  stop("R package 'dlmtree' is required for method 'tdlnm'. Install with install.packages('dlmtree').")
}

cfg <- fromJSON(file.path(bench, "config.json"), simplifyVector = TRUE)
dat <- read.csv(file.path(bench, cfg$data))

x       <- dat[[cfg$exposure_col]]
y       <- dat[[cfg$target_col]]
grid    <- as.numeric(cfg$grid)
ref     <- as.numeric(cfg$reference)
lag_max <- as.integer(cfg$lag_max)
ci      <- as.numeric(cfg$ci_level)
vdf_g   <- as.integer(cfg$value_df_grid)
ldf_g   <- as.integer(cfg$lag_df_grid)
vdf_pen <- if (!is.null(cfg$penalized_value_df)) as.integer(unlist(cfg$penalized_value_df)) else max(vdf_g)
ldf_pen <- if (!is.null(cfg$penalized_lag_df)) as.integer(unlist(cfg$penalized_lag_df)) else max(ldf_g)

log_lag_ns <- function(df, lag = lag_max) {
  df <- as.integer(df)
  if (df < 2L) stop("natural-spline lag df must be at least 2")
  if (df == 2L) return(list(fun = "ns", df = df))
  list(
    fun = "ns",
    knots = logknots(lag, fun = "ns", df = df, intercept = TRUE)
  )
}

tdlnm_burn <- if (!is.null(cfg$tdlnm_burn)) as.integer(unlist(cfg$tdlnm_burn)) else 5000L
tdlnm_iter <- if (!is.null(cfg$tdlnm_iter)) as.integer(unlist(cfg$tdlnm_iter)) else 15000L
tdlnm_thin <- if (!is.null(cfg$tdlnm_thin)) as.integer(unlist(cfg$tdlnm_thin)) else 10L
tdlnm_attempts <- if (!is.null(cfg$tdlnm_attempts)) as.integer(unlist(cfg$tdlnm_attempts)) else 3L
tdlnm_exposure_splits <- if (!is.null(cfg$tdlnm_exposure_splits)) {
  as.integer(unlist(cfg$tdlnm_exposure_splits))
} else {
  30L
}
tdlnm_trees <- if (!is.null(cfg$tdlnm_trees)) {
  as.integer(unlist(cfg$tdlnm_trees))
} else {
  20L
}
tdlnm_seed <- if (!is.null(cfg$tdlnm_seed)) as.integer(unlist(cfg$tdlnm_seed)) else 0L
tdlnm_settings <- list(
  family = "gaussian_log1p",
  n_burn = tdlnm_burn,
  n_iter = tdlnm_iter,
  n_thin = tdlnm_thin,
  n_attempts = tdlnm_attempts,
  exposure_splits = tdlnm_exposure_splits,
  n_trees = tdlnm_trees,
  seed = tdlnm_seed
)
write_environment(file.path(bench, "r_environment.json"), methods, tdlnm_settings)
if ("tdlnm" %in% methods) {
  unlink(file.path(bench, "tdlnm_fit_status.json"), force = TRUE)
}

# --- adjustment terms (matching the original dlnm_chicago.R) ------------------
# Dew point enters as a natural spline; ozone and PM10 enter linearly; plus a
# seasonal/long-term ns(time) trend and day-of-week. cfg$confounder_spec maps
# each column name -> either an integer ns df, or 0 / "linear" for a linear term.
time <- dat[[cfg$time_col]]            # true calendar index (carried through NA drops)
conf_terms <- character(0)
spec <- cfg$confounder_spec
for (cn in names(spec)) {
  v <- spec[[cn]]
  if (is.numeric(v) && v > 0)
    conf_terms <- c(conf_terms, sprintf("ns(%s, df=%d)", cn, as.integer(v)))
  else
    conf_terms <- c(conf_terms, cn)            # linear term
}
adj <- paste(c(sprintf("ns(time, df=%d)", cfg$trend_df),
               conf_terms,
               sprintf("factor(%s)", cfg$dow_col)), collapse = " + ")

# model frame: holds y, time, and every confounder/dow column the formula names,
# so glm()/gam() resolve them via data=. (The cross-basis is passed separately by
# name -- dlnm's crosspred needs the object, not a data column.)
mf <- data.frame(y = y, time = time)
for (cn in c(names(spec), cfg$dow_col)) mf[[cn]] <- dat[[cn]]

prefix_of <- function(m) c(qaic = "", qbic = "qbic_", pen = "pen_", tdlnm = "tree_")[[m]]
pre <- function(prefix, f) file.path(bench, paste0(prefix, f))
method_paths <- function(method) {
  prefix <- prefix_of(method)
  c(pre(prefix, cfg$cum_name), pre(prefix, cfg$surf_name))
}
clear_method_outputs <- function(method) {
  unlink(method_paths(method), force = TRUE)
}
for (method in methods) clear_method_outputs(method)

write_out <- function(cp, prefix) {
  cum <- data.frame(value = grid, fit = as.numeric(cp$allRRfit),
                    lo = as.numeric(cp$allRRlow), hi = as.numeric(cp$allRRhigh))
  write.csv(cum, pre(prefix, cfg$cum_name), row.names = FALSE)
  mat <- cp$matRRfit
  surf <- data.frame(value = rep(grid, times = ncol(mat)),
                     lag = rep(0:(ncol(mat) - 1), each = nrow(mat)),
                     rr = as.numeric(mat))
  write.csv(surf, pre(prefix, cfg$surf_name), row.names = FALSE)
}

make_lag_matrix <- function(x, lag_max) {
  n <- length(x) - lag_max
  out <- matrix(NA_real_, nrow = n, ncol = lag_max + 1)
  for (lag in 0:lag_max) {
    out[, lag + 1] <- x[(lag_max + 1 - lag):(length(x) - lag)]
  }
  colnames(out) <- paste0("lag", 0:lag_max)
  out
}

record_tdlnm_status <- function(status, attempts, message = "") {
  write_json(list(
    method = "tdlnm",
    status = status,
    attempts = as.integer(attempts),
    message = message
  ), file.path(bench, "tdlnm_fit_status.json"), auto_unbox = TRUE, pretty = TRUE)
}

# Gasparrini et al. (2010) eq. (13): penalty uses k = vx*vl, the number of
# TEMPERATURE cross-basis parameters -- NOT the whole model's df (which is
# dominated by the ~98-df trend and would mask the exposure complexity).
ic_value <- function(model, kfac, k_cb) {
  ll  <- sum(dpois(model$y, model$fitted.values, log = TRUE))
  phi <- summary(model)$dispersion
  -2 * ll + kfac * phi * k_cb
}

# --- QAIC / QBIC: select temp cross-basis df, confounders held fixed ----------
fit_ic <- function(ic) {
  rng <- range(x, na.rm = TRUE)
  vk <- function(vdf) rng[1] + diff(rng) / vdf * seq_len(vdf - 1)
  kfac <- if (ic == "qbic") log(length(y)) else 2
  best <- list(q = Inf, vdf = NA, ldf = NA)
  for (vdf in vdf_g) for (ldf in ldf_g) {
    cb <- crossbasis(x, lag = lag_max,
                     argvar = list(fun = "ns", knots = vk(vdf)),
                     arglag = log_lag_ns(ldf))
    m <- glm(as.formula(paste("y ~ cb +", adj)), data = mf, family = quasipoisson(), na.action = na.omit)
    if (any(is.na(coef(m)))) next
    q <- tryCatch(ic_value(m, kfac, vdf * ldf), error = function(e) Inf)
    if (is.finite(q) && q < best$q) best <- list(q = q, vdf = vdf, ldf = ldf)
  }
  if (!is.finite(best$q)) stop(sprintf("no full-rank %s fit", ic))
  cb_temp <- crossbasis(x, lag = lag_max,
                        argvar = list(fun = "ns", knots = vk(best$vdf)),
                        arglag = log_lag_ns(best$ldf))
  m <- glm(as.formula(paste("y ~ cb_temp +", adj)), data = mf, family = quasipoisson(), na.action = na.omit)
  cp <- crosspred(cb_temp, m, at = grid, cen = ref, bylag = 1, ci.level = ci)
  write_out(cp, prefix_of(ic))
  cat(sprintf("  %-10s value df %-2d   lag df %-2d\n", toupper(ic), best$vdf, best$ldf))
}

# --- penalised: P-spline temp cross-basis (cbPen + REML), confounders fixed ---
fit_pen <- function() {
  cb_temp <- crossbasis(x, lag = lag_max,
                        argvar = list(fun = "ps", df = vdf_pen),
                        arglag = list(fun = "ps", df = ldf_pen))
  cbp <- cbPen(cb_temp)
  m <- gam(as.formula(paste("y ~ cb_temp +", adj)), data = mf, family = quasipoisson(),
           paraPen = list(cb_temp = cbp), method = "REML", na.action = na.omit)
  cp <- crosspred(cb_temp, m, at = grid, cen = ref, bylag = 1, ci.level = ci)
  write_out(cp, "pen_")
  cb_cols <- grep("^cb_temp", names(coef(m)))
  edf <- sum(m$edf[cb_cols])
  k <- length(cb_cols)
  cat(sprintf("  %-10s edf %.1f / %d (%.0f%%)   basis %d x %d\n",
              "Penalised", edf, k, 100 * edf / k, vdf_pen, ldf_pen))
}

summary_tdlnm <- NULL
if ("tdlnm" %in% methods) {
  summary_tdlnm <- getS3method("summary", "tdlnm", optional = TRUE)
  if (is.null(summary_tdlnm)) {
    summary_tdlnm <- getFromNamespace("summary.tdlnm", "dlmtree")
  }
}

fit_tdlnm_once <- function(attempt) {
  x_lag <- make_lag_matrix(x, lag_max)
  fit_data <- mf[(lag_max + 1):nrow(mf), , drop = FALSE]
  fit_data$y <- log1p(fit_data$y)

  ok <- complete.cases(x_lag) & complete.cases(fit_data)
  x_lag <- x_lag[ok, , drop = FALSE]
  fit_data <- fit_data[ok, , drop = FALSE]

  set.seed(tdlnm_seed + 100000L * (attempt - 1L))
  fit <- dlmtree::dlmtree(
    formula = as.formula(paste("y ~", adj)),
    data = fit_data,
    exposure.data = x_lag,
    dlm.type = "nonlinear",
    family = "gaussian",
    control.tdlnm = list(exposure.splits = tdlnm_exposure_splits),
    # n.trees passed as a double: dlmtree's guard on these four is inverted
    # (it aborts when all ARE valid positive integers) and its own default is a
    # double, so this reproduces the working code path. See fit_dlnm.R.
    control.mcmc = list(n.burn = tdlnm_burn, n.iter = tdlnm_iter,
                        n.thin = tdlnm_thin, n.trees = as.numeric(tdlnm_trees)),
    control.diagnose = list(verbose = FALSE)
  )

  fit$Xrange <- range(grid)
  sm <- summary_tdlnm(
    fit,
    conf.level = ci,
    pred.at = grid,
    cenval = ref,
    verbose = FALSE
  )
  if (!is.list(sm) || is.null(sm$cumulative.effect) || is.null(sm$plot.dat)) {
    stop("dlmtree summary did not return cumulative.effect and plot.dat outputs")
  }

  cum <- sm$cumulative.effect
  write.csv(data.frame(
    value = as.numeric(cum$vals),
    fit = exp(as.numeric(cum$mean)),
    lo = exp(as.numeric(cum$lower)),
    hi = exp(as.numeric(cum$upper))
  ), pre(prefix_of("tdlnm"), cfg$cum_name), row.names = FALSE)

  pd <- sm$plot.dat
  write.csv(data.frame(
    value = as.numeric(pd$PredVal),
    lag = as.numeric(pd$Tmin),
    rr = exp(as.numeric(pd$Est)),
    lo = exp(as.numeric(pd$CIMin)),
    hi = exp(as.numeric(pd$CIMax))
  ), pre(prefix_of("tdlnm"), cfg$surf_name), row.names = FALSE)

  cat(sprintf("  %-10s saved (attempt %d, gaussian log1p counts)\n", "TDLNM", attempt))
}

fit_tdlnm <- function() {
  last_message <- ""
  for (attempt in seq_len(tdlnm_attempts)) {
    err <- tryCatch({
      fit_tdlnm_once(attempt)
      NULL
    }, error = function(e) e)

    if (is.null(err)) {
      record_tdlnm_status("saved", attempt)
      return(invisible(TRUE))
    }

    last_message <- conditionMessage(err)
    unlink(c(pre(prefix_of("tdlnm"), cfg$cum_name),
             pre(prefix_of("tdlnm"), cfg$surf_name)), force = TRUE)
    cat(sprintf("  %-10s attempt %d failed: %s\n", "TDLNM", attempt, last_message))
  }

  record_tdlnm_status("failed", tdlnm_attempts, last_message)
  stop(sprintf("TDLNM failed after %d attempt(s): %s", tdlnm_attempts, last_message))
}

cat(sprintf("  adjustment %s\n", adj))
if ("qaic" %in% methods) fit_ic("qaic")
if ("qbic" %in% methods) fit_ic("qbic")
if ("pen"  %in% methods) fit_pen()
if ("tdlnm" %in% methods) fit_tdlnm()
