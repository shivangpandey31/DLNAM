#!/usr/bin/env Rscript
# fit_joint.R -- DLNM with four concurrent distributed-lag exposures.
#
# Joint df selection over k exposures scales as grid^k (~infeasible at k=4), so
# QAIC/QBIC here use MARGINAL selection: select each exposure's cross-basis df
# while holding the others at a default, looping over exposures (linear cost).
# This is the standard practical compromise -- and the limitation the experiment
# illustrates. The penalised variant avoids selection entirely (cbPen on every
# cross-basis, one joint REML gam). Each exposure's cumulative + surface is
# crosspred separately, centered at the shared reference.
#
# Usage:  Rscript fit_joint.R <bench_dir> [methods]
# Requires: dlnm, splines, mgcv, jsonlite.  NOT validated in this environment.

suppressMessages({ library(dlnm); library(splines); library(mgcv); library(jsonlite) })

args  <- commandArgs(trailingOnly = TRUE)
bench <- args[1]
methods <- if (length(args) >= 2) trimws(strsplit(args[2], ",")[[1]]) else c("qaic","qbic","pen")

cfg <- fromJSON(file.path(bench, "manifest.json"), simplifyVector = TRUE)
dat <- read.csv(file.path(bench, cfg$data))

exps    <- cfg$exposures
y       <- dat[[cfg$target_col]]
grid    <- as.numeric(cfg$grid)
ref     <- as.numeric(cfg$reference)
lag_max <- as.integer(cfg$lag_max)
ci      <- as.numeric(cfg$ci_level)
vdf_g   <- as.integer(cfg$value_df_grid)
ldf_g   <- as.integer(cfg$lag_df_grid)
vdf_pen <- if (!is.null(cfg$penalized_value_df)) as.integer(unlist(cfg$penalized_value_df)) else max(vdf_g)
ldf_pen <- if (!is.null(cfg$penalized_lag_df)) as.integer(unlist(cfg$penalized_lag_df)) else max(ldf_g)
n       <- length(y)
DEF_V <- 4L; DEF_L <- 4L                 # default df for the "held" exposures

vk <- function(v, df) { r <- range(v, na.rm = TRUE)
  r[1] + diff(r) / df * seq_len(df - 1) }

log_lag_ns <- function(df, lag = lag_max) {
  df <- as.integer(df)
  if (df < 2L) stop("natural-spline lag df must be at least 2")
  if (df == 2L) return(list(fun = "ns", df = df))
  list(
    fun = "ns",
    knots = logknots(lag, fun = "ns", df = df, intercept = TRUE)
  )
}

# build an "ns" cross-basis for exposure e at given (vdf, ldf)
cb_ns <- function(e, vdf, ldf)
  crossbasis(dat[[e]], lag = lag_max,
             argvar = list(fun = "ns", knots = vk(dat[[e]], vdf)),
             arglag = log_lag_ns(ldf))

ic_value <- function(model, kfac, k_cb) {           # Gasparrini eq.(13), k=vx*vl
  ll <- sum(dpois(model$y, model$fitted.values, log = TRUE))
  -2 * ll + kfac * summary(model)$dispersion * k_cb
}

out_path <- function(prefix, relname) {              # relname like "out/x_sep_cum.csv"
  d <- dirname(relname); b <- basename(relname)
  file.path(bench, d, paste0(prefix, b))
}
prefix_of <- function(m) c(qaic = "", qbic = "qbic_", pen = "pen_")[[m]]

method_paths <- function(method, e) {
  prefix <- prefix_of(method)
  c(out_path(prefix, cfg$cum_name[[e]]), out_path(prefix, cfg$surf_name[[e]]))
}

clear_method_outputs <- function(method) {
  for (e in exps) unlink(method_paths(method, e), force = TRUE)
}
for (method in methods) clear_method_outputs(method)

write_cp <- function(cp, prefix, e) {
  cum <- data.frame(value = grid, fit = as.numeric(cp$allRRfit),
                    lo = as.numeric(cp$allRRlow), hi = as.numeric(cp$allRRhigh))
  write.csv(cum, out_path(prefix, cfg$cum_name[[e]]), row.names = FALSE)
  mat <- cp$matRRfit
  surf <- data.frame(value = rep(grid, times = ncol(mat)),
                     lag = rep(0:(ncol(mat) - 1), each = nrow(mat)),
                     rr = as.numeric(mat))
  write.csv(surf, out_path(prefix, cfg$surf_name[[e]]), row.names = FALSE)
}

# ---- QAIC / QBIC via MARGINAL selection -------------------------------------
fit_ic <- function(ic) {
  kfac <- if (ic == "qbic") log(n) else 2
  prefix <- if (ic == "qbic") "qbic_" else ""
  # start every exposure at the default df, then refine ONE exposure at a time
  sel <- setNames(rep(list(c(DEF_V, DEF_L)), length(exps)), exps)
  for (e in exps) {
    best <- list(q = Inf, vdf = DEF_V, ldf = DEF_L)
    for (vdf in vdf_g) for (ldf in ldf_g) {
      cbs <- lapply(exps, function(ee) {
        d <- if (ee == e) c(vdf, ldf) else sel[[ee]]
        cb_ns(ee, d[1], d[2])
      })
      names(cbs) <- paste0("cb_", exps)
      for (nm in names(cbs)) assign(nm, cbs[[nm]])
      form <- as.formula(paste("y ~", paste(names(cbs), collapse = " + ")))
      m <- tryCatch(glm(form, family = quasipoisson(), na.action = na.omit),
                    error = function(err) NULL)
      if (is.null(m) || any(is.na(coef(m)))) next
      q <- ic_value(m, kfac, vdf * ldf)
      if (is.finite(q) && q < best$q) best <- list(q = q, vdf = vdf, ldf = ldf)
    }
    sel[[e]] <- c(best$vdf, best$ldf)
    cat(sprintf("  %-5s %-7s selected vdf=%d ldf=%d\n", toupper(ic), e, best$vdf, best$ldf))
  }
  # final joint fit at the marginally-selected df, then crosspred each exposure
  cbs <- lapply(exps, function(ee) cb_ns(ee, sel[[ee]][1], sel[[ee]][2]))
  names(cbs) <- paste0("cb_", exps)
  for (nm in names(cbs)) assign(nm, cbs[[nm]])
  form <- as.formula(paste("y ~", paste(names(cbs), collapse = " + ")))
  m <- glm(form, family = quasipoisson(), na.action = na.omit)
  for (i in seq_along(exps)) {
    # crosspred matches coefficients by the DEPARSED name of the basis arg, so it
    # must see the literal "cb_<exp>" variable name (not cbs[[i]]).
    cp <- eval(parse(text = sprintf(
      "crosspred(%s, m, at = grid, cen = ref, bylag = 1, ci.level = ci)",
      names(cbs)[i])))
    write_cp(cp, prefix, exps[i])
  }
}

# ---- penalised: cbPen on every cross-basis, one joint REML gam ---------------
fit_pen <- function() {
  cbs <- lapply(exps, function(e)
    crossbasis(dat[[e]], lag = lag_max,
               argvar = list(fun = "ps", df = vdf_pen),
               arglag = list(fun = "ps", df = ldf_pen)))
  names(cbs) <- paste0("cb_", exps)
  pen <- list()
  for (i in seq_along(exps)) {
    assign(names(cbs)[i], cbs[[i]])
    pen[[names(cbs)[i]]] <- cbPen(cbs[[i]])
  }
  form <- as.formula(paste("y ~", paste(names(cbs), collapse = " + ")))
  m <- gam(form, family = quasipoisson(), paraPen = pen, method = "REML",
           na.action = na.omit)
  for (i in seq_along(exps)) {
    cp <- eval(parse(text = sprintf(
      "crosspred(%s, m, at = grid, cen = ref, bylag = 1, ci.level = ci)",
      names(cbs)[i])))
    write_cp(cp, "pen_", exps[i])
    cc <- grep(paste0("^cb_", exps[i]), names(coef(m)))
    cat(sprintf("  PEN %-7s edf=%.1f / k=%d (%.0f%%)\n", exps[i],
                sum(m$edf[cc]), length(cc), 100 * sum(m$edf[cc]) / length(cc)))
  }
  cat(sprintf("      penalised basis vdf=%d, ldf=%d\n", vdf_pen, ldf_pen))
}

cat(sprintf("Multi-exposure DLNM [%s], %d exposures, lag_max=%d, ref=%.1f\n",
            paste(methods, collapse = ", "), length(exps), lag_max, ref))
if ("qaic" %in% methods) fit_ic("qaic")
if ("qbic" %in% methods) fit_ic("qbic")
if ("pen"  %in% methods) fit_pen()
cat("done.\n")
