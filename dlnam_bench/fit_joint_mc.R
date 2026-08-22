#!/usr/bin/env Rscript
script_start <- proc.time()[["elapsed"]]
# fit_joint_mc.R -- joint DLNM fits for the joint MC manifest.
#
# This is the MC version of fit_joint.R. Each replicate contains all
# exposures, and each method writes one cumulative RR curve per exposure. QAIC
# and QBIC use marginal/coordinate-wise df selection because full joint df
# search scales as grid^K. Penalised uses one joint REML gam with cbPen on each
# cross-basis. TDLNM is fitted target-exposure by target-exposure, adjusted for
# the other concurrent exposures with fixed natural-spline cross-basis terms.

suppressMessages({
  library(dlnm)
  library(splines)
  library(mgcv)
  library(jsonlite)
})
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
if (length(args) < 1) stop("usage: Rscript fit_joint_mc.R <bench_dir> [methods]")
bench <- args[1]
methods <- if (length(args) >= 2) trimws(strsplit(args[2], ",")[[1]]) else c("qaic", "qbic", "pen", "tdlnm")
valid_methods <- c("qaic", "qbic", "pen", "tdlnm")
unknown <- setdiff(methods, valid_methods)
if (length(unknown)) stop(sprintf("unknown method(s): %s", paste(unknown, collapse = ", ")))
if ("tdlnm" %in% methods && !requireNamespace("dlmtree", quietly = TRUE)) {
  stop("R package 'dlmtree' is required for method 'tdlnm'. Install with install.packages('dlmtree').")
}

cfg <- fromJSON(file.path(bench, "manifest.json"), simplifyVector = FALSE)
exps <- unlist(cfg$exposures)
grid <- as.numeric(unlist(cfg$grid))
ref <- as.numeric(cfg$reference)
lag_max <- as.integer(cfg$lag_max)
ci <- as.numeric(cfg$ci_level)
target <- cfg$target_col
base_seed <- if (!is.null(cfg$base_seed)) as.integer(unlist(cfg$base_seed)) else 0L
vdf_g <- as.integer(unlist(cfg$value_df_grid))
ldf_g <- as.integer(unlist(cfg$lag_df_grid))
vdf_pen <- if (!is.null(cfg$penalized_value_df)) {
  as.integer(unlist(cfg$penalized_value_df))
} else {
  max(vdf_g)
}
ldf_pen <- if (!is.null(cfg$penalized_lag_df)) {
  as.integer(unlist(cfg$penalized_lag_df))
} else {
  max(ldf_g)
}

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
tdlnm_adjust_vdf <- if (!is.null(cfg$tdlnm_adjust_value_df)) {
  as.integer(unlist(cfg$tdlnm_adjust_value_df))
} else {
  4L
}
tdlnm_adjust_ldf <- if (!is.null(cfg$tdlnm_adjust_lag_df)) {
  as.integer(unlist(cfg$tdlnm_adjust_lag_df))
} else {
  4L
}
tdlnm_settings <- list(
  family = "gaussian",
  response = "log1p",
  n_burn = tdlnm_burn,
  n_iter = tdlnm_iter,
  n_thin = tdlnm_thin,
  n_attempts = tdlnm_attempts,
  exposure_splits = tdlnm_exposure_splits,
  n_trees = tdlnm_trees,
  adjustment_value_df = tdlnm_adjust_vdf,
  adjustment_lag_df = tdlnm_adjust_ldf
)
write_environment(file.path(bench, "r_environment.json"), methods, tdlnm_settings)
if ("tdlnm" %in% methods) {
  unlink(file.path(bench, "tdlnm_fit_status.json"), force = TRUE)
}
unlink(file.path(bench, "timing.json"), force = TRUE)

timing_records <- list()
record_timing <- function(method, rec, fit_seconds, exposure = NULL, extra = list()) {
  entry <- list(
    rep = as.integer(rec$rep),
    method = method,
    fit_seconds = as.numeric(fit_seconds)
  )
  if (!is.null(exposure)) entry$exposure <- exposure
  timing_records[[length(timing_records) + 1L]] <<- c(entry, extra)
}
write_timing <- function() {
  write_json(list(
    kind = "joint_dlnm_mc",
    elapsed_seconds_total = as.numeric(proc.time()[["elapsed"]] - script_start),
    records = timing_records
  ), file.path(bench, "timing.json"), auto_unbox = TRUE, pretty = TRUE)
}

DEF_V <- 4L
DEF_L <- 4L

vk <- function(x, df) {
  r <- range(x, na.rm = TRUE)
  r[1] + diff(r) / df * seq_len(df - 1)
}

ic_value <- function(model, kfac, k_cb) {
  ll <- sum(dpois(model$y, model$fitted.values, log = TRUE))
  -2 * ll + kfac * summary(model)$dispersion * k_cb
}

out_path <- function(prefix, relname) {
  d <- dirname(relname)
  b <- basename(relname)
  file.path(bench, d, paste0(prefix, b))
}
prefix_of <- function(m) c(qaic = "qaic_", qbic = "qbic_", pen = "pen_", tdlnm = "tree_")[[m]]

method_paths <- function(rec, method, e) {
  prefix <- prefix_of(method)
  c(
    out_path(prefix, rec$cumulative[[e]]),
    out_path(prefix, rec$surface[[e]])
  )
}

clear_method_outputs <- function(rec, method) {
  for (e in exps) unlink(method_paths(rec, method, e), force = TRUE)
}

write_cp <- function(cp, rec, prefix, e) {
  cum <- data.frame(
    value = grid,
    fit = as.numeric(cp$allRRfit),
    lo = as.numeric(cp$allRRlow),
    hi = as.numeric(cp$allRRhigh)
  )
  write.csv(cum, out_path(prefix, rec$cumulative[[e]]), row.names = FALSE)

  mat <- cp$matRRfit
  surf <- data.frame(
    value = rep(grid, times = ncol(mat)),
    lag = rep(0:(ncol(mat) - 1), each = nrow(mat)),
    rr = as.numeric(mat)
  )
  write.csv(surf, out_path(prefix, rec$surface[[e]]), row.names = FALSE)
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

tdlnm_paths <- function(rec, e) {
  list(
    cumulative = method_paths(rec, "tdlnm", e)[[1]],
    surface = method_paths(rec, "tdlnm", e)[[2]]
  )
}

record_tdlnm_status <- function(rec, e, status, attempts, message = "") {
  list(
    rep = as.integer(rec$rep),
    exposure = e,
    status = status,
    attempts = as.integer(attempts),
    message = message
  )
}

cb_ns <- function(dat, e, vdf, ldf) {
  crossbasis(
    dat[[e]],
    lag = lag_max,
    argvar = list(fun = "ns", knots = vk(dat[[e]], vdf)),
    arglag = log_lag_ns(ldf)
  )
}

summary_tdlnm <- NULL
if ("tdlnm" %in% methods) {
  summary_tdlnm <- getS3method("summary", "tdlnm", optional = TRUE)
  if (is.null(summary_tdlnm)) {
    summary_tdlnm <- getFromNamespace("summary.tdlnm", "dlmtree")
  }
}

fit_ic <- function(dat, rec, ic) {
  fit_start <- proc.time()[["elapsed"]]
  n_grid_fits <- 0L
  y <- dat[[target]]
  kfac <- if (ic == "qbic") log(length(y)) else 2
  prefix <- paste0(ic, "_")

  sel <- setNames(vector("list", length(exps)), exps)
  for (e in exps) sel[[e]] <- c(DEF_V, DEF_L)

  for (e in exps) {
    best <- list(q = Inf, vdf = DEF_V, ldf = DEF_L)
    for (vdf in vdf_g) for (ldf in ldf_g) {
      n_grid_fits <- n_grid_fits + 1L
      cbs <- lapply(exps, function(ee) {
        d <- if (ee == e) c(vdf, ldf) else sel[[ee]]
        cb_ns(dat, ee, d[1], d[2])
      })
      names(cbs) <- paste0("cb_", exps)
      for (nm in names(cbs)) assign(nm, cbs[[nm]])
      form <- as.formula(paste("y ~", paste(names(cbs), collapse = " + ")))
      m <- tryCatch(
        glm(form, family = quasipoisson(), na.action = na.omit),
        error = function(err) NULL
      )
      if (is.null(m) || any(is.na(coef(m)))) next
      q <- tryCatch(ic_value(m, kfac, vdf * ldf), error = function(err) Inf)
      if (is.finite(q) && q < best$q) best <- list(q = q, vdf = vdf, ldf = ldf)
    }
    if (!is.finite(best$q)) {
      stop(sprintf(
        "no full-rank %s fit for rep %d exposure %s",
        ic, as.integer(rec$rep), e
      ))
    }
    sel[[e]] <- c(best$vdf, best$ldf)
    cat(sprintf(
      "  rep %3d %-4s %-18s selected vdf=%d ldf=%d\n",
      as.integer(rec$rep), toupper(ic), e, best$vdf, best$ldf
    ))
  }

  cbs <- lapply(exps, function(e) cb_ns(dat, e, sel[[e]][1], sel[[e]][2]))
  names(cbs) <- paste0("cb_", exps)
  for (nm in names(cbs)) assign(nm, cbs[[nm]])
  form <- as.formula(paste("y ~", paste(names(cbs), collapse = " + ")))
  m <- glm(form, family = quasipoisson(), na.action = na.omit)

  for (i in seq_along(exps)) {
    cp <- eval(parse(text = sprintf(
      "crosspred(%s, m, at = grid, cen = ref, bylag = 1, ci.level = ci)",
      names(cbs)[i]
    )))
    write_cp(cp, rec, prefix, exps[i])
  }
  selected <- lapply(exps, function(e) list(
    exposure = e,
    value_df = as.integer(sel[[e]][1]),
    lag_df = as.integer(sel[[e]][2])
  ))
  record_timing(ic, rec, proc.time()[["elapsed"]] - fit_start, extra = list(
    n_exposures = as.integer(length(exps)),
    grid_fits = as.integer(n_grid_fits),
    coordinatewise_selection = TRUE,
    selected = selected
  ))
}

fit_pen <- function(dat, rec) {
  fit_start <- proc.time()[["elapsed"]]
  y <- dat[[target]]

  cbs <- lapply(exps, function(e) {
    crossbasis(
      dat[[e]],
      lag = lag_max,
      argvar = list(fun = "ps", df = vdf_pen),
      arglag = list(fun = "ps", df = ldf_pen)
    )
  })
  names(cbs) <- paste0("cb_", exps)
  pen <- list()
  for (i in seq_along(exps)) {
    assign(names(cbs)[i], cbs[[i]])
    pen[[names(cbs)[i]]] <- cbPen(cbs[[i]])
  }

  form <- as.formula(paste("y ~", paste(names(cbs), collapse = " + ")))
  m <- gam(
    form,
    family = quasipoisson(),
    paraPen = pen,
    method = "REML",
    na.action = na.omit
  )

  for (i in seq_along(exps)) {
    cp <- eval(parse(text = sprintf(
      "crosspred(%s, m, at = grid, cen = ref, bylag = 1, ci.level = ci)",
      names(cbs)[i]
    )))
    write_cp(cp, rec, "pen_", exps[i])
    cc <- grep(paste0("^", names(cbs)[i]), names(coef(m)))
    cat(sprintf(
      "  rep %3d PEN  %-18s edf=%.1f / k=%d (%.0f%%)\n",
      as.integer(rec$rep), exps[i], sum(m$edf[cc]), length(cc),
      100 * sum(m$edf[cc]) / length(cc)
    ))
  }
  cat(sprintf("      penalised basis vdf=%d, ldf=%d\n", vdf_pen, ldf_pen))
  edf_by_exposure <- lapply(seq_along(exps), function(i) {
    cc <- grep(paste0("^", names(cbs)[i]), names(coef(m)))
    list(
      exposure = exps[i],
      edf = as.numeric(sum(m$edf[cc])),
      basis_dim = as.integer(length(cc))
    )
  })
  record_timing("pen", rec, proc.time()[["elapsed"]] - fit_start, extra = list(
    n_exposures = as.integer(length(exps)),
    value_df = as.integer(vdf_pen),
    lag_df = as.integer(ldf_pen),
    edf_by_exposure = edf_by_exposure
  ))
}

fit_tdlnm_once <- function(dat, rec, target_exp, attempt) {
  paths <- tdlnm_paths(rec, target_exp)
  x_lag <- make_lag_matrix(dat[[target_exp]], lag_max)
  y <- dat[[target]][(lag_max + 1):nrow(dat)]

  nuisance <- list()
  for (e in setdiff(exps, target_exp)) {
    cb <- cb_ns(dat, e, tdlnm_adjust_vdf, tdlnm_adjust_ldf)
    mat <- as.matrix(cb)[(lag_max + 1):nrow(dat), , drop = FALSE]
    colnames(mat) <- make.names(paste0("adj_", e, "_", seq_len(ncol(mat))))
    nuisance[[e]] <- mat
  }

  ok <- complete.cases(x_lag) & is.finite(y)
  for (mat in nuisance) ok <- ok & complete.cases(mat)
  x_lag <- x_lag[ok, , drop = FALSE]
  y <- y[ok]

  fit_data <- data.frame(
    y = log1p(y),
    trend = seq(-0.5, 0.5, length.out = length(y))
  )
  adjustment_terms <- character(0)
  for (mat in nuisance) {
    mat <- mat[ok, , drop = FALSE]
    fit_data <- cbind(fit_data, as.data.frame(mat, check.names = FALSE))
    adjustment_terms <- c(adjustment_terms, colnames(mat))
  }
  rhs <- paste(c("trend", adjustment_terms), collapse = " + ")

  seed <- base_seed + as.integer(rec$rep) + sum(utf8ToInt(as.character(target_exp)))
  if (attempt > 1L) seed <- seed + 100000L * (attempt - 1L)
  set.seed(seed)

  fit <- dlmtree::dlmtree(
    formula = as.formula(paste("y ~", rhs)),
    data = fit_data,
    exposure.data = x_lag,
    dlm.type = "nonlinear",
    family = "gaussian",
    control.tdlnm = list(exposure.splits = tdlnm_exposure_splits),
    # n.trees as a double: dlmtree's guard on these four is inverted and its
    # own default is a double, so this reproduces the working path (see fit_dlnm.R).
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
  ), paths$cumulative, row.names = FALSE)

  pd <- sm$plot.dat
  write.csv(data.frame(
    value = as.numeric(pd$PredVal),
    lag = as.numeric(pd$Tmin),
    rr = exp(as.numeric(pd$Est)),
    lo = exp(as.numeric(pd$CIMin)),
    hi = exp(as.numeric(pd$CIMax))
  ), paths$surface, row.names = FALSE)

  cat(sprintf(
    "  rep %3d TDLNM %-18s saved (attempt %d; adjusted for %d exposure(s))\n",
    as.integer(rec$rep), target_exp, attempt, length(setdiff(exps, target_exp))
  ))
}

fit_tdlnm_target <- function(dat, rec, target_exp) {
  paths <- tdlnm_paths(rec, target_exp)
  last_message <- ""
  fit_start <- proc.time()[["elapsed"]]
  for (attempt in seq_len(tdlnm_attempts)) {
    err <- tryCatch({
      fit_tdlnm_once(dat, rec, target_exp, attempt)
      NULL
    }, error = function(e) e)

    if (is.null(err)) {
      elapsed <- proc.time()[["elapsed"]] - fit_start
      record_timing("tdlnm", rec, elapsed, exposure = target_exp, extra = list(
        attempts = as.integer(attempt),
        status = "saved",
        adjusted_exposures = as.integer(length(setdiff(exps, target_exp)))
      ))
      status <- record_tdlnm_status(rec, target_exp, "saved", attempt)
      status$fit_seconds <- as.numeric(elapsed)
      return(status)
    }

    last_message <- conditionMessage(err)
    unlink(unlist(paths, use.names = FALSE), force = TRUE)
    cat(sprintf(
      "  rep %3d TDLNM %-18s attempt %d failed: %s\n",
      as.integer(rec$rep), target_exp, attempt, last_message
    ))
  }

  cat(sprintf(
    "  rep %3d TDLNM %-18s failed after %d attempts\n",
    as.integer(rec$rep), target_exp, tdlnm_attempts
  ))
  elapsed <- proc.time()[["elapsed"]] - fit_start
  record_timing("tdlnm", rec, elapsed, exposure = target_exp, extra = list(
    attempts = as.integer(tdlnm_attempts),
    status = "failed",
    adjusted_exposures = as.integer(length(setdiff(exps, target_exp)))
  ))
  status <- record_tdlnm_status(rec, target_exp, "failed", tdlnm_attempts, last_message)
  status$fit_seconds <- as.numeric(elapsed)
  status
}

fit_tdlnm <- function(dat, rec) {
  lapply(exps, function(e) fit_tdlnm_target(dat, rec, e))
}

fit_one <- function(rec) {
  for (method in methods) clear_method_outputs(rec, method)
  dat <- read.csv(file.path(bench, rec$data))
  if ("qaic" %in% methods) fit_ic(dat, rec, "qaic")
  if ("qbic" %in% methods) fit_ic(dat, rec, "qbic")
  if ("pen" %in% methods) fit_pen(dat, rec)
  if ("tdlnm" %in% methods) return(fit_tdlnm(dat, rec))
  NULL
}

cat(sprintf(
  "Multi-exposure MC DLNM [%s], %d reps, %d exposures, lag_max=%d, ref=%.1f\n",
  paste(methods, collapse = ", "), length(cfg$datasets), length(exps), lag_max, ref
))
statuses <- unlist(lapply(cfg$datasets, fit_one), recursive = FALSE)
tdlnm_statuses <- Filter(Negate(is.null), statuses)
if (length(tdlnm_statuses)) {
  write_json(tdlnm_statuses, file.path(bench, "tdlnm_fit_status.json"),
             auto_unbox = TRUE, pretty = TRUE)
  failed <- vapply(tdlnm_statuses, function(x) identical(x$status, "failed"), logical(1))
  if (any(failed)) {
    cat(sprintf(
      "warning: %d TDLNM target-exposure fit(s) failed; missing outputs will be skipped by Python scoring.\n",
      sum(failed)
    ))
  }
}
write_timing()
cat("done.\n")
