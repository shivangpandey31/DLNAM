#!/usr/bin/env Rscript
script_start <- proc.time()[["elapsed"]]
# fit_dlnm.R -- fit DLNM-family comparators for every manifest dataset:
#   qaic  classical cross-basis (ns), value/lag df chosen by QAIC  -> qaic_
#   qbic  same, df chosen by QBIC (log(n) penalty, parsimonious)   -> qbic_
#   pen   penalised P-spline DLNM (mgcv gam + cbPen + REML)         -> pen_
#   tdlnm treed DLNM via dlmtree, Gaussian on log1p(count)           -> tree_
# QAIC and QBIC SHARE one df grid search (they differ only in the penalty applied
# to the same fitted models), so the grid is fit once. Each approach writes the
# cumulative exposure-response + lag surface CSVs on the manifest grid, centered
# at the reference.
#
# Usage:  Rscript fit_dlnm.R <out_dir> [methods]
#   methods: comma-separated subset of {qaic,qbic,pen,tdlnm}; default all.
#
# Requires R packages: dlnm, splines, mgcv, jsonlite; dlmtree for tdlnm.

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
if (length(args) < 1) stop("usage: Rscript fit_dlnm.R <out_dir> [methods]")
out_dir <- args[1]
methods <- if (length(args) >= 2) trimws(strsplit(args[2], ",")[[1]]) else c("qaic", "qbic", "pen", "tdlnm")
valid_methods <- c("qaic", "qbic", "pen", "tdlnm")
unknown <- setdiff(methods, valid_methods)
if (length(unknown)) stop(sprintf("unknown method(s): %s", paste(unknown, collapse = ", ")))
if ("tdlnm" %in% methods && !requireNamespace("dlmtree", quietly = TRUE)) {
  stop("R package 'dlmtree' is required for method 'tdlnm'. Install with install.packages('dlmtree').")
}

manifest <- fromJSON(file.path(out_dir, "manifest.json"), simplifyVector = FALSE)
grid       <- as.numeric(unlist(manifest$grid))
reference  <- as.numeric(manifest$reference)
lag_max    <- as.integer(manifest$lag_max)
ci_level   <- as.numeric(manifest$ci_level)
exposure   <- manifest$exposure_col
target     <- manifest$target_col
base_seed  <- if (!is.null(manifest$base_seed)) as.integer(manifest$base_seed) else 0L
vdf_grid   <- as.integer(unlist(manifest$value_df_grid))
ldf_grid   <- as.integer(unlist(manifest$lag_df_grid))

log_lag_ns <- function(df, lag = lag_max) {
  df <- as.integer(df)
  if (df < 2L) stop("natural-spline lag df must be at least 2")
  if (df == 2L) return(list(fun = "ns", df = df))
  list(
    fun = "ns",
    knots = logknots(lag, fun = "ns", df = df, intercept = TRUE)
  )
}

tdlnm_burn <- if (!is.null(manifest$tdlnm_burn)) as.integer(unlist(manifest$tdlnm_burn)) else 5000L
tdlnm_iter <- if (!is.null(manifest$tdlnm_iter)) as.integer(unlist(manifest$tdlnm_iter)) else 15000L
tdlnm_thin <- if (!is.null(manifest$tdlnm_thin)) as.integer(unlist(manifest$tdlnm_thin)) else 10L
tdlnm_attempts <- if (!is.null(manifest$tdlnm_attempts)) as.integer(unlist(manifest$tdlnm_attempts)) else 3L
tdlnm_exposure_splits <- if (!is.null(manifest$tdlnm_exposure_splits)) {
  as.integer(unlist(manifest$tdlnm_exposure_splits))
} else {
  30L
}
# A = 20 trees, Mork and Wilson (2022) section 4.3.
tdlnm_trees <- if (!is.null(manifest$tdlnm_trees)) {
  as.integer(unlist(manifest$tdlnm_trees))
} else {
  20L
}
tdlnm_settings <- list(
  family = "gaussian",
  response = "log1p",
  n_burn = tdlnm_burn,
  n_iter = tdlnm_iter,
  n_thin = tdlnm_thin,
  n_attempts = tdlnm_attempts,
  exposure_splits = tdlnm_exposure_splits,
  n_trees = tdlnm_trees
)
write_environment(file.path(out_dir, "r_environment.json"), methods, tdlnm_settings)
if ("tdlnm" %in% methods) {
  unlink(file.path(out_dir, "tdlnm_fit_status.json"), force = TRUE)
}
unlink(file.path(out_dir, "timing.json"), force = TRUE)

timing_records <- list()
record_timing <- function(method, rec, fit_seconds, extra = list()) {
  timing_records[[length(timing_records) + 1L]] <<- c(list(
    scenario = rec$scenario,
    rep = as.integer(rec$rep),
    method = method,
    fit_seconds = as.numeric(fit_seconds)
  ), extra)
}
write_timing <- function() {
  write_json(list(
    kind = "single_exposure_dlnm_mc",
    elapsed_seconds_total = as.numeric(proc.time()[["elapsed"]] - script_start),
    records = timing_records
  ), file.path(out_dir, "timing.json"), auto_unbox = TRUE, pretty = TRUE)
}

# penalised P-spline basis dimensions (smoothness selected by REML). Defaults
# match the upper end of the QAIC/QBIC search grid for old manifests.
vdf_pen <- if (!is.null(manifest$penalized_value_df)) {
  as.integer(unlist(manifest$penalized_value_df))
} else {
  max(vdf_grid)
}
ldf_pen <- if (!is.null(manifest$penalized_lag_df)) {
  as.integer(unlist(manifest$penalized_lag_df))
} else {
  max(ldf_grid)
}

# --- information criteria for a quasi-Poisson fit (Gasparrini & Armstrong) ----
# Gasparrini et al. (2010) eq. (13): -2*ll + kfac * phi * k, where k = vx*vl is
# the number of cross-basis parameters (NOT the model's fitted edf) and kfac = 2
# (QAIC) or log(n) (QBIC). Matches fit_chicago.R exactly.
.ic <- function(model, kfac, k_cb) {
  ll  <- sum(dpois(model$y, model$fitted.values, log = TRUE))
  phi <- summary(model)$dispersion
  -2 * ll + kfac * phi * k_cb
}
fqaic <- function(m, k_cb) .ic(m, 2, k_cb)
fqbic <- function(m, k_cb) .ic(m, log(length(m$y)), k_cb)

# --- output helpers -----------------------------------------------------------
prefixed_path <- function(f, prefix) file.path(dirname(f), paste0(prefix, basename(f)))
prefix_of <- function(m) c(qaic = "qaic_", qbic = "qbic_", pen = "pen_", tdlnm = "tree_")[[m]]

method_paths <- function(rec, method) {
  prefix <- prefix_of(method)
  c(
    file.path(out_dir, prefixed_path(rec$cumulative, prefix)),
    file.path(out_dir, prefixed_path(rec$surface, prefix))
  )
}

clear_method_outputs <- function(rec, method) {
  unlink(method_paths(rec, method), force = TRUE)
}

# --- write a crosspred's cumulative + surface CSVs with a filename prefix ------
write_cp <- function(cp, rec, prefix) {
  cum <- data.frame(value = grid,
                    fit = as.numeric(cp$allRRfit),
                    lo  = as.numeric(cp$allRRlow),
                    hi  = as.numeric(cp$allRRhigh))
  write.csv(cum, file.path(out_dir, prefixed_path(rec$cumulative, prefix)), row.names = FALSE)
  mat <- cp$matRRfit                      # rows = grid values, cols = lags
  low <- cp$matRRlow
  high <- cp$matRRhigh
  surf <- data.frame(value = rep(grid, times = ncol(mat)),
                     lag   = rep(0:(ncol(mat) - 1), each = nrow(mat)),
                     rr    = as.numeric(mat),
                     lo    = as.numeric(low),
                     hi    = as.numeric(high))
  write.csv(surf, file.path(out_dir, prefixed_path(rec$surface, prefix)), row.names = FALSE)
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

tdlnm_paths <- function(rec) {
  list(
    cumulative = method_paths(rec, "tdlnm")[[1]],
    surface = method_paths(rec, "tdlnm")[[2]]
  )
}

record_tdlnm_status <- function(rec, status, attempts, message = "") {
  list(
    scenario = rec$scenario,
    rep = as.integer(rec$rep),
    status = status,
    attempts = as.integer(attempts),
    message = message
  )
}

# --- QAIC/QBIC: one shared grid search, both criteria scored per fit ----------
fit_ic <- function(rec, x, y, want) {
  search_start <- proc.time()[["elapsed"]]
  n_grid_fits <- 0L
  rng <- range(x, na.rm = TRUE)
  var_knots <- function(vdf) rng[1] + diff(rng) / vdf * seq_len(vdf - 1)

  best <- list(qaic = list(q = Inf, vdf = NA, ldf = NA),
               qbic = list(q = Inf, vdf = NA, ldf = NA))
  for (vdf in vdf_grid) for (ldf in ldf_grid) {
    n_grid_fits <- n_grid_fits + 1L
    cb <- crossbasis(x, lag = lag_max,
                     argvar = list(fun = "ns", knots = var_knots(vdf)),
                     arglag = log_lag_ns(ldf))
    m <- glm(y ~ cb, family = quasipoisson(), na.action = na.omit)
    if (any(is.na(coef(m)))) next                 # rank-deficient -> skip
    qa <- tryCatch(fqaic(m, vdf * ldf), error = function(e) Inf)
    qb <- tryCatch(fqbic(m, vdf * ldf), error = function(e) Inf)
    if (is.finite(qa) && qa < best$qaic$q) best$qaic <- list(q = qa, vdf = vdf, ldf = ldf)
    if (is.finite(qb) && qb < best$qbic$q) best$qbic <- list(q = qb, vdf = vdf, ldf = ldf)
  }
  search_seconds <- proc.time()[["elapsed"]] - search_start

  for (ic in want) {                              # ic in {"qaic","qbic"}
    refit_start <- proc.time()[["elapsed"]]
    b <- best[[ic]]
    if (!is.finite(b$q))
      stop(sprintf("no full-rank %s fit for %s rep %d", ic, rec$scenario, rec$rep))
    # refit FRESH with a stably-named cross-basis so crosspred matches coef names
    cb_temp <- crossbasis(x, lag = lag_max,
                          argvar = list(fun = "ns", knots = var_knots(b$vdf)),
                          arglag = log_lag_ns(b$ldf))
    fit <- glm(y ~ cb_temp, family = quasipoisson(), na.action = na.omit)
    cp  <- crosspred(cb_temp, fit, at = grid, cen = reference,
                     bylag = 1, ci.level = ci_level)
    write_cp(cp, rec, paste0(ic, "_"))
    refit_seconds <- proc.time()[["elapsed"]] - refit_start
    record_timing(ic, rec, search_seconds + refit_seconds, list(
      search_seconds = as.numeric(search_seconds),
      refit_seconds = as.numeric(refit_seconds),
      grid_fits = as.integer(n_grid_fits),
      selected_value_df = as.integer(b$vdf),
      selected_lag_df = as.integer(b$ldf),
      shared_search = length(want) > 1L
    ))
    cat(sprintf("  %-18s rep %3d  %-4s %.1f  (vdf=%d, ldf=%d)\n",
                rec$scenario, rec$rep, toupper(ic), b$q, b$vdf, b$ldf))
  }
}

# --- penalised P-spline DLNM (REML chooses smoothness) ------------------------
fit_pen <- function(rec, x, y) {
  fit_start <- proc.time()[["elapsed"]]
  cb_temp <- crossbasis(x, lag = lag_max,
                        argvar = list(fun = "ps", df = vdf_pen),
                        arglag = list(fun = "ps", df = ldf_pen))
  cbp <- cbPen(cb_temp)                            # P-spline difference penalties
  fit <- gam(y ~ cb_temp, family = quasipoisson(),
             paraPen = list(cb_temp = cbp), method = "REML", na.action = na.omit)
  cp  <- crosspred(cb_temp, fit, at = grid, cen = reference,
                   bylag = 1, ci.level = ci_level)
  write_cp(cp, rec, "pen_")

  # k-check: cbPen enters via paraPen (not s()), so gam.check() does not apply.
  # Instead compare the term's effective df to the number of basis columns it
  # COULD use. If edf is near-saturated (close to k_cols), the basis dimension is
  # binding -> the penalty is not what controls smoothness, and vdf_pen/ldf_pen
  # should be raised or the penalised comparison is secretly df-limited.
  cb_cols  <- grep("^cb_temp", names(coef(fit)))
  k_cols   <- length(cb_cols)                      # available basis dimension
  term_edf <- sum(fit$edf[cb_cols])                # effective df actually used
  record_timing("pen", rec, proc.time()[["elapsed"]] - fit_start, list(
    value_df = as.integer(vdf_pen),
    lag_df = as.integer(ldf_pen),
    edf = as.numeric(term_edf),
    basis_dim = as.integer(k_cols)
  ))
  flag <- if (term_edf > 0.85 * k_cols) "  <-- k BINDING, raise vdf_pen/ldf_pen" else ""
  cat(sprintf("  %-18s rep %3d  PEN  REML edf=%.1f / k=%d (%.0f%%)  (basis vdf=%d, ldf=%d)%s\n",
              rec$scenario, rec$rep, term_edf, k_cols, 100 * term_edf / k_cols,
              vdf_pen, ldf_pen, flag))
}

summary_tdlnm <- NULL
if ("tdlnm" %in% methods) {
  summary_tdlnm <- getS3method("summary", "tdlnm", optional = TRUE)
  if (is.null(summary_tdlnm)) {
    summary_tdlnm <- getFromNamespace("summary.tdlnm", "dlmtree")
  }
}

fit_tdlnm_once <- function(rec, attempt) {
  paths <- tdlnm_paths(rec)
  dat <- read.csv(file.path(out_dir, rec$data))
  x_lag <- make_lag_matrix(dat[[exposure]], lag_max)
  y <- dat[[target]][(lag_max + 1):nrow(dat)]
  ok <- complete.cases(x_lag) & is.finite(y)
  x_lag <- x_lag[ok, , drop = FALSE]
  y <- y[ok]

  # dlmtree may drop matrix dimensions when there is only one fixed-effect
  # column. A centered trend keeps the fixed-effect design two-dimensional.
  fit_data <- data.frame(
    y = log1p(y),
    trend = seq(-0.5, 0.5, length.out = length(y))
  )

  scenario_seed <- sum(utf8ToInt(as.character(rec$scenario)))
  seed <- base_seed + as.integer(rec$rep)
  if (attempt > 1L) {
    seed <- seed + scenario_seed + 100000L * (attempt - 1L)
  }
  set.seed(seed)

  fit <- dlmtree::dlmtree(
    formula = y ~ trend,
    data = fit_data,
    exposure.data = x_lag,
    dlm.type = "nonlinear",
    family = "gaussian",
    control.tdlnm = list(exposure.splits = tdlnm_exposure_splits),
    # n.trees is passed as a double, deliberately. dlmtree guards these with
    #   if (all(sapply(list(n.trees, n.burn, n.iter, n.thin),
    #                  function(i) is.integer(i) & i > 0))) stop(...)
    # whose condition is inverted: it aborts when all four ARE valid positive
    # integers. The package's own default n.trees = 20 is a double, so every
    # call that leaves it alone slips past the guard. Passing 20L instead makes
    # all four integers and trips it. Keeping the double reproduces the default
    # code path exactly while still stating the tree count explicitly.
    control.mcmc = list(n.burn = tdlnm_burn, n.iter = tdlnm_iter,
                        n.thin = tdlnm_thin, n.trees = as.numeric(tdlnm_trees)),
    control.diagnose = list(verbose = FALSE)
  )

  # summary.tdlnm filters pred.at to fit$Xrange. The MC study scores every
  # estimator on the manifest grid, including sparse boundary points.
  fit$Xrange <- range(grid)
  sm <- summary_tdlnm(
    fit,
    conf.level = ci_level,
    pred.at = grid,
    cenval = reference,
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

  cat(sprintf("  %-18s rep %3d  TDLNM saved (attempt %d)\n", rec$scenario, rec$rep, attempt))
}

fit_tdlnm <- function(rec) {
  paths <- tdlnm_paths(rec)
  last_message <- ""
  fit_start <- proc.time()[["elapsed"]]
  for (attempt in seq_len(tdlnm_attempts)) {
    err <- tryCatch({
      fit_tdlnm_once(rec, attempt)
      NULL
    }, error = function(e) e)

    if (is.null(err)) {
      elapsed <- proc.time()[["elapsed"]] - fit_start
      record_timing("tdlnm", rec, elapsed, list(
        attempts = as.integer(attempt),
        status = "saved"
      ))
      status <- record_tdlnm_status(rec, "saved", attempt)
      status$fit_seconds <- as.numeric(elapsed)
      return(status)
    }

    last_message <- conditionMessage(err)
    unlink(unlist(paths, use.names = FALSE), force = TRUE)
    cat(sprintf(
      "  %-18s rep %3d  TDLNM attempt %d failed: %s\n",
      rec$scenario, rec$rep, attempt, last_message
    ))
  }

  cat(sprintf("  %-18s rep %3d  TDLNM failed after %d attempts\n",
              rec$scenario, rec$rep, tdlnm_attempts))
  elapsed <- proc.time()[["elapsed"]] - fit_start
  record_timing("tdlnm", rec, elapsed, list(
    attempts = as.integer(tdlnm_attempts),
    status = "failed"
  ))
  status <- record_tdlnm_status(rec, "failed", tdlnm_attempts, last_message)
  status$fit_seconds <- as.numeric(elapsed)
  status
}

fit_one <- function(rec) {
  for (method in methods) clear_method_outputs(rec, method)
  dat <- read.csv(file.path(out_dir, rec$data))
  x <- dat[[exposure]]; y <- dat[[target]]
  ic_want <- intersect(methods, c("qaic", "qbic"))
  if (length(ic_want)) fit_ic(rec, x, y, ic_want)
  if ("pen" %in% methods) fit_pen(rec, x, y)
  if ("tdlnm" %in% methods) return(fit_tdlnm(rec))
  NULL
}

cat(sprintf("Fitting DLNM-family methods [%s] to %d datasets (lag_max=%d, ref=%.1f)\n",
            paste(methods, collapse = ", "), length(manifest$datasets), lag_max, reference))
statuses <- lapply(manifest$datasets, fit_one)
tdlnm_statuses <- Filter(Negate(is.null), statuses)
if (length(tdlnm_statuses)) {
  write_json(tdlnm_statuses, file.path(out_dir, "tdlnm_fit_status.json"),
             auto_unbox = TRUE, pretty = TRUE)
  failed <- vapply(tdlnm_statuses, function(x) identical(x$status, "failed"), logical(1))
  if (any(failed)) {
    cat(sprintf("warning: %d TDLNM fit(s) failed; missing outputs will be skipped by Python scoring.\n",
                sum(failed)))
  }
}
write_timing()
cat("done.\n")
