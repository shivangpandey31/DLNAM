#!/usr/bin/env python3
"""Generate results_numbers.tex and supp_tables.tex from the experiment JSONs.

Usage:  python experiments/export_results_tex.py <dlnam_dir> <out_dir>

    python experiments/export_results_tex.py . "../Overleaf files"

Every number quoted in the manuscript comes from here; no digits are typed by hand.
Re-run after the high-replication job and recompile -- the prose is untouched.

Covers both evaluation targets. Cumulative exposure-response summaries live under
"results" in each JSON and produce the Mc/Abl/Exu macro families; full exposure-lag
surface summaries live under "surface_results" and produce the matching McSurf/
AblSurf/ExuSurf families. Runtime, environment, treed-sampler status, Chicago lag
shares and the training configuration are emitted too, so the specification tables
in the supplement never drift from the run that produced the results.
"""
import json, sys, os

SCEN_NUM = {
    "dgp1": 1, "dgp2": 2, "dgp3": 3, "dgp4": 4,
    "smooth": 1, "delayed_peaks": 2, "localized_peak": 3, "tilting_threshold": 4,
}
SCEN = {
    "dgp1": "DgpOne", "dgp2": "DgpTwo", "dgp3": "DgpThree", "dgp4": "DgpFour",
    "smooth": "DgpOne", "delayed_peaks": "DgpTwo",
    "localized_peak": "DgpThree", "tilting_threshold": "DgpFour",
}
MODEL = {"DLNAM": "Dlnam", "QAIC": "Qaic", "QBIC": "Qbic", "Penalised": "Pen", "TDLNM": "Tdlnm",
         "concat": "Concat", "unified_shared_bias": "Shared", "unified_local_bias": "Local",
         "reference": "Ref", "no_exu": "NoExu", "no_subnets": "NoSub", "no_smooth": "NoSmooth"}
REG = {"tot": "Tot", "int": "Int", "bnd": "Bnd"}
SCEN_LABEL = {key: f"DGP {num}" for key, num in SCEN_NUM.items()}


def _scenario_num(key: str) -> int:
    return SCEN_NUM[key]


def _is_dgp1(key: str) -> bool:
    return _scenario_num(key) == 1


def _dgp1_key(keys):
    return next((key for key in keys if _is_dgp1(key)), keys[0])
# Display names must match the figure legends exactly.
MC_LABEL = {"DLNAM": "DLNAM", "QAIC": "DLNM (QAIC)", "QBIC": "DLNM (QBIC)",
            "Penalised": "P-DLNM", "TDLNM": "T-DLNM"}


def num(x, dp):
    return f"{x:.{dp}f}"


def _grid_step(settings):
    """Spacing of the evaluation grid, from its range and point count."""
    lo, hi = min(settings["value_range"]), max(settings["value_range"])
    return (hi - lo) / (settings["n_value_grid"] - 1)


def emit_macros(d, prefix, out, key="results"):
    """RMSE/coverage/bias/var macros for one experiment.

    `key` selects the evaluation target: "results" holds the cumulative
    exposure-response summaries, "surface_results" the full exposure-lag
    surface summaries. Both dictionaries share the same schema, so the same
    emitter serves each; callers distinguish them through `prefix`.
    """
    for s in d["scenarios"]:
        for m in d["models"]:
            v = d[key][s][m]
            tag = prefix + SCEN[s] + MODEL[m]
            for r in d["regions"]:
                out.append(rf"\newcommand{{\{tag}Err{REG[r]}}}{{{num(v[f'err_{r}'],4)}}}")
                out.append(rf"\newcommand{{\{tag}ErrSE{REG[r]}}}{{{num(v[f'err_{r}_se'],4)}}}")
                out.append(rf"\newcommand{{\{tag}Cov{REG[r]}}}{{{num(v[f'cov_{r}'],3)}}}")
                out.append(rf"\newcommand{{\{tag}CovSE{REG[r]}}}{{{num(v[f'cov_{r}_se'],3)}}}")
                b, va = v[f"bias2_{r}"], v[f"var_{r}"]
                share = 100 * b / (b + va) if (b + va) > 0 else 0.0
                out.append(rf"\newcommand{{\{tag}BiasShare{REG[r]}}}{{{num(share,0)}}}")
    return out


def ratio_macros(d, out):
    """Derived: DLNAM advantage ratios and boundary/interior degradation."""
    for s in d["scenarios"]:
        dl = d["results"][s]["DLNAM"]
        out.append(rf"\newcommand{{\Mc{SCEN[s]}DegDlnam}}{{{num(dl['err_bnd']/dl['err_int'],2)}}}")
        for m in d["models"]:
            if m == "DLNAM":
                continue
            v = d["results"][s][m]
            out.append(rf"\newcommand{{\Mc{SCEN[s]}Ratio{MODEL[m]}}}{{{num(v['err_tot']/dl['err_tot'],1)}}}")
            out.append(rf"\newcommand{{\Mc{SCEN[s]}Deg{MODEL[m]}}}{{{num(v['err_bnd']/v['err_int'],2)}}}")
    return out


MC_TAGS = [("DLNAM", "Dlnam"), ("QAIC", "Qaic"), ("QBIC", "Qbic"),
           ("Penalised", "Pen"), ("TDLNM", "Tdlnm")]
ABL_TAGS = [("no_exu", "NoExu"), ("no_subnets", "NoSub"), ("no_smooth", "NoSmooth")]
EXU_TAGS = [("unified_shared_bias", "Shared"), ("unified_local_bias", "Local")]


def _best_count(d, model, key):
    """How many scenarios `model` attains the strictly lowest total RMSE in."""
    n = 0
    for s in d["scenarios"]:
        errs = {m: d[key][s][m]["err_tot"] for m in d["models"]}
        if min(errs, key=errs.get) == model:
            n += 1
    return n


def derived_macros(mc, exu, abl, out, key="results", mcp="Mc", ablp="Abl", exup="Exu"):
    """Ranges across DGPs: the only quantities quoted in the Results prose.

    Called twice, once for the cumulative target and once for the full
    exposure-lag surface, with the prefixes distinguishing the two families.
    """
    def rng(d, num_model, den_model, metric="err_tot", tag=""):
        vals = [d[key][s][num_model][metric] / d[key][s][den_model][metric]
                for s in d["scenarios"]]
        out.append(rf"\newcommand{{\{tag}Min}}{{{min(vals):.1f}}}")
        out.append(rf"\newcommand{{\{tag}Max}}{{{max(vals):.1f}}}")

    # DLNAM advantage over each comparator (total RMSE), min/max across DGPs
    for m, t in MC_TAGS[1:]:
        rng(mc, m, "DLNAM", tag=mcp + "Ratio" + t)
    # spread of DLNAM error across DGPs (max/min)
    e = [mc[key][s]["DLNAM"]["err_tot"] for s in mc["scenarios"]]
    out.append(rf"\newcommand{{\{mcp}DlnamSpread}}{{{max(e)/min(e):.1f}}}")
    # how many of the four scenarios each estimator wins outright
    for m, t in MC_TAGS:
        out.append(rf"\newcommand{{\{mcp}Best{t}}}{{{_best_count(mc, m, key)}}}")
    # boundary/interior degradation, min/max across DGPs
    for m, t in MC_TAGS:
        v = [mc[key][s][m]["err_bnd"] / mc[key][s][m]["err_int"] for s in mc["scenarios"]]
        out.append(rf"\newcommand{{\{mcp}Deg{t}Min}}{{{min(v):.1f}}}")
        out.append(rf"\newcommand{{\{mcp}Deg{t}Max}}{{{max(v):.1f}}}")
    # coverage ranges as integer percentages
    for m, t in MC_TAGS:
        c = [mc[key][s][m]["cov_tot"] for s in mc["scenarios"]]
        out.append(rf"\newcommand{{\{mcp}Cov{t}PctMin}}{{{100*min(c):.0f}}}")
        out.append(rf"\newcommand{{\{mcp}Cov{t}PctMax}}{{{100*max(c):.0f}}}")
    mcse = max(mc[key][s][m]["cov_tot_se"] for s in mc["scenarios"] for m in mc["models"])
    out.append(rf"\newcommand{{\{mcp}CovMcsePctMax}}{{{100*mcse:.0f}}}")
    # ablation and ExU degradation ranges
    for m, t in ABL_TAGS:
        rng(abl, m, "reference", tag=ablp + "Ratio" + t)
        c = [abl[key][s][m]["cov_tot"] for s in abl["scenarios"]]
        out.append(rf"\newcommand{{\{ablp}Cov{t}PctMin}}{{{100*min(c):.0f}}}")
        out.append(rf"\newcommand{{\{ablp}Cov{t}PctMax}}{{{100*max(c):.0f}}}")
    for m, t in EXU_TAGS:
        rng(exu, m, "concat", tag=exup + "Ratio" + t)
        c = [exu[key][s][m]["cov_tot"] for s in exu["scenarios"]]
        out.append(rf"\newcommand{{\{exup}Cov{t}PctMin}}{{{100*min(c):.0f}}}")
    return out


SIMPLE_SCENARIO = "dgp1"


def contrast_macros(mc, out, key="results", prefix="Mc"):
    """DLNAM against the *best* competing estimator in each DGP.

    The manuscript's central comparison is not an average over comparators but
    the margin against whichever comparator does best on that surface, split by
    whether the generating surface is the simple separable one or one of DGPs 2--4. A ratio above one means the DLNAM is ahead.
    """
    complex_scen = [s for s in mc["scenarios"] if not _is_dgp1(s)]
    simple_key = _dgp1_key(mc["scenarios"])

    def best_other(s, metric):
        return min(mc[key][s][m][metric] for m in mc["models"] if m != "DLNAM")

    for metric, tag in (("err_tot", "VsBest"), ("bias2_tot", "BiasVsBest")):
        ratios = {}
        for s in mc["scenarios"]:
            dl = mc[key][s]["DLNAM"][metric]
            ratios[s] = best_other(s, metric) / dl if dl > 0 else float("nan")
            out.append(rf"\newcommand{{\{prefix}{SCEN[s]}{tag}}}{{{ratios[s]:.1f}}}")
        out.append(rf"\newcommand{{\{prefix}{tag}Simple}}{{{ratios[simple_key]:.1f}}}")
        cx = [ratios[s] for s in complex_scen]
        out.append(rf"\newcommand{{\{prefix}{tag}ComplexMin}}{{{min(cx):.1f}}}")
        out.append(rf"\newcommand{{\{prefix}{tag}ComplexMax}}{{{max(cx):.1f}}}")

    # Largest bias share carried by any comparator, and the DLNAM's own, on the
    # complex surfaces: the bias-versus-variance composition of the error.
    def share(s, m):
        v = mc[key][s][m]
        b, va = v["bias2_tot"], v["var_tot"]
        return 100 * b / (b + va) if (b + va) > 0 else 0.0

    dl_shares = [share(s, "DLNAM") for s in complex_scen]
    out.append(rf"\newcommand{{\{prefix}BiasShareDlnamComplexMin}}{{{min(dl_shares):.0f}}}")
    out.append(rf"\newcommand{{\{prefix}BiasShareDlnamComplexMax}}{{{max(dl_shares):.0f}}}")
    td = [share(s, "TDLNM") for s in complex_scen]
    out.append(rf"\newcommand{{\{prefix}BiasShareTdlnmComplexMin}}{{{min(td):.0f}}}")
    out.append(rf"\newcommand{{\{prefix}BiasShareTdlnmComplexMax}}{{{max(td):.0f}}}")
    out.append(rf"\newcommand{{\{prefix}NComplex}}{{{len(complex_scen)}}}")
    return out


def width_macros(mc, out, key="results", prefix="Mc"):
    """How much wider a comparator's interval is where it out-covers the DLNAM.

    Coverage on its own cannot separate a well-calibrated interval from an
    uninformatively wide one, so the comparison of interest is the width paid
    for the extra coverage. Emitted only where the runs recorded widths.
    """
    ratios, exceptions = [], 0
    for s in mc["scenarios"]:
        d = mc[key][s].get("DLNAM", {})
        if "width_tot" not in d:
            return out
        for m in mc["models"]:
            if m == "DLNAM":
                continue
            v = mc[key][s][m]
            if v["cov_tot"] <= d["cov_tot"]:
                continue
            r = v["width_tot"] / d["width_tot"]
            ratios.append(r) if r > 1 else exceptions
            if r <= 1:
                exceptions += 1
    if not ratios:
        return out
    out.append(rf"\newcommand{{\{prefix}WidthRatioLo}}{{{min(ratios):.1f}}}")
    out.append(rf"\newcommand{{\{prefix}WidthRatioHi}}{{{max(ratios):.1f}}}")
    out.append(rf"\newcommand{{\{prefix}WidthExceptions}}{{{exceptions}}}")
    return out


def inflated_coverage_macros(mc, out, key="results", prefix="Mc"):
    """What the representation term contributes to the reported coverage.

    The reported interval combines the last-layer Laplace variance with the
    between-member spread. `covcond_*` is the counterfactual coverage of the
    Laplace term alone, so the difference is the contribution of the omitted
    representation uncertainty. Emitted only when the runs retained the
    quantities needed, so older result files simply yield no macros.
    """
    cond, base = [], []
    for s in mc["scenarios"]:
        v = mc[key][s].get("DLNAM", {})
        if "covcond_tot" not in v:
            return out
        cond.append(v["covcond_tot"])
        base.append(v["cov_tot"])
    out.append(rf"\newcommand{{\{prefix}CovCondDlnamPctMin}}{{{100*min(cond):.0f}}}")
    out.append(rf"\newcommand{{\{prefix}CovCondDlnamPctMax}}{{{100*max(cond):.0f}}}")
    gain = [100 * (a - b) for a, b in zip(base, cond)]
    out.append(rf"\newcommand{{\{prefix}CovEnsGainPctMin}}{{{min(gain):.0f}}}")
    out.append(rf"\newcommand{{\{prefix}CovEnsGainPctMax}}{{{max(gain):.0f}}}")
    return out


def runtime_macros(rt, out):
    """Wall-clock and peak-memory macros for the joint-exposure scaling study.

    Runtime is the primary metric. Memory is emitted for descriptive use only:
    the DLNAM figure combines host RSS with peak reserved CUDA memory, while the
    R comparators are CPU processes measured by RSS alone, so the two are not
    on a common scale.
    """
    counts = [str(c) for c in rt["settings"]["exposure_counts"]]
    lo, hi = counts[0], counts[-1]
    out.append(rf"\newcommand{{\RtExpLo}}{{{lo}}}")
    out.append(rf"\newcommand{{\RtExpHi}}{{{hi}}}")
    out.append(rf"\newcommand{{\RtRepeats}}{{{rt['settings']['timing_repeats']}}}")
    missing = []
    for m, t in MC_TAGS:
        cells = [rt["results"].get(c, {}).get(m) for c in counts]
        if any(c is None or "runtime" not in c for c in cells):
            missing.append(m)
            continue
        secs = [c["runtime"]["median_seconds"] for c in cells]
        mems = [c["peak_memory"]["median_bytes"] / 1e9 for c in cells]
        out.append(rf"\newcommand{{\Rt{t}Lo}}{{{secs[0]:.0f}}}")
        out.append(rf"\newcommand{{\Rt{t}Hi}}{{{secs[-1]:.0f}}}")
        out.append(rf"\newcommand{{\Rt{t}Growth}}{{{secs[-1]/secs[0]:.1f}}}")
        out.append(rf"\newcommand{{\Mem{t}Lo}}{{{mems[0]:.1f}}}")
        out.append(rf"\newcommand{{\Mem{t}Hi}}{{{mems[-1]:.1f}}}")
    out.append(rf"\newcommand{{\RtComplete}}{{{'no' if missing else 'yes'}}}")
    if missing:
        out.append("% WARNING: runtime_scaling.json is missing cells for: "
                   + ", ".join(missing))
    hw = rt.get("hardware", {})
    out.append(rf"\newcommand{{\RtGpu}}{{{hw.get('gpu', 'unknown')}}}")
    out.append(rf"\newcommand{{\RtCpuCores}}{{{hw.get('logical_cpu_count', '?')}}}")
    return out


def _sci(x):
    """Render a positive float as LaTeX scientific notation, e.g. 8\\cdot10^{-4}."""
    from math import floor, log10
    x = float(x)
    if x == 0:
        return "0"
    e = int(floor(log10(abs(x))))
    m = x / (10 ** e)
    ms = f"{m:g}"
    return ms if e == 0 else rf"{ms}\cdot10^{{{e}}}"


def config_macros(mc, chi, out):
    """Training and comparator settings, so the specification tables in the
    supplement are generated from the same JSONs as the results."""
    s = mc["settings"]
    out.append(rf"\newcommand{{\TrLr}}{{{_sci(s['learning_rate'])}}}")
    out.append(rf"\newcommand{{\TrLrMin}}{{{_sci(s['minimum_learning_rate'])}}}")
    out.append(rf"\newcommand{{\TrWeightDecay}}{{{_sci(s['weight_decay'])}}}")
    out.append(rf"\newcommand{{\TrGradClip}}{{{s['gradient_clip']:g}}}")
    out.append(rf"\newcommand{{\TrSchedule}}{{{s['schedule']}}}")
    out.append(rf"\newcommand{{\TrSubnets}}{{{s['n_subnets']}}}")
    out.append(rf"\newcommand{{\TrWidths}}{{({', '.join(str(w) for w in s['hidden_widths'])})}}")
    out.append(rf"\newcommand{{\TrActivation}}{{{s['activation']}}}")
    out.append(rf"\newcommand{{\TrExuMeanX}}{{{s['exu_weight_mean']:g}}}")
    out.append(rf"\newcommand{{\TrExuMeanLag}}{{{s['exu_lag_weight_mean']:g}}}")
    out.append(rf"\newcommand{{\TrExuStd}}{{{s['exu_weight_std']:g}}}")
    out.append(rf"\newcommand{{\TrSeed}}{{{s['seed']}}}")
    out.append(rf"\newcommand{{\TrDevice}}{{{s['device'].upper()}}}")
    out.append(rf"\newcommand{{\CmpPenDf}}{{{s['penalized_value_df']}}}")
    out.append(rf"\newcommand{{\CmpChiPenDf}}{{{chi['settings']['penalized_value_df']}}}")
    out.append(rf"\newcommand{{\TdlnmBurn}}{{{s['tdlnm_burn']:,}}}".replace(",", "{,}"))
    out.append(rf"\newcommand{{\TdlnmIter}}{{{s['tdlnm_iter']:,}}}".replace(",", "{,}"))
    out.append(rf"\newcommand{{\TdlnmThin}}{{{s['tdlnm_thin']}}}")
    out.append(rf"\newcommand{{\TdlnmSplits}}{{{s['tdlnm_exposure_splits']}}}")
    if "tdlnm_trees" in s:
        out.append(rf"\newcommand{{\TdlnmTrees}}{{{s['tdlnm_trees']}}}")
    out.append(rf"\newcommand{{\TdlnmAttempts}}{{{s['tdlnm_attempts']}}}")
    return out


def selection_macros(mc, out):
    """Instability of criterion-based cross-basis dimension selection.

    Each replicate of a scenario is a fresh draw from the *same* data-generating
    process, so any spread in the selected marginal dimensions is selection
    variability rather than a difference in the truth. The disagreement between
    QAIC and QBIC on the same search grid is recorded alongside it.
    """
    recs = (mc.get("r_timing") or {}).get("records", [])
    if not recs:
        return out
    sel = {}
    for r in recs:
        if "selected_value_df" not in r:
            continue
        sel.setdefault((r.get("method"), r.get("scenario")), []).append(
            (int(r["selected_value_df"]), int(r["selected_lag_df"]))
        )
    if not sel:
        return out

    for meth, tag in (("qaic", "Qaic"), ("qbic", "Qbic")):
        pairs = [p for (m, _), v in sel.items() if m == meth for p in v]
        if not pairs:
            continue
        vx = [a for a, _ in pairs]
        lg = [b for _, b in pairs]
        out.append(rf"\newcommand{{\Sel{tag}VdfLo}}{{{min(vx)}}}")
        out.append(rf"\newcommand{{\Sel{tag}VdfHi}}{{{max(vx)}}}")
        out.append(rf"\newcommand{{\Sel{tag}LdfLo}}{{{min(lg)}}}")
        out.append(rf"\newcommand{{\Sel{tag}LdfHi}}{{{max(lg)}}}")
        # widest within-scenario spread, and most distinct values, for one DGP
        spread = max(max(a for a, _ in v) - min(a for a, _ in v)
                     for (m, _), v in sel.items() if m == meth)
        distinct = max(len({a for a, _ in v})
                       for (m, _), v in sel.items() if m == meth)
        out.append(rf"\newcommand{{\Sel{tag}VdfSpreadMax}}{{{spread}}}")
        out.append(rf"\newcommand{{\Sel{tag}VdfDistinctMax}}{{{distinct}}}")

    # Size of the search the selected cross-basis fits were granted, and the
    # share of their runtime it consumes.
    gf = [r["grid_fits"] for r in recs if "grid_fits" in r]
    if gf:
        out.append(rf"\newcommand{{\DfGridCombos}}{{{max(gf)}}}")
    ss = [r["search_seconds"] for r in recs if "search_seconds" in r]
    rs = [r["refit_seconds"] for r in recs if "refit_seconds" in r]
    if ss and rs:
        srt, rrt = sorted(ss), sorted(rs)
        med_s = srt[len(srt) // 2]
        med_r = rrt[len(rrt) // 2]
        pct = 100 * med_s / (med_s + med_r) if (med_s + med_r) > 0 else 0.0
        out.append(rf"\newcommand{{\SearchSharePct}}{{{pct:.0f}}}")

    # Penalised DLNM: fitted effective degrees of freedom against the basis rank
    # made available. If the realised edf approached the rank, the rank rather
    # than the penalty would be controlling smoothness and the comparison would
    # be capacity-limited rather than penalty-limited.
    pen = [r for r in recs if r.get("method") == "pen" and "edf" in r]
    if pen:
        e = sorted(float(r["edf"]) for r in pen)
        out.append(rf"\newcommand{{\PenEdfMin}}{{{e[0]:.0f}}}")
        out.append(rf"\newcommand{{\PenEdfMed}}{{{e[len(e)//2]:.0f}}}")
        out.append(rf"\newcommand{{\PenEdfMax}}{{{e[-1]:.0f}}}")
        out.append(rf"\newcommand{{\PenBasisDim}}{{{int(pen[0]['basis_dim'])}}}")
        out.append(rf"\newcommand{{\PenEdfPctMax}}{{{100*e[-1]/int(pen[0]['basis_dim']):.0f}}}")

    # Largest coverage disagreement between the two criteria, percentage points.
    gaps = [abs(mc["results"][s]["QAIC"]["cov_tot"] - mc["results"][s]["QBIC"]["cov_tot"])
            for s in mc["scenarios"]]
    out.append(rf"\newcommand{{\SelIcCovGapMaxPct}}{{{100*max(gaps):.0f}}}")
    return out


def selection_table(mc):
    """Selected marginal dimensions per scenario and criterion, across replicates."""
    recs = (mc.get("r_timing") or {}).get("records", [])
    sel = {}
    for r in recs:
        if "selected_value_df" not in r:
            continue
        sel.setdefault((r.get("method"), r.get("scenario")), []).append(
            (int(r["selected_value_df"]), int(r["selected_lag_df"]))
        )
    if not sel:
        return ""
    rows = [r"\begin{table}[htbp]", r"\centering", r"\small",
            r"\begin{tabular}{l l cc cc}", r"\toprule",
            r" & & \multicolumn{2}{c}{Exposure df} & \multicolumn{2}{c}{Lag df}\\",
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
            r"Scenario & Criterion & Range & Distinct & Range & Distinct\\",
            r"\midrule"]
    for s in mc["scenarios"]:
        for i, (meth, lab) in enumerate((("qaic", "QAIC"), ("qbic", "QBIC"))):
            v = sel.get((meth, s))
            if not v:
                continue
            vx = [a for a, _ in v]
            lg = [b for _, b in v]
            name = SCEN_LABEL[s] if i == 0 else ""
            rows.append(
                f"{name} & {lab} & {min(vx)}--{max(vx)} & {len(set(vx))} & "
                f"{min(lg)}--{max(lg)} & {len(set(lg))}\\\\")
        rows.append(r"\addlinespace")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Cross-basis dimensions selected by QAIC and QBIC over the "
             r"prespecified search grid, across Monte Carlo replicates of the same "
             r"data-generating process. Every replicate within a scenario is a fresh "
             r"draw from an identical truth, so the range and the number of distinct "
             r"selected values measure selection variability, not differences in the "
             r"underlying surface.}",
             r"\label{tab:supp_selection}", r"\end{table}"]
    return "\n".join(rows)


def tdlnm_status_macros(mc, out):
    """Treed-DLNM sampler outcomes, so retries are reported rather than absorbed.

    The runner re-seeds and re-attempts a treed fit that fails to initialise. That
    is a defensible numerical safeguard, but it conditions the reported comparison
    on eventual success, so the attempt counts belong in the manuscript.
    """
    st = mc.get("tdlnm_fit_status") or []
    if not st:
        return out
    saved = [r for r in st if r.get("status") == "saved"]
    attempts = [int(r.get("attempts", 1)) for r in saved]
    out.append(rf"\newcommand{{\TdlnmFits}}{{{len(st)}}}")
    out.append(rf"\newcommand{{\TdlnmSaved}}{{{len(saved)}}}")
    out.append(rf"\newcommand{{\TdlnmFailed}}{{{len(st) - len(saved)}}}")
    out.append(rf"\newcommand{{\TdlnmRetried}}{{{sum(1 for a in attempts if a > 1)}}}")
    out.append(rf"\newcommand{{\TdlnmMaxAttempts}}{{{max(attempts) if attempts else 0}}}")
    return out


def environment_macros(mc, out):
    """Software versions actually used, so the supplement never drifts from the run."""
    pk = mc.get("environment", {}).get("packages", {})
    rv = mc.get("r_environment", {})
    rp = rv.get("packages", {})
    out.append(rf"\newcommand{{\EnvPython}}{{{mc.get('environment', {}).get('python', '?')}}}")
    for name, tag in [("torch", "Torch"), ("numpy", "Numpy"), ("pandas", "Pandas")]:
        out.append(rf"\newcommand{{\Env{tag}}}{{{pk.get(name, '?')}}}")
    rver = rv.get("r_version", "?").replace("R version ", "").split(" ")[0]
    out.append(rf"\newcommand{{\EnvR}}{{{rver}}}")
    for name, tag in [("dlnm", "Dlnm"), ("mgcv", "Mgcv"), ("dlmtree", "Dlmtree")]:
        out.append(rf"\newcommand{{\Env{tag}}}{{{rp.get(name, '?')}}}")
    return out


def _fmt(x, dp=4):
    return f"{x:.{dp}f}"


def _fmt_scaled(x, scale=1e3):
    """Bias^2 / variance on a common 10^3x scale, fixed 4 decimals.

    Fixed decimals rather than significant figures, so cells share one format and
    align on the decimal point. The 10^3 scale with 4 dp is chosen so the cells are
    format-identical to the RMSE cells: over all three experiments the scaled values
    span 0.0058 to 75.47, giving one or two integer digits (versus one to three at
    10^4 with 2 dp) and nothing rounding away.
    """
    return f"{x*scale:.4f}"


def add_relative_error(d, baseline, key="results"):
    """Inject rel_{tot,int,bnd}: RMSE as a multiple of `baseline`'s RMSE.

    Follows the convention of the penalised-DLNM study, which reports simulation
    RMSE relative to a single reference fit rather than in absolute units, so the
    across-scenario comparison does not depend on each surface's effect size.
    That study's unit baseline is the model it proposes, so the baseline here is
    the DLNAM and every comparator cell reads as a multiple of its error.
    Values above 1 are worse than the baseline. No Monte Carlo SE is attached:
    numerator and denominator are computed on the same replicates and are not
    independent, so a naive ratio SE would be misleading.
    """
    if baseline not in d["models"]:
        return
    for s_ in d["scenarios"]:
        base = d[key][s_][baseline]
        for m in d["models"]:
            v = d[key][s_][m]
            for r in ("tot", "int", "bnd"):
                b = base.get(f"err_{r}")
                x = v.get(f"err_{r}")
                if b:
                    v[f"rel_{r}"] = x / b


def metric_table(d, caption, label, m1, m2, models_label=None, unit="Model",
                 key="results"):
    """DGP x Model, two metrics each decomposed over total/interior/boundary,
    with Monte Carlo standard errors. Both supplementary tables share this layout.

    m1, m2 = (key, header, formatter)
    """
    L = models_label or {m: m for m in d["models"]}
    k1, h1, f1 = m1
    k2, h2, f2 = m2
    rows = [r"\begin{sidewaystable}[p]", r"\centering", r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{ll ccc ccc}", r"\toprule",
            rf" & & \multicolumn{{3}}{{c}}{{{h1}}} & \multicolumn{{3}}{{c}}{{{h2}}}\\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            rf"DGP & {unit} & Total & Interior & Boundary & Total & Interior & Boundary\\",
            r"\midrule"]
    for s_ in d["scenarios"]:
        for i, m in enumerate(d["models"]):
            v = d[key][s_][m]
            name = SCEN_LABEL[s_] if i == 0 else ""
            cells = []
            for mkey, fmt in ((k1, f1), (k2, f2)):
                for r in ("tot", "int", "bnd"):
                    # A metric may carry no Monte Carlo SE (interval width, relative
                    # RMSE), and may be absent entirely on results produced before the
                    # metric was added; both degrade to a plain cell rather than
                    # breaking the build.
                    val = v.get(f"{mkey}_{r}")
                    if val is None:
                        cells.append("--")
                        continue
                    se = v.get(f"{mkey}_{r}_se")
                    cells.append(rf"{fmt(val)} ({fmt(se)})" if se is not None
                                 else rf"{fmt(val)}")
            rows.append(f"{name} & {L.get(m, m)} & " + " & ".join(cells) + r"\\")
        rows.append(r"\addlinespace")
    rows += [r"\bottomrule", r"\end{tabular}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{sidewaystable}"]
    return "\n".join(rows)


CHI_COLD, CHI_HEAT = -15.0, 30.0     # representative cold and hot days (deg C)
CHI_EARLY, CHI_MID = 3, 10           # lag blocks: 0-3, 4-10, 11-L


def chicago_macros(chi, out):
    """Where the Chicago cumulative effect accumulates over the lag window, and
    how wide each estimator's interval is at the extremes of the temperature range.

    Emitted so that the qualitative statements in the Results ("heat acts within
    days, cold accumulates over weeks") are carried by regenerated numbers.
    """
    import numpy as np

    surf = chi["surfaces"]["DLNAM"]
    v = np.asarray(surf["value"], float)
    lag = np.asarray(surf["lag"], float)
    lrr = np.log(np.asarray(surf["rr"], float))
    for temp, tag in ((CHI_COLD, "Cold"), (CHI_HEAT, "Heat")):
        vt = v[np.argmin(np.abs(v - temp))]
        sel = np.abs(v - vt) < 1e-9
        lg, val = lag[sel], lrr[sel]
        total = val.sum()
        early = val[lg <= CHI_EARLY].sum()
        mid = val[(lg > CHI_EARLY) & (lg <= CHI_MID)].sum()
        late = val[lg > CHI_MID].sum()
        out.append(rf"\newcommand{{\Chi{tag}Temp}}{{{vt:.0f}}}")
        out.append(rf"\newcommand{{\Chi{tag}CumRr}}{{{np.exp(total):.2f}}}")
        for share, part in ((early, "Early"), (mid, "Mid"), (late, "Late")):
            pct = 100 * share / total if total != 0 else 0.0
            out.append(rf"\newcommand{{\Chi{tag}Share{part}}}{{{pct:.0f}}}")
    out.append(rf"\newcommand{{\ChiLagEarly}}{{{CHI_EARLY}}}")
    out.append(rf"\newcommand{{\ChiLagMid}}{{{CHI_MID}}}")

    # Interval width on the relative-risk scale at the coldest evaluated day.
    for m, tag in MC_TAGS:
        cur = chi["curves"].get(m)
        if cur is None:
            continue
        width = float(np.asarray(cur["hi"], float)[0] - np.asarray(cur["lo"], float)[0])
        out.append(rf"\newcommand{{\ChiWidthCold{tag}}}{{{width:.2f}}}")
        fit = np.asarray(cur["fit"], float)
        vv = np.asarray(cur["value"], float)
        out.append(rf"\newcommand{{\ChiMmt{tag}}}{{{vv[int(np.argmin(fit))]:.1f}}}")
        out.append(rf"\newcommand{{\ChiHeatRr{tag}}}{{{fit[-1]:.2f}}}")
    return out


def runtime_table(rt):
    """Joint-exposure scaling: wall-clock and peak memory per estimator."""
    counts = [str(c) for c in rt["settings"]["exposure_counts"]]
    head = " & ".join(rf"\multicolumn{{1}}{{c}}{{{c}}}" for c in counts)
    rows = [r"\begin{table}[htbp]", r"\centering", r"\small",
            r"\begin{tabular}{l " + "c" * len(counts) + " c " + "c" * len(counts) + "}",
            r"\toprule",
            rf" & \multicolumn{{{len(counts)}}}{{c}}{{Runtime (s)}} & "
            rf"& \multicolumn{{{len(counts)}}}{{c}}{{Peak memory (GB)}}\\",
            rf"\cmidrule(lr){{2-{1+len(counts)}}}\cmidrule(lr){{{3+len(counts)}-{2+2*len(counts)}}}",
            rf"Estimator & {head} & & {head}\\",
            r"\midrule"]
    for m, _ in MC_TAGS:
        cells_t, cells_m = [], []
        for c in counts:
            cell = rt["results"].get(c, {}).get(m)
            if cell is None or "runtime" not in cell:
                cells_t.append("---"); cells_m.append("---"); continue
            cells_t.append(f"{cell['runtime']['median_seconds']:.0f}")
            cells_m.append(f"{cell['peak_memory']['median_bytes']/1e9:.1f}")
        rows.append(f"{MC_LABEL[m]} & " + " & ".join(cells_t) + " & & "
                    + " & ".join(cells_m) + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}",
             r"\caption{Computational scaling with the number of jointly fitted exposures. "
             r"Runtime covers model construction, preprocessing, fitting, uncertainty estimation "
             r"and effect extraction. Peak memory is the peak resident memory of the isolated "
             r"process tree, plus peak reserved CUDA memory for the DLNAM; the DLNAM figure is "
             r"therefore not on the same measurement scale as the CPU-only comparators and is "
             r"reported descriptively.}",
             r"\label{tab:supp_runtime}", r"\end{table}"]
    return "\n".join(rows)


def _results_dir(dlnam_dir):
    """Return the canonical results directory under the DLNAM project root."""
    jdir = os.path.join(os.path.abspath(dlnam_dir), "experiments", "results")
    if not os.path.isdir(jdir):
        raise FileNotFoundError(
            f"expected result JSONs under {jdir!r}; pass the DLNAM project folder"
        )
    return jdir


def main(dlnam_dir, odir):
    jdir = _results_dir(dlnam_dir)
    def load(name):
        path = os.path.join(jdir, name)
        return json.load(open(path)) if os.path.exists(path) else None

    mc = json.load(open(os.path.join(jdir, "mc_model_comparison.json")))
    exu = json.load(open(os.path.join(jdir, "mc_exu.json")))
    abl = json.load(open(os.path.join(jdir, "mc_ablation.json")))
    chi = json.load(open(os.path.join(jdir, "chicago_model_comparison.json")))
    rt = load("runtime_scaling.json")
    mal = load("malaria_model_comparison.json")

    out = ["% AUTO-GENERATED by export_results_tex.py -- do not edit by hand.",
           "% Safe to \\input more than once (e.g. from the preamble and from results.tex).",
           "\\ifdefined\\resultsnumbersloaded\\endinput\\fi",
           "\\newcommand{\\resultsnumbersloaded}{}",
           ""]
    # Cumulative exposure-response target.
    emit_macros(mc, "Mc", out); ratio_macros(mc, out)
    emit_macros(exu, "Exu", out)
    emit_macros(abl, "Abl", out)
    derived_macros(mc, exu, abl, out)
    contrast_macros(mc, out)
    inflated_coverage_macros(mc, out)
    width_macros(mc, out)
    # Full exposure-lag surface target (same schema, "Surf" macro families).
    out.append("")
    emit_macros(mc, "McSurf", out, key="surface_results")
    emit_macros(exu, "ExuSurf", out, key="surface_results")
    emit_macros(abl, "AblSurf", out, key="surface_results")
    derived_macros(mc, exu, abl, out, key="surface_results",
                   mcp="McSurf", ablp="AblSurf", exup="ExuSurf")
    contrast_macros(mc, out, key="surface_results", prefix="McSurf")
    inflated_coverage_macros(mc, out, key="surface_results", prefix="McSurf")
    width_macros(mc, out, key="surface_results", prefix="McSurf")

    st = mc["settings"]
    out += ["",
            rf"\newcommand{{\NReps}}{{{st['n_reps']}}}",
            rf"\newcommand{{\NRepsAbl}}{{{abl['settings']['n_reps']}}}",
            rf"\newcommand{{\NRepsExu}}{{{exu['settings']['n_reps']}}}",
            rf"\newcommand{{\NGrid}}{{{st['n_value_grid']}}}",
            # Also emitted as a sequence (lo, lo+step, ..., hi). The point count
            # is one more than the number of intervals, which reads as an
            # oddity in prose; the step states the spacing directly and makes
            # it evident that the centring reference falls on a grid node.
            rf"\newcommand{{\GridLo}}{{{min(st['value_range']):.0f}}}",
            rf"\newcommand{{\GridHi}}{{{max(st['value_range']):.0f}}}",
            rf"\newcommand{{\GridStep}}{{{_grid_step(st):g}}}",
            rf"\newcommand{{\GridNext}}{{{min(st['value_range']) + _grid_step(st):g}}}",
            rf"\newcommand{{\NSurfPts}}{{{st['n_surface_points']}}}",
            rf"\newcommand{{\NObs}}{{{st['n_obs']}}}",
            rf"\newcommand{{\NEpochs}}{{{st['epochs']}}}",
            rf"\newcommand{{\NEnsemble}}{{{st['n_ensemble']}}}",
            rf"\newcommand{{\LagMax}}{{{st['lag']}}}",
            rf"\newcommand{{\XRef}}{{{st['reference']:.0f}}}",
            rf"\newcommand{{\DfGridLo}}{{{min(st['value_df_grid'])}}}",
            rf"\newcommand{{\DfGridHi}}{{{max(st['value_df_grid'])}}}"]

    bnd = mc["boundary"][_dgp1_key(mc["scenarios"])]
    out += [rf"\newcommand{{\BndLo}}{{{bnd[0]:.1f}}}", rf"\newcommand{{\BndHi}}{{{bnd[1]:.1f}}}"]

    tim = [mc["timing"][s]["DLNAM"]["fit_seconds_mean"] for s in mc["timing"]]
    out += [rf"\newcommand{{\McFitSecs}}{{{sum(tim)/len(tim):.0f}}}"]

    cs, cp, cl = chi["settings"], chi["dlnam_performance"], chi["dlnam_laplace"]
    out += ["",
            rf"\newcommand{{\ChiLagMax}}{{{cs['lag_max']}}}",
            rf"\newcommand{{\ChiRef}}{{{cs['reference']:.1f}}}",
            rf"\newcommand{{\ChiTrendDf}}{{{cs['trend_df']}}}",
            rf"\newcommand{{\ChiDfLo}}{{{min(cs['value_df_grid'])}}}",
            rf"\newcommand{{\ChiDfHi}}{{{max(cs['value_df_grid'])}}}",
            rf"\newcommand{{\ChiNObs}}{{{chi['dlnam_fit_summary']['n_samples']}}}",
            rf"\newcommand{{\ChiPhi}}{{{cp['Phi']:.2f}}}",
            rf"\newcommand{{\ChiLaplacePhi}}{{{cl['phi']:.2f}}}",
            rf"\newcommand{{\ChiPriorPrec}}{{{cl['prior_precision']:.1f}}}",
            rf"\newcommand{{\ChiMcFadden}}{{{cp['McFadden_R2']:.3f}}}",
            rf"\newcommand{{\ChiFitSecs}}{{{chi['dlnam_fit_summary']['fit_seconds']:.0f}}}"]

    out.append("")
    chicago_macros(chi, out)
    if rt is not None:
        out.append("")
        runtime_macros(rt, out)
    if mal is not None:
        ms = mal["settings"]
        any_fit = next(iter(mal["dlnam_fit_summary"].values()))
        out += ["",
                rf"\newcommand{{\MalNObs}}{{{any_fit['n_samples']:,}}}".replace(",", "{,}"),
                rf"\newcommand{{\MalNExp}}{{{len(mal['exposures'])}}}",
                rf"\newcommand{{\MalNLags}}{{{ms['lag_count']}}}",
                rf"\newcommand{{\MalEpochs}}{{{ms['epochs']}}}",
                rf"\newcommand{{\MalBatchPct}}{{{100*ms['batch_fraction']:.0f}}}",
                rf"\newcommand{{\MalDlnmDf}}{{{ms['reference_dlnm_value_df']}}}"]
    out.append("")
    config_macros(mc, chi, out)
    selection_macros(mc, out)
    tdlnm_status_macros(mc, out)
    environment_macros(mc, out)

    with open(os.path.join(odir, "results_numbers.tex"), "w") as f:
        f.write("\n".join(out) + "\n")

    exu_lab = exu["settings"]["labels"]; abl_lab = abl["settings"]["labels"]
    ERR = ("err", "RMSE (MCSE)", _fmt)
    COV = ("cov", "Coverage (MCSE)", lambda x: _fmt(x, 3))
    BIAS = ("bias2", r"$10^{3}\times\mathrm{Bias}^2$ (MCSE)", _fmt_scaled)
    VAR = ("var", r"$10^{3}\times\mathrm{Variance}$ (MCSE)", _fmt_scaled)
    REL = ("rel", r"RMSE relative to DLNAM", lambda x: _fmt(x, 2))
    COVCOND = ("covcond", r"Coverage, Laplace term only (MCSE)",
               lambda x: _fmt(x, 3))
    WIDTH = ("width", r"Mean interval width (log-RR)", lambda x: _fmt(x, 3))
    # Relative RMSE is defined against the DLNAM, following the convention of the
    # penalised-DLNM study, whose unit baseline is the model that study proposes.
    # Every comparator cell then reads directly as a multiple of the DLNAM error.
    for _k in ("results", "surface_results"):
        if _k in mc:
            add_relative_error(mc, "DLNAM", key=_k)
    tables = [
        metric_table(mc, r"Simulation study, model comparison: cumulative log-RR RMSE and pointwise "
                         r"95\% interval coverage by DGP, estimator, and region, with Monte Carlo "
                         r"standard errors in parentheses.",
                     "tab:supp_mc", ERR, COV, models_label=MC_LABEL),
        metric_table(mc, r"Simulation study, model comparison: squared bias and variance of the cumulative "
                         r"log-RR by DGP, estimator, and region, with Monte Carlo standard errors in "
                         r"parentheses. Both are scaled by $10^{3}$ and satisfy "
                         r"$\mathrm{RMSE}^2=\mathrm{Bias}^2+\mathrm{Variance}$ within each region before rounding.",
                     "tab:supp_mc_bv", BIAS, VAR, models_label=MC_LABEL),
        metric_table(mc, r"Simulation study, model comparison: cumulative log-RR RMSE expressed as a "
                         r"multiple of the DLNAM's, and mean width of the pointwise "
                         r"95\% interval on the log-RR scale, by DGP, estimator, and region. Values "
                         r"above 1 are worse than the DLNAM. Width is reported alongside coverage "
                         r"because coverage alone cannot separate a well-calibrated interval from an "
                         r"uninformatively wide one. Neither quantity carries a Monte Carlo standard "
                         r"error: the relative RMSE shares replicates between numerator and denominator, "
                         r"and the width is a mean over replicates rather than an estimated rate.",
                     "tab:supp_mc_rel", REL, WIDTH, models_label=MC_LABEL),
        metric_table(mc, r"DLNAM interval decomposition: empirical coverage of the reported "
                         r"interval, which combines the last-layer Laplace variance with the "
                         r"between-member spread, against coverage of the Laplace term alone. The "
                         r"difference is the contribution of the uncertainty in the learned "
                         r"representation, which conditioning on it omits. Comparator rows are "
                         r"unaffected by the decomposition and are shown for reference.",
                     "tab:supp_cov_decomp", COV, COVCOND, models_label=MC_LABEL),
        metric_table(exu, r"ExU encoder comparison: cumulative log-RR RMSE and pointwise 95\% interval "
                          r"coverage by DGP, encoder, and region.",
                     "tab:supp_exu", ERR, COV, models_label=exu_lab, unit="Encoder"),
        metric_table(exu, r"ExU encoder comparison: squared bias and variance of the cumulative log-RR by "
                          r"DGP, encoder, and region, scaled by $10^{3}$.",
                     "tab:supp_exu_bv", BIAS, VAR, models_label=exu_lab, unit="Encoder"),
        metric_table(abl, r"Architecture ablation: cumulative log-RR RMSE and pointwise 95\% interval "
                          r"coverage by DGP, configuration, and region.",
                     "tab:supp_abl", ERR, COV, models_label=abl_lab, unit="Configuration"),
        metric_table(abl, r"Architecture ablation: squared bias and variance of the cumulative log-RR by "
                          r"DGP, configuration, and region, scaled by $10^{3}$.",
                     "tab:supp_abl_bv", BIAS, VAR, models_label=abl_lab, unit="Configuration"),
        # Full exposure-lag surface: the stricter structural target.
        metric_table(mc, r"Simulation study, model comparison: full exposure-lag surface log-RR RMSE and "
                         r"pointwise 95\% interval coverage by DGP, estimator, and region, with Monte "
                         r"Carlo standard errors in parentheses. Regions refer to the exposure coordinate; "
                         r"every lag is included.",
                     "tab:supp_mc_surf", ERR, COV, models_label=MC_LABEL, key="surface_results"),
        metric_table(mc, r"Simulation study, model comparison: squared bias and variance of the full "
                         r"exposure-lag surface log-RR by DGP, estimator, and region, scaled by $10^{3}$.",
                     "tab:supp_mc_surf_bv", BIAS, VAR, models_label=MC_LABEL, key="surface_results"),
        metric_table(mc, r"Simulation study, model comparison: full exposure-lag surface log-RR RMSE "
                         r"expressed as a multiple of the DLNAM's, and mean width of "
                         r"the pointwise 95\% interval on the log-RR scale, by DGP, estimator, and "
                         r"region. Values above 1 are worse than the DLNAM.",
                     "tab:supp_mc_surf_rel", REL, WIDTH, models_label=MC_LABEL, key="surface_results"),
        metric_table(exu, r"ExU encoder comparison: full exposure-lag surface log-RR RMSE and pointwise "
                          r"95\% interval coverage by DGP, encoder, and region.",
                     "tab:supp_exu_surf", ERR, COV, models_label=exu_lab, unit="Encoder",
                     key="surface_results"),
        metric_table(abl, r"Architecture ablation: full exposure-lag surface log-RR RMSE and pointwise "
                          r"95\% interval coverage by DGP, configuration, and region.",
                     "tab:supp_abl_surf", ERR, COV, models_label=abl_lab, unit="Configuration",
                     key="surface_results"),
    ]
    # The selection-instability table is no longer reported: the point is
    # established in the literature and is cited rather than re-demonstrated.
    # selection_table remains available if it is reinstated.
    # The computational-scaling study is no longer reported, so its table is not
    # emitted; runtime_table remains available if it is reinstated.

    with open(os.path.join(odir, "supp_tables.tex"), "w") as f:
        f.write("% AUTO-GENERATED by export_results_tex.py -- do not edit by hand.\n\n"
                + "\n\n".join(tables) + "\n")
    print("wrote results_numbers.tex and supp_tables.tex")


if __name__ == "__main__":
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    project_root = sys.argv[1] if len(sys.argv) > 1 else default_root
    output_dir = (
        sys.argv[2] if len(sys.argv) > 2 else _results_dir(project_root)
    )
    main(project_root, output_dir)
