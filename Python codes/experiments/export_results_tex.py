#!/usr/bin/env python3
"""Generate result_macros.tex and supplement_tables.tex from experiment JSONs.

Usage:  python experiments/export_results_tex.py <dlnam_dir> <out_dir>

Every number quoted in the manuscript comes from here; no digits are typed by hand.
Re-run after the R=200 job and recompile -- prose is untouched.
"""
import json, sys, os

SCEN = {"smooth": "Smooth", "delayed_peaks": "Delayed",
        "localized_peak": "Localized", "tilting_threshold": "Tilting"}
MODEL = {"DLNAM": "Dlnam", "QAIC": "Qaic", "QBIC": "Qbic", "Penalised": "Pen", "TDLNM": "Tdlnm",
         "concat": "Concat", "unified_shared_bias": "Shared", "unified_local_bias": "Local",
         "reference": "Ref", "no_exu": "NoExu", "no_subnets": "NoSub", "no_smooth": "NoSmooth"}
REG = {"tot": "Tot", "int": "Int", "bnd": "Bnd"}
SCEN_LABEL = {"smooth": "Smooth", "delayed_peaks": "Delayed Peaks",
              "localized_peak": "Localized Peak", "tilting_threshold": "Tilting Threshold"}
# Display names must match the figure legends exactly.
MC_LABEL = {"DLNAM": "DLNAM", "QAIC": "DLNM (QAIC)", "QBIC": "DLNM (QBIC)",
            "Penalised": "P-DLNM", "TDLNM": "T-DLNM"}


def num(x, dp):
    return f"{x:.{dp}f}"


def emit_macros(d, prefix, out):
    """RMSE/coverage/bias/var macros for one experiment."""
    for s in d["scenarios"]:
        for m in d["models"]:
            v = d["results"][s][m]
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


def derived_macros(mc, exu, abl, out):
    """Ranges across scenarios: the only quantities quoted in the Results prose."""
    def rng(d, num_model, den_model, key="err_tot", tag=""):
        vals = [d["results"][s][num_model][key] / d["results"][s][den_model][key] for s in d["scenarios"]]
        out.append(rf"\newcommand{{\{tag}Min}}{{{min(vals):.1f}}}")
        out.append(rf"\newcommand{{\{tag}Max}}{{{max(vals):.1f}}}")

    # DLNAM advantage over each comparator (total RMSE), min/max across scenarios
    for m, t in [("QAIC", "Qaic"), ("QBIC", "Qbic"), ("Penalised", "Pen"), ("TDLNM", "Tdlnm")]:
        rng(mc, m, "DLNAM", tag="McRatio" + t)
    # spread of DLNAM error across scenarios (max/min)
    e = [mc["results"][s]["DLNAM"]["err_tot"] for s in mc["scenarios"]]
    out.append(rf"\newcommand{{\McDlnamSpread}}{{{max(e)/min(e):.1f}}}")
    # boundary/interior degradation, min/max across scenarios
    for m, t in [("DLNAM", "Dlnam"), ("QAIC", "Qaic"), ("QBIC", "Qbic"), ("Penalised", "Pen"), ("TDLNM", "Tdlnm")]:
        v = [mc["results"][s][m]["err_bnd"] / mc["results"][s][m]["err_int"] for s in mc["scenarios"]]
        out.append(rf"\newcommand{{\McDeg{t}Min}}{{{min(v):.1f}}}")
        out.append(rf"\newcommand{{\McDeg{t}Max}}{{{max(v):.1f}}}")
    # coverage ranges as integer percentages
    for m, t in [("DLNAM", "Dlnam"), ("QAIC", "Qaic"), ("QBIC", "Qbic"), ("Penalised", "Pen"), ("TDLNM", "Tdlnm")]:
        c = [mc["results"][s][m]["cov_tot"] for s in mc["scenarios"]]
        out.append(rf"\newcommand{{\McCov{t}PctMin}}{{{100*min(c):.0f}}}")
        out.append(rf"\newcommand{{\McCov{t}PctMax}}{{{100*max(c):.0f}}}")
    mcse = max(mc["results"][s][m]["cov_tot_se"] for s in mc["scenarios"] for m in mc["models"])
    out.append(rf"\newcommand{{\McCovMcsePctMax}}{{{100*mcse:.0f}}}")
    # ablation and ExU degradation ranges
    for m, t in [("no_exu", "NoExu"), ("no_subnets", "NoSub"), ("no_smooth", "NoSmooth")]:
        rng(abl, m, "reference", tag="AblRatio" + t)
        c = [abl["results"][s][m]["cov_tot"] for s in abl["scenarios"]]
        out.append(rf"\newcommand{{\AblCov{t}PctMin}}{{{100*min(c):.0f}}}")
    for m, t in [("unified_shared_bias", "Shared"), ("unified_local_bias", "Local")]:
        rng(exu, m, "concat", tag="ExuRatio" + t)
        c = [exu["results"][s][m]["cov_tot"] for s in exu["scenarios"]]
        out.append(rf"\newcommand{{\ExuCov{t}PctMin}}{{{100*min(c):.0f}}}")
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


def metric_table(d, caption, label, m1, m2, models_label=None, unit="Model"):
    """Scenario x Model, two metrics each decomposed over total/interior/boundary,
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
            rf"Scenario & {unit} & Total & Interior & Boundary & Total & Interior & Boundary\\",
            r"\midrule"]
    for s_ in d["scenarios"]:
        for i, m in enumerate(d["models"]):
            v = d["results"][s_][m]
            name = SCEN_LABEL[s_] if i == 0 else ""
            cells = []
            for key, fmt in ((k1, f1), (k2, f2)):
                for r in ("tot", "int", "bnd"):
                    cells.append(rf"{fmt(v[f'{key}_{r}'])} ({fmt(v[f'{key}_{r}_se'])})")
            rows.append(f"{name} & {L.get(m, m)} & " + " & ".join(cells) + r"\\")
        rows.append(r"\addlinespace")
    rows += [r"\bottomrule", r"\end{tabular}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{sidewaystable}"]
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
    mc = json.load(open(os.path.join(jdir, "mc_model_comparison.json")))
    exu = json.load(open(os.path.join(jdir, "mc_exu.json")))
    abl = json.load(open(os.path.join(jdir, "mc_ablation.json")))
    chi = json.load(open(os.path.join(jdir, "chicago_model_comparison.json")))

    out = ["% AUTO-GENERATED by export_results_tex.py -- do not edit by hand.",
           "% Safe to \\input more than once (e.g. from the preamble and from results.tex).",
           "\\ifdefined\\resultsnumbersloaded\\endinput\\fi",
           "\\newcommand{\\resultsnumbersloaded}{}",
           ""]
    emit_macros(mc, "Mc", out); ratio_macros(mc, out)
    emit_macros(exu, "Exu", out)
    emit_macros(abl, "Abl", out)
    derived_macros(mc, exu, abl, out)

    st = mc["settings"]
    out += ["",
            rf"\newcommand{{\NReps}}{{{st['n_reps']}}}",
            rf"\newcommand{{\NObs}}{{{st['n_obs']}}}",
            rf"\newcommand{{\NEpochs}}{{{st['epochs']}}}",
            rf"\newcommand{{\NEnsemble}}{{{st['n_ensemble']}}}",
            rf"\newcommand{{\LagMax}}{{{st['lag']}}}",
            rf"\newcommand{{\XRef}}{{{st['reference']:.0f}}}",
            rf"\newcommand{{\DfGridLo}}{{{min(st['value_df_grid'])}}}",
            rf"\newcommand{{\DfGridHi}}{{{max(st['value_df_grid'])}}}"]

    bnd = mc["boundary"]["smooth"]
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

    with open(os.path.join(odir, "result_macros.tex"), "w") as f:
        f.write("\n".join(out) + "\n")

    exu_lab = exu["settings"]["labels"]; abl_lab = abl["settings"]["labels"]
    ERR = ("err", "RMSE (MCSE)", _fmt)
    COV = ("cov", "Coverage (MCSE)", lambda x: _fmt(x, 3))
    BIAS = ("bias2", r"$10^{3}\times\mathrm{Bias}^2$ (MCSE)", _fmt_scaled)
    VAR = ("var", r"$10^{3}\times\mathrm{Variance}$ (MCSE)", _fmt_scaled)
    tables = [
        metric_table(mc, r"Simulation study, model comparison: cumulative log-RR RMSE and pointwise "
                         r"95\% interval coverage by scenario, estimator, and region, with Monte Carlo "
                         r"standard errors in parentheses.",
                     "tab:supp_mc", ERR, COV, models_label=MC_LABEL),
        metric_table(mc, r"Simulation study, model comparison: squared bias and variance of the cumulative "
                         r"log-RR by scenario, estimator, and region, with Monte Carlo standard errors in "
                         r"parentheses. Both are scaled by $10^{3}$ and satisfy "
                         r"$\mathrm{RMSE}^2=\mathrm{Bias}^2+\mathrm{Variance}$ within each region before rounding.",
                     "tab:supp_mc_bv", BIAS, VAR, models_label=MC_LABEL),
        metric_table(exu, r"ExU encoder comparison: cumulative log-RR RMSE and pointwise 95\% interval "
                          r"coverage by scenario, encoder, and region.",
                     "tab:supp_exu", ERR, COV, models_label=exu_lab, unit="Encoder"),
        metric_table(exu, r"ExU encoder comparison: squared bias and variance of the cumulative log-RR by "
                          r"scenario, encoder, and region, scaled by $10^{3}$.",
                     "tab:supp_exu_bv", BIAS, VAR, models_label=exu_lab, unit="Encoder"),
        metric_table(abl, r"Architecture ablation: cumulative log-RR RMSE and pointwise 95\% interval "
                          r"coverage by scenario, configuration, and region.",
                     "tab:supp_abl", ERR, COV, models_label=abl_lab, unit="Configuration"),
        metric_table(abl, r"Architecture ablation: squared bias and variance of the cumulative log-RR by "
                          r"scenario, configuration, and region, scaled by $10^{3}$.",
                     "tab:supp_abl_bv", BIAS, VAR, models_label=abl_lab, unit="Configuration"),
    ]
    with open(os.path.join(odir, "supplement_tables.tex"), "w") as f:
        f.write("% AUTO-GENERATED by export_results_tex.py -- do not edit by hand.\n\n"
                + "\n\n".join(tables) + "\n")
    print("wrote result_macros.tex and supplement_tables.tex")


if __name__ == "__main__":
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    main(sys.argv[1] if len(sys.argv) > 1 else default_root,
         sys.argv[2] if len(sys.argv) > 2 else ".")
