import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "results"
rs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(HERE / "toy_seed*.json")))]
print("seeds", [r["seed"] for r in rs])
med = lambda xs: float(np.median(xs))
out = {}
out["theta1"] = {
    k: [round(r["theta1"][k], 3) for r in rs] for k in ("task_acc", "harmful_compliance", "youden_J", "drift")
}
out["T1"] = {
    k: [round(r["T1"][k], 3) for r in rs]
    for k in ("total_gap", "amplification", "nonlinearity", "median_margin", "flipped_share")
}
out["T1"]["max_local_exponent"] = [round(max(r["T1"]["local_exponent"]), 2) for r in rs]
out["T1"]["first_local_exponent"] = [round(r["T1"]["local_exponent"][0], 2) for r in rs]
out["T1"]["exp_t_quarter_to_1"] = [
    round(float(np.log(r["T1"]["curve"][-1]["measured"] / r["T1"]["curve"][4]["measured"]) / np.log(4)), 2) for r in rs
]
for frac in ("keep_90pct_task_gain", "keep_75pct_task_gain"):
    out[frac] = {
        m: {
            "harmful_compliance": [
                r["T2_summary"][frac][m]["harmful_compliance"] if r["T2_summary"][frac][m] else None for r in rs
            ],
            "drift": [r["T2_summary"][frac][m]["drift"] if r["T2_summary"][frac][m] else None for r in rs],
            "J": [r["T2_summary"][frac][m]["youden_J"] if r["T2_summary"][frac][m] else None for r in rs],
        }
        for m in ("wise", "module_true_objective", "convex_linearised", "diag_fisher")
    }
out["T2_fit_obj_never_worse"] = all(
    o["fit_obj_final"] <= o["fit_obj_wise"] + 1e-6 for r in rs for o in r["T2"]["module_true_objective"]
)
out["T4_sweep"] = {
    str(row["rel_norm"]): {
        "coherent": [round(r["T4"]["norm_sweep"][i]["coherent"]["ratio"], 2) for r in rs],
        "random": [round(r["T4"]["norm_sweep"][i]["random"]["ratio"], 2) for r in rs],
    }
    for i, row in enumerate(rs[0]["T4"]["norm_sweep"])
}
out["T4_quant"] = {
    b: [round(r["T4"][b]["measured_over_quadratic"], 2) for r in rs]
    for b in ("quantised_8bit", "quantised_4bit", "quantised_3bit")
}
out["T4_quant_relnorm"] = {
    b: [round(r["T4"][b]["rel_norm_vs_delta"], 3) for r in rs]
    for b in ("quantised_8bit", "quantised_4bit", "quantised_3bit")
}
out["T5"] = [(round(r["T5"]["s_star_predicted"], 3), round(r["T5"]["s_star_measured"], 3)) for r in rs]
print(json.dumps(out, indent=1))
(HERE / "toy_summary.json").write_text(json.dumps(out, indent=1))
