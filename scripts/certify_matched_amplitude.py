"""Two frozen matched-precision bounds; execute numerics only on Colab CPU."""
from __future__ import annotations

from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from scripts import certify_native_noise_obstruction as support

Q_VALUES = ("1/10", "3/10")
SIGMA = F(1, 20)
BETA = 400
T = F(1, 2)
K = 2
RESULT_FILENAME = "matched_amplitude_results.json"
PROTOCOL_PATH = "refine-logs/runs/vg-sae-colab-continuation-20261013/MATCHED_AMPLITUDE_PROTOCOL.md"
SOURCE_PATHS = (
    "scripts/certify_matched_amplitude.py",
    "scripts/certify_native_noise_obstruction.py",
    "tests/test_matched_amplitude.py",
    "src/sae_model.py", "src/model.py", PROTOCOL_PATH,
    "configs/colab_matched_amplitude.json",
    "configs/colab_matched_amplitude_sources.json",
)


def certify_case(q_text: str) -> dict:
    if q_text not in Q_VALUES:
        raise ValueError("Case must belong to the frozen two-case roster")
    from flint import arb
    b = support.ball
    q, sigma, t = b(q_text), b(SIGMA), b(T)
    gamma = ((1-q)/q).log()
    u0 = (b(1).exp()-1).log()
    c = t/(1+(-u0+t*sigma).exp())
    mu2 = (1/b(2).sqrt()).erf() - (2/arb.pi()).sqrt()*b(F(-1, 2)).exp()
    h = b(F(1, 2)) + sigma**2*gamma
    alpha, delta = h/sigma, (1-h)/sigma
    premises = bool(0 < h and h < 1-sigma and 0 < c and c < 1
                    and mu2 > 0 and BETA*SIGMA**2 == 1)
    if not premises:
        raise ArithmeticError("Frozen bound premises were not certified")
    def phi(z):
        return (-z*z/2).exp()/(2*arb.pi()).sqrt()
    tail0, tail1 = phi(alpha)/alpha, phi(delta)/delta
    objective_lb = K*q/2*c*(2-c)*mu2
    hard_delta_lb = c*c*sigma*sigma*mu2 - (1-q)/q*tail0
    hard_nmse_ub = (t*t*sigma*sigma + tail1 + (1-q)/q*phi(alpha)
                    *((1+sigma*sigma)/alpha+sigma*sigma*alpha))
    slope = -K*q*(1-b(-1).exp())
    checks = {
        "bound_premises": premises,
        "objective_decreases": support.endpoint(objective_lb, upper=False) > 0,
        "hard_nmse_increases": support.endpoint(hard_delta_lb, upper=False) > 0,
        "hard_nmse_scale_below_1of50": support.endpoint(hard_nmse_ub, upper=True) < F(1, 50),
    }
    values = {"gamma": gamma, "u0": u0, "c": c, "mu2": mu2,
              "threshold": h, "objective_gap_lower_bound": objective_lb,
              "hard_nmse_increase_lower_bound": hard_delta_lb,
              "hard_nmse_upper_bound": hard_nmse_ub,
              "objective_derivative_at_zero": slope,
              "inactive_selection_probability_upper_bound": tail0,
              "active_omission_probability_upper_bound": tail1}
    primary = all(checks[k] for k in ("bound_premises", "objective_decreases", "hard_nmse_increases"))
    return {"q": q_text, "status": "pair_certified" if primary else "bound_inconclusive",
            "checks": checks, "bounds": {name: support.enclosure(v) for name, v in values.items()}}


def source_hashes() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_PATHS}


def main() -> int:
    output_name = os.environ.get("COLAB_TASK_OUTPUT")
    if not output_name:
        raise SystemExit("Use the Colab CPU controller; no local output default")
    output = Path(output_name)
    support.validate_runtime(output)
    if (output/RESULT_FILENAME).exists():
        raise FileExistsError("Preserve the existing matched-amplitude result")
    budget = support.ExecutionBudget()
    initial = source_hashes()
    rows, environment = [], {}
    try:
        budget.install()
        environment = support.initialize_arithmetic()
        for q in Q_VALUES:
            budget.check()
            try:
                rows.append(certify_case(q))
            except support.BudgetExceeded:
                raise
            except Exception as e:
                rows.append({"q": q, "status": "implementation_exception", "error_type": type(e).__name__, "error": str(e)})
    except support.BudgetExceeded:
        for q in Q_VALUES[len(rows):]:
            rows.append({"q": q, "status": "budget_unresolved"})
    finally:
        budget.finalize()
    final = source_hashes()
    result = {
        "version": "matched-amplitude-v1", "cases": rows,
        "protocol": {"K": K, "sigma": str(SIGMA), "beta": BETA, "t": str(T),
                     "q_values": list(Q_VALUES), "gamma": "log((1-q)/q)",
                     "samples": 0, "quadrature": False, "training_steps": 0,
                     "precision_bits": 512, "cpu_limit_seconds": 60, "wall_limit_seconds": 120},
        "source_sha256": initial, "final_source_sha256": final,
        "sources_unchanged": initial == final,
        "environment": {"python": platform.python_version(), **environment},
        "scope": "Exact real-parameter pair; same supplied decoder and hard support. No all-recovering-class or C2 conclusion.",
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output/RESULT_FILENAME).open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    receipt = {"process_cpu_seconds": time.process_time(),
               "process_wall_seconds": time.monotonic()-support._WALL_ORIGIN,
               "cpu_limit_seconds": 60, "wall_limit_seconds": 120,
               "includes_imports_and_principal_result_serialization": True,
               "excludes_receipt_own_serialization": True}
    with (output/"runtime_receipt.json").open("x") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"cases": [{"q": r["q"], "status": r["status"]} for r in rows]}), flush=True)
    return int(initial != final or any(r["status"] != "pair_certified" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
