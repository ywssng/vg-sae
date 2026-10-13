"""Certify a fixed native-VG objective versus clean-risk obstruction on Colab CPU.

The twelve cases use exact rational scientific inputs and official python-flint
Arb at 512 bits. No samples, quadrature, optimization, training or Torch are
used by this runner. Model parameters denote the real-valued native architecture;
the certificates do not silently replace them by rounded float64 parameters.
"""
from __future__ import annotations

import time

_MODULE_START = time.monotonic()

import csv
from dataclasses import asdict, dataclass
from fractions import Fraction as F
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import signal


VERSION = "native-noise-clean-risk-obstruction-v1"
PRECISION_BITS = 512
CPU_LIMIT_SECONDS = 60
WALL_LIMIT_SECONDS = 120
FINALIZATION_SECONDS = 5
RESULT_FILENAME = "native_noise_obstruction_results.json"
REPO_ROOT = Path(__file__).resolve().parents[1]

DIMENSION = 2
SIGMA = F(1, 20)
EPSILON = F(DIMENSION) * SIGMA**2 / 100
CENTER_SHIFT = F(10)
GATE_LOGIT = F(32)
DECODER = ((F(3, 5), F(-4, 5)), (F(4, 5), F(3, 5)))
Q_VALUES = ("1/10", "3/10")
BETA_VALUES = ("400", "1600", "25600")
GAMMA_VALUES = ("1", "16")

_arb = None
_fmpq = None
_ctx = None


def _process_wall_origin() -> tuple[float, str]:
    try:
        tail = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        start_ticks = int(tail[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        age = max(0.0, uptime - start_ticks / os.sysconf("SC_CLK_TCK"))
        return time.monotonic() - age - 0.05, "linux_process_age_with_0.05s_guard"
    except (OSError, ValueError, IndexError):
        return _MODULE_START, "module_entry_before_flint_import"


_WALL_ORIGIN, _WALL_SCOPE = _process_wall_origin()


@dataclass(frozen=True)
class Case:
    q: str
    beta: str
    gamma: str

    @property
    def case_id(self) -> str:
        return f"q{self.q.replace('/', 'of')}_beta{self.beta}_gamma{self.gamma}"


def frozen_cases() -> tuple[Case, ...]:
    return tuple(Case(*values) for values in itertools.product(
        Q_VALUES, BETA_VALUES, GAMMA_VALUES
    ))


def exact(value) -> F:
    if isinstance(value, (float, bool)):
        raise ValueError("Scientific inputs must be exact integers or rational strings")
    return F(value)


def initialize_arithmetic() -> dict:
    global _arb, _fmpq, _ctx
    import flint
    from flint import arb, ctx, fmpq
    ctx.prec = PRECISION_BITS
    ctx.threads = 1
    _arb, _fmpq, _ctx = arb, fmpq, ctx
    return {"python_flint": flint.__version__, "arb_precision_bits": ctx.prec,
            "flint_threads": ctx.threads, "arithmetic": "official_python_flint_arb"}


def ball(value):
    if _arb is None:
        raise RuntimeError("Initialize official Arb arithmetic first")
    value = exact(value)
    return _arb(_fmpq(value.numerator, value.denominator))


def endpoint(value, *, upper: bool) -> F:
    out = value.upper() if upper else value.lower()
    if not out.is_finite() or not out.is_exact():
        raise ArithmeticError("A finite exact outward endpoint is required")
    return F(str(out.fmpq()))


def enclosure(value) -> dict:
    if not value.is_finite():
        raise ArithmeticError("A nonfinite interval is not a certificate")
    lo, hi = endpoint(value, upper=False), endpoint(value, upper=True)
    if lo > hi:
        raise ArithmeticError("Reversed interval")
    return {"lower": str(lo), "upper": str(hi),
            "display_only": value.str(24), "endpoints_are_exact_rationals": True}


def orthogonal_decoder_check() -> bool:
    return all(sum(DECODER[k][i] * DECODER[k][j] for k in range(DIMENSION))
               == int(i == j) for i in range(DIMENSION) for j in range(DIMENSION))


def dense_second_moment(q) -> F:
    q = exact(q)
    if not 0 < q < 1:
        raise ValueError("Require an interior Bernoulli activity")
    return (DIMENSION * q + 2 * CENTER_SHIFT * q * sum(map(sum, DECODER))
            + DIMENSION * CENTER_SHIFT**2 + DIMENSION * SIGMA**2)


def softplus_residual_second_moment_bound(q):
    """Bound sum_j E softplus(-t_j)^2 by exact Gaussian exponential moments.

    t_j = c + sum_i D_ij S_i + sigma*Z_j, so the product is over each
    decoder column, not its rows. All source and noise expectations are analytic.
    """
    q = exact(q)
    if not 0 < q < 1:
        raise ValueError("Require an interior Bernoulli activity")
    columns = []
    for j in range(DIMENSION):
        value = ball(1)
        for i in range(DIMENSION):
            value *= ball(1 - q) + ball(q) * ball(-2 * DECODER[i][j]).exp()
        columns.append(value)
    gaussian_and_shift = ball(-2 * CENTER_SHIFT + 2 * SIGMA**2).exp()
    value = gaussian_and_shift * sum(columns, ball(0))
    if not value > 0:
        raise ArithmeticError("Positive finite exponential moment was not certified")
    return value, columns, gaussian_and_shift


def clean_class_lower_bound(beta) -> F:
    beta = exact(beta)
    if beta <= 0:
        raise ValueError("Precision must be positive")
    # epsilon=d*sigma^2/100, hence the reverse-triangle factor is (1-1/10)^2.
    return beta * F(DIMENSION, 2) * SIGMA**2 * F(81, 100)


def strict_separation(upper, lower) -> bool:
    """Equality and overlapping bounds never receive a strict certificate."""
    return exact(upper) < exact(lower)


def dense_model_parameters(case: Case) -> dict:
    bias = [-CENTER_SHIFT * sum(row) for row in DECODER]
    transpose = [[DECODER[j][i] for j in range(DIMENSION)] for i in range(DIMENSION)]
    return {
        "arithmetic_contract": "exact_real_native_architecture",
        "config": {"input_dim": 2, "n_latents": 2, "beta": case.beta,
                   "lambda_sparsity": case.gamma, "beta_mode": "learned",
                   "frozen_precision": True, "use_variance_term": True,
                   "use_entropy_term": True, "entropy_weight": 1,
                   "decoder_bias": True, "normalize_decoder": True,
                   "nonnegative_amplitudes": True, "inference_threshold": "1/2"},
        "parameters": {
            "decoder.weight": [[str(v) for v in row] for row in DECODER],
            "pre_bias": list(map(str, bias)),
            "amplitude_encoder.weight": [[str(v) for v in row] for row in transpose],
            "amplitude_encoder.bias": ["0", "0"],
            "gate_encoder.weight": [["0", "0"], ["0", "0"]],
            "gate_encoder.bias": [str(GATE_LOGIT)] * DIMENSION,
            "log_beta": {"expression": "log(beta)", "beta": case.beta},
        },
        "native_hard_density_exact": "1",
        "fitted_data_or_teacher_labels_used": False,
    }


def good_class_witness(case: Case) -> tuple[dict, bool]:
    A = 1 / (2 * SIGMA**2)
    exponent = -A / 2 + A**2 * SIGMA**2 / 2
    risk_upper_expression = ball(DIMENSION) * ball(exponent).exp()
    risk_upper = endpoint(risk_upper_expression, upper=True)
    return {
        "arithmetic_contract": "exact_real_native_architecture",
        "config": dense_model_parameters(case)["config"],
        "parameters": {
            "decoder.weight": [["1", "0"], ["0", "1"]],
            "pre_bias": ["0", "0"],
            "amplitude_encoder.weight": [["0", "0"], ["0", "0"]],
            "amplitude_encoder.bias": [{"expression": "log(exp(1)-1)"}] * 2,
            "gate_encoder.weight": [[str(A), "0"], ["0", str(A)]],
            "gate_encoder.bias": [str(-A / 2)] * 2,
            "log_beta": {"expression": "log(beta)", "beta": case.beta},
        },
        "clean_risk_definition": "E ||S - reconstruction(X,B)||^2, without a factor of one half",
        "clean_risk_upper_formula": "d*exp(-A/2 + A^2*sigma^2/2)",
        "exponent_exact": str(exponent),
        "clean_risk_upper_expression": enclosure(risk_upper_expression),
        "epsilon_exact": str(EPSILON),
        "nonempty_good_class_certified": risk_upper < EPSILON,
        "uses_X_only": True,
    }, risk_upper < EPSILON


def certify_case(case: Case) -> dict:
    if case not in frozen_cases():
        raise ValueError("Case is outside the frozen twelve-case panel")
    if _ctx is None or _ctx.prec != PRECISION_BITS or _ctx.threads != 1:
        raise RuntimeError("Official Arb must use the frozen 512-bit, one-thread context")
    if not orthogonal_decoder_check():
        raise ArithmeticError("The specified exact-rational decoder must be orthogonal")
    q, beta, gamma = map(exact, (case.q, case.beta, case.gamma))
    tail = ball(-GATE_LOGIT).exp()
    delta = tail / (1 + tail)
    gate = 1 / (1 + tail)
    t2 = dense_second_moment(q)
    l2_bound, columns, gaussian_shift = softplus_residual_second_moment_bound(q)
    observed_risk_upper_expression = delta * ball(t2) + gate * l2_bound
    prior_upper_expression = ball(DIMENSION) * (ball(gamma) + ball(-gamma).exp().log1p())
    dense_upper_expression = ball(beta / 2) * observed_risk_upper_expression + prior_upper_expression
    dense_upper = endpoint(dense_upper_expression, upper=True)
    clean_lower = clean_class_lower_bound(beta)
    lower_formula = ball(beta / 2) * (
        ball(SIGMA) * ball(DIMENSION).sqrt() - ball(EPSILON).sqrt()
    ) ** 2
    lower_identity = (endpoint(lower_formula, upper=False) <= clean_lower
                      <= endpoint(lower_formula, upper=True))
    good, nonempty = good_class_witness(case)
    if not lower_identity:
        raise ArithmeticError("Reverse-triangle and exact rational lower bounds disagree")
    margin = clean_lower - dense_upper
    certified = nonempty and strict_separation(dense_upper, clean_lower)
    return {
        "case_id": case.case_id, **asdict(case),
        "status": "obstruction_certified" if certified else "bound_inconclusive",
        "sigma_exact": str(SIGMA), "epsilon_exact": str(EPSILON),
        "beta_over_true_precision_exact": str(beta * SIGMA**2),
        "dense_model": dense_model_parameters(case),
        "good_class_witness": good,
        "bounds": {
            "T2_exact": str(t2),
            "delta": enclosure(delta), "constant_gate": enclosure(gate),
            "exponential_column_terms": [enclosure(value) for value in columns],
            "gaussian_and_shift_factor": enclosure(gaussian_shift),
            "softplus_residual_second_moment_upper_expression": enclosure(l2_bound),
            "dense_observed_squared_risk_upper_expression": enclosure(observed_risk_upper_expression),
            "dense_prior_KL_upper_expression": enclosure(prior_upper_expression),
            "dense_objective_upper_expression": enclosure(dense_upper_expression),
            "dense_objective_certified_upper": str(dense_upper),
            "clean_class_objective_lower_exact": str(clean_lower),
            "clean_class_lower_formula_interval": enclosure(lower_formula),
            "objective_gap_certified_lower": str(margin),
            "objective_gap_display_only": ball(margin).str(24),
        },
        "checks": {
            "exact_decoder_orthogonality": True,
            "exact_lower_bound_identity": lower_identity,
            "nonempty_good_class": nonempty,
            "strict_upper_below_lower": certified,
            "exact_input_fractions": True, "outward_endpoint_comparison": True,
        },
        "interpretation": (
            "All native models with clean stochastic risk at most epsilon have higher population objective than this finite native witness; no optimizer attainment is assumed."
            if certified else
            "This fixed sufficient bound does not certify an obstruction; absence of obstruction or correctness of the objective is not established."
        ),
        "scope": "Fixed-beta normalized native conditional objective; not a dictionary-recovery gap, unknown-D consistency or a density-estimation result.",
    }


def protocol() -> dict:
    return {
        "version": VERSION, "case_count": len(frozen_cases()),
        "cases": [asdict(case) for case in frozen_cases()],
        "d": DIMENSION, "K": DIMENSION, "q_values": list(Q_VALUES),
        "beta_values": list(BETA_VALUES), "gamma_values": list(GAMMA_VALUES),
        "sigma": str(SIGMA), "epsilon": str(EPSILON),
        "decoder": [[str(v) for v in row] for row in DECODER],
        "c": str(CENTER_SHIFT), "L": str(GATE_LOGIT),
        "arithmetic": "official_python_flint_Arb", "precision_bits": PRECISION_BITS,
        "threads": 1, "samples": 0, "quadrature": False,
        "optimizer_steps": 0, "training_steps": 0, "seeds": [],
        "adaptive_choices": False, "torch_used_by_runner": False,
        "required_runtime_flag": "VG_SAE_COLAB_CPU_ONLY=1",
        "total_process_cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "total_process_wall_limit_seconds": WALL_LIMIT_SECONDS,
        "finalization_reserve_seconds": FINALIZATION_SECONDS,
        "overwrite_outputs": False,
        "success_rule": "nonempty good class and strict dense objective upper < clean-class lower",
        "failed_bound_status": "bound_inconclusive",
    }


class BudgetExceeded(RuntimeError):
    pass


def validate_runtime(output: Path, *, environ=None, gpu_visible=None) -> None:
    environment = os.environ if environ is None else environ
    if environment.get("VG_SAE_COLAB_CPU_ONLY") != "1":
        raise RuntimeError("VG_SAE_COLAB_CPU_ONLY must be exactly '1'")
    if gpu_visible is None:
        gpu_flag = environment.get("COLAB_GPU", "").strip().lower()
        gpu_visible = (gpu_flag not in {"", "0", "false"}
                       or any(Path("/dev").glob("nvidia[0-9]*")))
    if gpu_visible:
        raise RuntimeError("This task requires the verified Colab CPU-only runtime")
    if (output / RESULT_FILENAME).exists():
        raise FileExistsError(f"Preserve existing attempt: {RESULT_FILENAME} already exists")


class ExecutionBudget:
    """Entrypoint-only limits: total process usage includes all imports."""

    def __init__(self):
        inherited = [int(value) for value in resource.getrlimit(resource.RLIMIT_CPU)
                     if value != resource.RLIM_INFINITY]
        self.total_cpu = min([CPU_LIMIT_SECONDS, *inherited])
        self.science_cpu = max(0, self.total_cpu - FINALIZATION_SECONDS)
        self.science_wall = WALL_LIMIT_SECONDS - FINALIZATION_SECONDS
        self.installed = False

    @staticmethod
    def _science_timeout(_signum, _frame):
        raise BudgetExceeded("Scientific CPU/wall budget reached")

    @staticmethod
    def _hard_wall_timeout(_signum, _frame):
        raise SystemExit(124)

    def install(self):
        signal.signal(signal.SIGXCPU, self._science_timeout)
        signal.signal(signal.SIGALRM, self._science_timeout)
        resource.setrlimit(resource.RLIMIT_CPU, (self.science_cpu, self.total_cpu))
        self.installed = True
        remaining = self.science_wall - (time.monotonic() - _WALL_ORIGIN)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))

    def check(self):
        if (time.process_time() >= self.science_cpu
                or time.monotonic() - _WALL_ORIGIN >= self.science_wall):
            raise BudgetExceeded("Scientific CPU/wall budget reached")

    def finalize(self):
        if self.installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            resource.setrlimit(resource.RLIMIT_CPU, (self.total_cpu, self.total_cpu))
            signal.signal(signal.SIGALRM, self._hard_wall_timeout)
            remaining = WALL_LIMIT_SECONDS - (time.monotonic() - _WALL_ORIGIN)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))


def retain_budget_remainder(rows: list[dict], reason: str, interrupted: bool) -> None:
    for index, case in enumerate(frozen_cases()[len(rows):]):
        rows.append({"case_id": case.case_id, **asdict(case),
                     "status": ("budget_unresolved_interrupted" if interrupted and index == 0
                                else "budget_unresolved_not_started"),
                     "budget_reason": reason, "numeric_result_available": False})


def source_hashes() -> dict:
    paths = [
        "scripts/certify_native_noise_obstruction.py",
        "tests/test_native_noise_obstruction.py", "src/sae_model.py", "src/model.py",
        "refine-logs/runs/vg-sae-colab-continuation-20261003/NATIVE_NOISE_OBSTRUCTION_PROTOCOL.md",
    ]
    return {path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() for path in paths}


def write_outputs(output: Path, result: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / RESULT_FILENAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    columns = ["case_id", "q", "beta", "gamma", "status",
               "beta_over_true_precision_exact", "dense_objective_upper",
               "clean_class_objective_lower", "gap_lower", "nonempty_good_class"]
    with (output / "native_noise_obstruction_summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in result["cases"]:
            summary = {key: row.get(key, "") for key in columns}
            if "bounds" in row:
                summary.update(
                    dense_objective_upper=row["bounds"]["dense_objective_certified_upper"],
                    clean_class_objective_lower=row["bounds"]["clean_class_objective_lower_exact"],
                    gap_lower=row["bounds"]["objective_gap_certified_lower"],
                    nonempty_good_class=row["checks"]["nonempty_good_class"],
                )
            writer.writerow(summary)
    counts = result["status_counts"]
    report = [
        "# Native population objective versus clean stochastic risk", "",
        "All twelve frozen cases are retained. Exact rational inputs and 512-bit Arb",
        "outward arithmetic evaluate analytic bounds; no observations, quadrature,",
        "fitting, optimizer updates or GPU computation are used.", "",
        f"- Obstruction certified: {counts.get('obstruction_certified', 0)}",
        f"- Sufficient bound inconclusive: {counts.get('bound_inconclusive', 0)}",
        f"- Budget-unresolved: {sum(v for k, v in counts.items() if k.startswith('budget_'))}",
        f"- Implementation errors: {sum(v for k, v in counts.items() if k.startswith('implementation_'))}",
        "",
        "A certificate separates one explicit finite native witness from the entire",
        "class with clean stochastic squared risk at most epsilon. It does not",
        "exclude accurate dictionaries, prove practical optimization, or establish",
        "unknown-D consistency or generating-density estimation. Inconclusive",
        "matched-precision cases remain in the denominator.", "",
        "Certificates concern the real-valued architecture at exact rational",
        "weights and stated analytic log/softplus parameters. They do not certify",
        "a rounded float64 implementation. Full bounds, exact endpoints, model",
        "parameters and source hashes are in native_noise_obstruction_results.json.", "",
    ]
    with (output / "native_noise_obstruction_report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(report))


def main() -> int:
    output_name = os.environ.get("COLAB_TASK_OUTPUT")
    if not output_name:
        raise SystemExit("Run via Colab with COLAB_TASK_OUTPUT; no local output default")
    output = Path(output_name)
    validate_runtime(output)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    budget = ExecutionBudget()
    frozen = protocol()
    initial_sources, environment = {}, {}
    rows, budget_reason = [], None
    current_case_id = None
    try:
        budget.install()
        budget.check()
        initial_sources = source_hashes()
        environment = initialize_arithmetic()
        for case in frozen_cases():
            current_case_id = None
            budget.check()
            current_case_id = case.case_id
            try:
                row = certify_case(case)
            except BudgetExceeded:
                raise
            except Exception as error:
                row = {"case_id": case.case_id, **asdict(case),
                       "status": "implementation_exception",
                       "error_type": type(error).__name__, "error": str(error)}
            rows.append(row)
            current_case_id = None
    except BudgetExceeded as error:
        budget_reason = str(error)
        interrupted = bool(current_case_id and (not rows or rows[-1]["case_id"] != current_case_id))
        retain_budget_remainder(rows, budget_reason, interrupted)
    finally:
        budget.finalize()
    final_sources = source_hashes()
    unchanged = final_sources == initial_sources if initial_sources else None
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    result = {
        "execution_status": "completed_with_budget_unresolved" if budget_reason else "completed",
        "protocol": frozen,
        "protocol_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_sha256": initial_sources, "final_source_sha256": final_sources,
        "sources_unchanged": unchanged, "initial_source_seal_complete": bool(initial_sources),
        "cases": rows, "status_counts": counts, "budget_reason": budget_reason,
        "environment": {"python": platform.python_version(), "platform": platform.system(), **environment},
        "resources_before_output_serialization": {
            "process_cpu_seconds": time.process_time(),
            "process_wall_seconds": time.monotonic() - _WALL_ORIGIN,
            "cpu_scope": "whole_process_including_imports", "wall_scope": _WALL_SCOPE,
            "cpu_limit_seconds": budget.total_cpu, "wall_limit_seconds": WALL_LIMIT_SECONDS,
        },
    }
    write_outputs(output, result)
    receipt = {
        "process_cpu_seconds": time.process_time(),
        "process_wall_seconds": time.monotonic() - _WALL_ORIGIN,
        "cpu_scope": "whole_process_including_imports", "wall_scope": _WALL_SCOPE,
        "cpu_limit_seconds": budget.total_cpu, "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "includes_principal_output_serialization": True,
        "excludes_receipt_own_serialization": True,
        "sources_unchanged": unchanged,
        "execution_status": result["execution_status"],
    }
    with (output / "native_noise_obstruction_runtime_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"cases": len(rows), "status_counts": counts}, sort_keys=True), flush=True)
    return int(unchanged is False or any(row["status"].startswith("implementation_") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
