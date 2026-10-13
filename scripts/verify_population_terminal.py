#!/usr/bin/env python3
"""Independently verify the fixed population panel's terminal partition.

This program never imports the primary engine, performs a new optimization, or
changes the historical run. Run it on Colab CPU. Only the official python-flint
Arb elementary primitives are shared with the original numerical implementation.
The fixed source and result seals bind archival provenance; every consumed
result/trace is hashed. Unused historical dependencies are not re-executed or
claimed rehashed. Output is a new certificate with all 20 rows, including any
unresolved suffix after a resource stop.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time


SOURCE_MANIFEST_SHA256 = "f042e708377505e907b9ddc183dfb0a1b9bb15050b9ccc7e1e5f26f7a17967b8"
RUN_SEAL_SHA256 = "e14c41e4cba0dacd63607550d51eec53f5edd117224af1f797568f854beb4fcb"
VERSION = "terminal-partition-v1"
PRECISION = 512
_arb = None
_fmpq = None


class CertificateError(ValueError):
    """An exact identity, provenance check, or cover invariant failed."""


class ArithmeticUnresolved(ArithmeticError):
    """The independent arithmetic did not prove a requested comparison."""


def require(condition, message):
    if not condition:
        raise CertificateError(message)


def exact(value):
    if isinstance(value, (bool, float)):
        raise CertificateError("non-exact rational input")
    return F(value)


def pair(values):
    require(isinstance(values, (list, tuple)) and len(values) == 2, "invalid interval")
    lo, hi = map(exact, values)
    require(lo <= hi, "reversed interval")
    return lo, hi


def digest(path, tick=lambda: None):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            tick()
            h.update(block)
    return h.hexdigest()


def certificate_digest(record):
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_summary_row(row, cell_result):
    # The frozen parent adds these after the worker has saved its result.
    # Every original field, including scientific values and worker receipts,
    # must still agree exactly. No arbitrary metadata is silently ignored.
    extras = set(row) - set(cell_result)
    require(extras <= {"parent_all_in_cell_cpu_seconds", "exit_code"},
            "unexpected summary-only fields")
    require(all(key in row and row[key] == value for key, value in cell_result.items()),
            "summary/cell result mismatch")
    if "parent_all_in_cell_cpu_seconds" in extras:
        value = row["parent_all_in_cell_cpu_seconds"]
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "invalid parent CPU receipt")
    if "exit_code" in extras:
        require(type(row["exit_code"]) is int, "invalid parent exit code")


def write_new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def expected_panel():
    return [
        {"q0": q, "d": d, "lambda": lam, "I": ["1/100", "2/5"],
         "id": f"q{q.replace('/', '_')}_d{d}_l{lam.replace('/', '_')}"}
        for q in ("1/10", "3/10")
        for d in (2, 10)
        for lam in ("0", "1/2", "2", "8", "16")
    ]


@dataclass
class TerminalPartition:
    root: tuple[F, F]
    active: list[dict]
    rejected: list[dict]
    witness_candidates: list[F]
    incumbent: F
    events: int
    splits: int
    trace_sha256: str


def check_exact_cover(root, certificates):
    """Closed intervals may meet at endpoints, but have no gap or overlap."""
    ordered = sorted(certificates, key=lambda c: exact(c["domain"][0]))
    cursor = root[0]
    for cert in ordered:
        lo, hi = pair(cert["domain"])
        require(lo == cursor and hi > lo, "terminal partition gap or overlap")
        cursor = hi
    require(bool(ordered) and cursor == root[1], "incomplete terminal root cover")
    return ordered


def parse_partition(path, row, expected_hash, tick=lambda: None):
    """Exact geometry and full-field binding; no transcendental arithmetic."""
    active = {}
    seen = set()
    rejected = []
    final = None
    root = None
    incumbent = None
    best_witness = None
    best_coordinate = None
    events = splits = 0
    h = hashlib.sha256()

    def register(cert, refresh=False):
        nonlocal best_witness, best_coordinate
        ident = cert["id"]
        require(type(ident) is int and ident >= 0, "invalid leaf ID")
        domain = pair(cert["domain"])
        require(domain[0] < domain[1], "zero-width leaf")
        midpoint = (domain[0] + domain[1]) / 2
        require(exact(cert["witness_c"]) == midpoint, "non-midpoint recorded witness")
        if refresh:
            require(ident in active and domain == active[ident][0], "refresh changed cover")
        else:
            require(ident not in seen, "reused tree ID")
            seen.add(ident)
        active[ident] = (domain, certificate_digest(cert))
        lower = exact(cert["witness_lower"])
        if best_witness is None or lower > best_witness:
            best_witness, best_coordinate = lower, midpoint
        return lower

    with Path(path).open("rb") as stream:
        for raw in stream:
            tick()
            require(final is None, "event after final remaining event")
            h.update(raw)
            event = json.loads(raw)
            events += 1
            kind = event["type"]
            if kind == "input":
                require(events == 1 and not active, "duplicate input event")
                require(event["spec"] == row["spec"], "trace specification mismatch")
                root = pair(event["root"]["domain"])
                require(root[0] == exact(row["spec"]["lambda"]) / 2, "wrong root start")
                require(root[1] == exact(event["tail_upper"]) == exact(row["tail_upper"]),
                        "tail endpoint mismatch")
                require(event["tail_bound"] == row["tail_bound"], "tail bound mismatch")
                lower = register(event["root"])
                incumbent = exact(event["incumbent_lower"])
                require(incumbent >= lower, "input incumbent below its midpoint witness")
            elif kind == "split":
                require(root is not None, "split before root")
                parent = event["parent"]
                ident = parent["id"]
                require(ident in active, "split of inactive leaf")
                domain = pair(parent["domain"])
                require(domain == active[ident][0], "parent domain changed")
                middle = (domain[0] + domain[1]) / 2
                children = event["children"]
                require(len(children) == 2, "nonbinary split")
                require(pair(children[0]["domain"]) == (domain[0], middle)
                        and pair(children[1]["domain"]) == (middle, domain[1]),
                        "children do not exactly cover parent")
                child_lowers = [register(child) for child in children]
                del active[ident]
                incumbent = max(incumbent, *child_lowers)
                require(exact(event["incumbent_lower"]) == incumbent, "split incumbent mismatch")
                splits += 1
            elif kind == "reject":
                cert = event["box"]
                ident = cert["id"]
                require(ident in active, "rejection of inactive leaf")
                require(certificate_digest(cert) == active[ident][1], "rejection fields changed")
                require(exact(event["incumbent_lower"]) == incumbent, "rejection incumbent mismatch")
                require(exact(cert["upper"]) < incumbent, "nonstrict recorded rejection")
                rejected.append(cert)
                del active[ident]
            elif kind == "precision_raise":
                certs = event["active"]
                ids = [cert["id"] for cert in certs]
                require(len(set(ids)) == len(ids) and set(ids) == set(active), "refresh lost leaves")
                require(event["precision_bits"] in (256, 512), "unsupported historical precision")
                lowers = [register(cert, refresh=True) for cert in certs]
                incumbent = max(incumbent, *lowers)
                require(exact(event["incumbent_lower"]) == incumbent, "refresh incumbent mismatch")
            elif kind == "remaining":
                final = event["active"]
                ids = [cert["id"] for cert in final]
                require(len(set(ids)) == len(ids) and set(ids) == set(active), "terminal leaf set changed")
                for cert in final:
                    require(certificate_digest(cert) == active[cert["id"]][1],
                            "terminal certificate fields differ from creation/refresh")
                require(exact(event["incumbent_lower"]) == incumbent, "terminal incumbent mismatch")
                require(event["termination"] == row["termination"], "termination mismatch")
            else:
                raise CertificateError(f"unsupported terminal trace event: {kind}")
    require(root is not None and final is not None and final, "complete terminal event missing")
    require(events == row["trace"]["events"], "event count mismatch")
    require(splits == row["subdivisions"] and len(rejected) == row["rejections"], "tree count mismatch")
    require(len(final) == row["retained_leaves"], "retained count mismatch")
    require(incumbent == exact(row["global_objective"][0]), "global lower bound mismatch")
    require(h.hexdigest() == expected_hash == row["trace"]["sha256"], "trace SHA mismatch")
    check_exact_cover(root, rejected + final)
    return TerminalPartition(root, final, rejected, [root[0], root[1], best_coordinate],
                             incumbent, events, splits, h.hexdigest())


def load_backend():
    global _arb, _fmpq
    import flint
    from flint import arb, fmpq, ctx
    ctx.prec = PRECISION
    _arb, _fmpq = arb, fmpq
    return {"python_flint_version": flint.__version__, "precision_bits": PRECISION,
            "shared_primitives": "official Arb elementary functions only"}


def ball(value):
    if isinstance(value, _arb):
        return value
    value = exact(value)
    return _arb(_fmpq(value.numerator, value.denominator))


def fraction(value):
    require(value.is_exact() and value.is_finite(), "nonexact or nonfinite endpoint")
    return F(str(value.fmpq()))


class Interval:
    """Endpoint arithmetic; each elementary operation rounds outwards."""
    def __init__(self, lo, hi=None):
        if hi is None:
            value = ball(lo)
            lo, hi = value.lower(), value.upper()
        require(lo.is_exact() and hi.is_exact() and lo.is_finite()
                and hi.is_finite() and lo <= hi, "invalid Arb interval")
        self.lo, self.hi = lo, hi

    def __add__(self, other):
        other = interval(other)
        return Interval((self.lo + other.lo).lower(), (self.hi + other.hi).upper())

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -interval(other)

    def __rsub__(self, other):
        return interval(other) + -self

    def __mul__(self, other):
        other = interval(other)
        products = [a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(min(p.lower() for p in products), max(p.upper() for p in products))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = interval(other)
        require(not other.lo <= 0 <= other.hi, "division interval contains zero")
        return self * Interval((1 / other.hi).lower(), (1 / other.lo).upper())

    def __rtruediv__(self, other):
        return interval(other) / self

    def square(self):
        values = [self.lo * self.lo, self.hi * self.hi]
        return Interval(_arb(0) if self.lo <= 0 <= self.hi else min(x.lower() for x in values),
                        max(x.upper() for x in values))

    def log(self):
        require(self.lo > 0, "log domain")
        return Interval(self.lo.log().lower(), self.hi.log().upper())

    def exp(self):
        return Interval(self.lo.exp().lower(), self.hi.exp().upper())

    def sqrt(self):
        require(self.lo >= 0, "sqrt domain")
        return Interval(self.lo.sqrt().lower(), self.hi.sqrt().upper())

    def erfc(self):
        return Interval(self.hi.erfc().lower(), self.lo.erfc().upper())

    def endpoints(self):
        return fraction(self.lo), fraction(self.hi)


def interval(value):
    return value if isinstance(value, Interval) else Interval(value)


def normal_tail(value):
    return (interval(value) / interval(2).sqrt()).erfc() / 2


def normal_density(value):
    return (-interval(value).square() / 2).exp() / (2 * interval(_arb.pi())).sqrt()


def clip(value, alpha, beta):
    low, high = interval(alpha), interval(beta)
    return Interval(min(max(value.lo, low.lo), high.lo), min(max(value.hi, low.hi), high.hi))


def assert_contained(independent, stored, label):
    lo, hi = independent.endpoints()
    old_lo, old_hi = pair(stored)
    if not old_lo <= lo <= hi <= old_hi:
        raise ArithmeticUnresolved(f"independent {label} is not contained in its stored enclosure")


class IndependentProfile:
    def __init__(self, spec):
        self.spec = spec
        self.q0, self.lam = exact(spec["q0"]), exact(spec["lambda"])
        self.d, self.k = spec["d"], self.lam / 2
        self.r = self.q0 * (1 - self.q0)
        self.a, self.b = (1 - self.q0) ** 2, self.q0 ** 2
        self.alpha, self.beta = map(exact, spec["I"])
        # Per-cell bounded cache; a full trace never accumulates Arb objects.
        self.point = lru_cache(maxsize=2048)(self._point)

    def count_objective(self, u, q):
        u, q = interval(u), interval(q)
        return 2 * u * q.log() + 2 * (1 - u) * (1 - q).log()

    def optimized_count(self, u):
        # K'(u)=2 log(q_I/(1-q_I))<0: evaluate separate exact endpoints.
        upper_u, lower_u = interval(u.hi), interval(u.lo)
        lower = self.count_objective(upper_u, clip(upper_u, self.alpha, self.beta)).lo
        upper = self.count_objective(lower_u, clip(lower_u, self.alpha, self.beta)).hi
        return Interval(lower, upper)

    def _point(self, coordinate):
        coordinate = exact(coordinate)
        c = interval(coordinate)
        forward, shifted, backward = normal_tail(c), normal_tail(c - self.lam), normal_tail(self.lam - c)
        g = self.a * normal_density(c) + self.b * normal_density(c - self.lam)
        lower_gap = self.a * forward + self.b * shifted
        upper_gap = self.b * backward - self.a * forward
        u = self.r + lower_gap
        error = (self.lam ** 2 * (self.a * forward + self.b * backward) - 2 * self.lam * g) / self.d
        v = 1 + error
        require(v.lo > 0, "nonpositive profiled variance")
        value = self.optimized_count(u) - F(self.d, 2) * v.log() - F(self.d, 2)
        return {"u": u, "q": clip(u, self.alpha, self.beta), "v": v,
                "variance_error": error, "P": value, "q_minus_r": lower_gap,
                "q0_minus_q": upper_gap}

    def tail_upper(self):
        vmax = 1 + self.b * self.lam ** 2 / self.d
        return interval(self.k) + 2 * vmax / self.lam * (interval((1 - self.r) / self.r)).log()

    def box(self, domain):
        lo, hi = pair(domain)
        require(lo >= self.k and hi > lo, "box outside threshold domain")
        left, right, middle = self.point(lo), self.point(hi), self.point((lo + hi) / 2)
        u = Interval(right["u"].lo, left["u"].hi)
        q = Interval(right["q"].lo, left["q"].hi)
        v = Interval(left["v"].lo, right["v"].hi)
        error = Interval(left["variance_error"].lo, right["variance_error"].hi)
        monotone = (self.optimized_count(right["u"]) - F(self.d, 2) * left["v"].log() - F(self.d, 2)).hi
        logit_lo = (2 * ((1 - interval(q.hi)) / interval(q.hi)).log()).lo
        logit_hi = (2 * ((1 - interval(q.lo)) / interval(q.lo)).log()).hi
        penalty_lo = (self.lam * interval(lo - self.k) / interval(v.hi)).lo
        penalty_hi = (self.lam * interval(hi - self.k) / interval(v.lo)).hi
        c = Interval(ball(lo), ball(hi))
        density = self.a * normal_density(c) + self.b * normal_density(c - self.lam)
        derivative = density * (Interval(logit_lo, logit_hi) - Interval(penalty_lo, penalty_hi))
        mean = (interval(middle["P"].hi) + (hi - lo) / 2 * interval(max(abs(derivative.lo), abs(derivative.hi)))).hi
        return {"upper": min(monotone, mean), "monotone_upper": monotone,
                "mean_value_upper": mean, "derivative": derivative, "u": u,
                "q": q, "v": v, "variance_error": error, "witness": middle["P"].lo}

    def outside_penalties(self, u):
        """Safe lower bounds for excluding q<=r and q>=q0, respectively."""
        lo, hi = u.endpoints()
        lo = max(lo, self.r)  # u(c)>r for every finite c, known analytically.
        arg = interval(lo)
        qfit = clip(arg, self.alpha, self.beta)
        lower_gap = self.count_objective(arg, qfit) - self.count_objective(arg, self.r)
        lower = max(F(0), fraction(lower_gap.lo))
        if hi >= self.q0:
            upper = F(0)
        else:
            arg = interval(hi)
            kl = 2 * (arg * (arg / self.q0).log()
                      + (1 - arg) * ((1 - arg) / (1 - self.q0)).log())
            upper = max(F(0), fraction(kl.lo))
        return lower, upper


def validate_leaf(profile, cert):
    independent = profile.box(cert["domain"])
    for key in ("q", "v", "variance_error", "derivative"):
        assert_contained(independent[key], cert[key], key)
    for key in ("upper", "monotone_upper", "mean_value_upper"):
        if fraction(independent[key]) > exact(cert[key]):
            raise ArithmeticUnresolved(f"stored {key} not independently validated")
    require(exact(cert["upper"]) == min(exact(cert["monotone_upper"]), exact(cert["mean_value_upper"])),
            "stored objective upper does not equal its stated construction")
    if exact(cert["witness_lower"]) > fraction(independent["witness"]):
        raise ArithmeticUnresolved("stored midpoint witness not independently validated")
    return independent


def component_groups(certificates):
    groups = []
    for cert in sorted(certificates, key=lambda c: exact(c["domain"][0])):
        if groups and exact(groups[-1][-1]["domain"][1]) == exact(cert["domain"][0]):
            groups[-1].append(cert)
        else:
            groups.append([cert])
    return groups


def intersect_exact(value, low, high):
    lo, hi = value.endpoints()
    lo, hi = max(lo, low), min(hi, high)
    return None if lo > hi else [str(lo), str(hi)]


def summarize_components(profile, certificates):
    output = []
    for group in component_groups(certificates):
        domain = [group[0]["domain"][0], group[-1]["domain"][1]]
        left, right = profile.point(exact(domain[0])), profile.point(exact(domain[1]))
        q = Interval(right["q"].lo, left["q"].hi)
        v = Interval(left["v"].lo, right["v"].hi)
        error = Interval(left["variance_error"].lo, right["variance_error"].hi)
        q_image = intersect_exact(q, profile.r, profile.q0)
        r_gap = intersect_exact(Interval(right["q_minus_r"].lo, left["q_minus_r"].hi), F(0), profile.q0 - profile.r)
        truth_gap = intersect_exact(Interval(left["q0_minus_q"].lo, right["q0_minus_q"].hi), F(0), profile.q0 - profile.r)
        feasible = q_image is not None and r_gap is not None and truth_gap is not None
        ve = error.endpoints()
        output.append({"threshold": domain, "leaf_ids": [cert["id"] for cert in group],
                       "raw_q": list(map(str, q.endpoints())), "q": q_image if feasible else None,
                       "v": list(map(str, v.endpoints())), "variance_error": list(map(str, ve)),
                       "q_minus_r": r_gap if feasible else None, "q0_minus_q": truth_gap if feasible else None,
                       "relative_activity_bias": [str(exact(x) / profile.q0) for x in truth_gap] if feasible else None,
                       "theorem_intersection_empty": not feasible,
                       "variance_error_sign": "positive" if ve[0] > 0 else "negative" if ve[1] < 0 else "unresolved",
                       "activity_width_target_met": feasible and exact(q_image[1]) - exact(q_image[0]) <= F(1, 10**6),
                       "variance_width_target_met": feasible and fraction(v.hi) - fraction(v.lo) <= F(1, 10**6)})
    return output


def validate_original_components(profile, row, active):
    by_id = {cert["id"]: cert for cert in active}
    consumed = set()
    for comp in row["components"]:
        ids = comp["leaf_ids"]
        require(ids and len(set(ids)) == len(ids) and not consumed.intersection(ids), "duplicate component leaf")
        require(set(ids) <= set(by_id), "component contains absent leaf")
        leaves = [by_id[ident] for ident in ids]
        domain = pair(comp["threshold"])
        check_exact_cover(domain, leaves)
        consumed.update(ids)
        generated = summarize_components(profile, leaves)
        require(len(generated) == 1, "disconnected original component")
        fresh = generated[0]
        left, right = profile.point(domain[0]), profile.point(domain[1])
        assert_contained(Interval(right["q_minus_r"].lo, left["q_minus_r"].hi),
                         comp["raw_q_minus_r"], "original raw gap above r")
        assert_contained(Interval(left["q0_minus_q"].lo, right["q0_minus_q"].hi),
                         comp["raw_q0_minus_u"], "original raw gap below q0")
        for key in ("raw_q", "v", "variance_error"):
            require(pair(comp[key])[0] <= pair(fresh[key])[0] <= pair(fresh[key])[1] <= pair(comp[key])[1],
                    f"original component {key} not independently enclosed")
        if comp["activity_width_target_met"]:
            require(pair(comp["raw_q"])[1] - pair(comp["raw_q"])[0] <= F(1, 10**6),
                    "original component overstates activity resolution")
        if comp["variance_width_target_met"]:
            require(pair(comp["v"])[1] - pair(comp["v"])[0] <= F(1, 10**6),
                    "original component overstates variance resolution")
        if comp["q"] is None:
            require(fresh["theorem_intersection_empty"], "unproved theorem-empty original component")
        elif not fresh["theorem_intersection_empty"]:
            # A narrower independently empty image is a valid stronger
            # exclusion; an old nonempty outer image did not assert existence.
            for key in ("q", "q_minus_r", "q0_minus_q", "relative_activity_bias"):
                lo, hi = pair(fresh[key])
                old_lo, old_hi = pair(comp[key])
                if not old_lo <= lo <= hi <= old_hi:
                    raise ArithmeticUnresolved(f"original component {key} not independently validated")
    require(consumed == set(by_id), "original components omit active leaves")
    if row["status"] == "certified_targets_met":
        require(all(comp["activity_width_target_met"] and comp["variance_width_target_met"]
                    for comp in row["components"]), "original resolved status contradicts its components")
        require(exact(row["objective_gap_upper"]) <= F(1, 10**10),
                "original resolved status overstates objective resolution")


def verify_positive(row, partition, tick=lambda: None):
    profile = IndependentProfile(row["spec"])
    tail = profile.tail_upper()
    require(partition.root[1] >= fraction(tail.hi), "unproved finite tail cutoff")
    assert_contained(tail, row["tail_bound"], "tail cutoff")
    witness = max(fraction(profile.point(c)["P"].lo) for c in partition.witness_candidates)
    lower = partition.incumbent
    if lower > witness:
        raise ArithmeticUnresolved("global incumbent has no independently verified feasible witness")
    all_leaves = check_exact_cover(partition.root, partition.rejected + partition.active)
    historical_ids = {cert["id"] for cert in partition.rejected}
    kept = []
    new_rejected = 0
    upper = None
    outside_lower = outside_upper = None
    for cert in all_leaves:
        tick()
        box = validate_leaf(profile, cert)
        numerical_upper = fraction(box["upper"])
        lower_penalty, upper_penalty = profile.outside_penalties(box["u"])
        candidate_lower = numerical_upper - lower_penalty
        candidate_upper = numerical_upper - upper_penalty
        outside_lower = candidate_lower if outside_lower is None else max(outside_lower, candidate_lower)
        outside_upper = candidate_upper if outside_upper is None else max(outside_upper, candidate_upper)
        drop = exact(cert["upper"]) < lower
        if cert["id"] in historical_ids:
            require(drop, "historical rejection not below final incumbent")
        if drop:
            if not numerical_upper < lower:
                raise ArithmeticUnresolved("strict exclusion not independently certified")
            if cert["id"] not in historical_ids:
                new_rejected += 1
        else:
            kept.append(cert)
            upper = numerical_upper if upper is None else max(upper, numerical_upper)
    require(kept and upper is not None, "empty all-maximizer cover")
    require(upper >= lower, "negative independent objective gap")
    old_lower, old_upper = pair(row["global_objective"])
    require(old_lower == lower and max(exact(c["upper"]) for c in partition.active) <= old_upper,
            "original global objective upper mismatch")
    require(old_upper - old_lower <= exact(row["objective_gap_upper"]), "original objective width mismatch")
    validate_original_components(profile, row, partition.active)
    end = profile.point(partition.root[1])
    lo_penalty, hi_penalty = profile.outside_penalties(end["u"])
    tail_lower = fraction(end["P"].hi) - lo_penalty
    tail_upper = fraction(end["P"].hi) - hi_penalty
    outside_lower, outside_upper = max(outside_lower, tail_lower), max(outside_upper, tail_upper)
    components = summarize_components(profile, kept)
    feasible = [comp for comp in components if not comp["theorem_intersection_empty"]]
    require(feasible, "no component intersects the proven activity bracket")
    targets = {"activity_width_target_met": all(c["activity_width_target_met"] for c in feasible),
               "variance_width_target_met": all(c["variance_width_target_met"] for c in feasible),
               "objective_width_target_met": upper - lower <= F(1, 10**10)}
    return {"spec": row["spec"], "status": "verified", "audit_status": "verified", "original_status": row["status"],
            "derived_status": "targets_met" if all(targets.values()) else "outer_unresolved",
            **targets, "components": components, "global_objective": [str(lower), str(upper)],
            "feasible_lower": str(lower), "verified_feasible_witness_lower": str(witness),
            "outside_upper": {"q_le_r": str(outside_lower), "q_ge_q0": str(outside_upper)},
            "terminal_partition_leaves": len(all_leaves), "historical_rejected_leaves": len(partition.rejected),
            "new_strictly_rejected_leaves": new_rejected, "retained_leaves": len(kept),
            "covering_splits": partition.splits, "trace_events": partition.events,
            "trace_sha256": partition.trace_sha256,
            "outside_activity": {"lower_interval": [str(profile.alpha), str(profile.r)],
                                 "upper_interval": [str(profile.q0), str(profile.beta)],
                                 "lower_region_objective_upper": str(outside_lower),
                                 "upper_region_objective_upper": str(outside_upper),
                                 "lower_margin": str(lower - outside_lower),
                                 "upper_margin": str(lower - outside_upper),
                                 "tail_cutoff": str(partition.root[1]),
                                 "tail_lower_region_upper": str(tail_lower),
                                 "tail_upper_region_upper": str(tail_upper),
                                 "covers_entire_threshold_halfline": True,
                                 "finite_noise_claim": False},
            "samples": 0, "new_optimization": False,
            "independence": "terminal-partition bounds independently recomputed; primary engine not imported"}


def verify_anchor(row):
    q0 = exact(row["spec"]["q0"])
    r = q0 * (1 - q0)
    require(exact(row["spec"]["lambda"]) == 0, "nonzero analytic anchor")
    require(row["status"] == "exact_analytic_anchor" and len(row["components"]) == 1,
            "invalid analytic anchor status")
    comp = row["components"][0]
    required = {"q": r, "v": F(1), "variance_error": F(0), "q_minus_r": F(0),
                "q0_minus_q": q0 * q0, "relative_activity_bias": q0}
    for key, value in required.items():
        require(pair(comp[key]) == (value, value), f"incorrect analytic anchor {key}")
    require(exact(row["objective_gap_upper"]) == 0, "nonzero analytic anchor objective gap")
    profile = IndependentProfile(row["spec"])
    feasible = profile.count_objective(r, r) - F(profile.d, 2)
    upper_region = profile.count_objective(r, q0) - F(profile.d, 2)
    return {"spec": row["spec"], "status": "verified", "audit_status": "verified", "original_status": row["status"],
            "derived_status": "exact_analytic_anchor", "components": row["components"],
            "feasible_lower": str(fraction(feasible.lo)),
            "outside_upper": {"q_le_r": None, "q_ge_q0": str(fraction(upper_region.hi))},
            "finite_noise_claim": False, "strict_lower_activity_margin_claimed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-seconds", type=int, default=600, choices=range(10, 601), metavar="10..600")
    parser.add_argument("--wall-seconds", type=int, default=900, choices=range(10, 901), metavar="10..900")
    args = parser.parse_args(argv)
    require(os.environ.get("VG_SAE_COLAB_CPU_ONLY") == "1",
            "numerical execution requires the authorized Colab CPU runtime marker")
    require(not sys.flags.optimize, "optimized Python execution is not accepted")
    source, destination = args.input_dir.resolve(), args.output_dir.resolve()
    require(source.is_dir(), "input directory missing")
    require(destination != source and source not in destination.parents, "output must not modify the historical run")
    require(not destination.exists(), "new output directory required")
    destination.mkdir(parents=True)
    start_cpu, start_wall = time.process_time(), time.monotonic()
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    _, hard_cpu = resource.getrlimit(resource.RLIMIT_CPU)
    hard = math.ceil(start_cpu + args.cpu_seconds + 1)
    require(hard_cpu == resource.RLIM_INFINITY or hard_cpu >= hard, "inherited CPU cap below requested audit allowance")
    resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
    old_memory = resource.getrlimit(resource.RLIMIT_AS)[1]
    memory = 512 * 1024 ** 2
    require(old_memory == resource.RLIM_INFINITY or old_memory >= memory, "inherited memory cap below 512 MiB")
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))

    def tick():
        if time.process_time() - start_cpu >= args.cpu_seconds - 5:
            raise TimeoutError("fixed CPU allowance reached; reserving result serialization")
        if time.monotonic() - start_wall >= args.wall_seconds - 5:
            raise TimeoutError("fixed wall allowance reached; reserving result serialization")

    def timed_stop(signum, frame):
        raise TimeoutError("fixed wall allowance reached")

    signal.signal(signal.SIGALRM, timed_stop)
    signal.setitimer(signal.ITIMER_REAL, max(1, args.wall_seconds - 5))
    write_new_json(destination / "START.json", {"version": VERSION, "source_sha256": digest(__file__),
                   "cpu_limit_seconds": args.cpu_seconds, "wall_limit_seconds": args.wall_seconds,
                   "memory_limit_bytes": memory, "cpu_cores": 1, "samples": 0,
                   "required_source_manifest_sha256": SOURCE_MANIFEST_SHA256,
                   "required_historical_run_seal_sha256": RUN_SEAL_SHA256})
    rows = expected_panel()
    results = []
    bindings = {}
    failure = None
    backend = None
    current = None
    try:
        manifest_path = source / "FROZEN_MANIFEST.json"
        run = source / "outputs" / "locked_run_001"
        require(digest(manifest_path, tick) == SOURCE_MANIFEST_SHA256, "frozen source manifest changed")
        seal_path = run / "RUN_FINAL_SEAL.json"
        require(digest(seal_path, tick) == RUN_SEAL_SHA256, "historical run seal changed")
        seal = read_json(seal_path)
        require(seal["manifest_sha256"] == SOURCE_MANIFEST_SHA256, "historical source/result binding mismatch")
        summary_hash = digest(run / "SUMMARY.json", tick)
        require(summary_hash == seal["files"]["SUMMARY.json"], "summary SHA mismatch")
        bindings["SUMMARY.json"] = summary_hash
        summary = read_json(run / "SUMMARY.json")
        require(summary["denominator"] == 20 and [row["spec"] for row in summary["results"]] == rows,
                "fixed 20-cell panel changed")
        backend = load_backend()
        with (destination / "VERIFIED_PREFIX.jsonl").open("x") as journal:
            for row in summary["results"]:
                tick()
                current = row["spec"]["id"]
                result_name = current + ".result.json"
                result_hash = digest(run / result_name, tick)
                require(result_hash == seal["files"][result_name], "cell result SHA mismatch")
                validate_summary_row(row, read_json(run / result_name))
                bindings[result_name] = result_hash
                cell_start = time.process_time()
                if exact(row["spec"]["lambda"]) == 0:
                    result = verify_anchor(row)
                else:
                    name = current + ".trace.jsonl"
                    partition = parse_partition(run / name, row, seal["files"][name], tick)
                    bindings[name] = partition.trace_sha256
                    result = verify_positive(row, partition, tick)
                    del partition
                result["audit_cpu_seconds"] = time.process_time() - cell_start
                results.append(result)
                journal.write(json.dumps(result, sort_keys=True) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(json.dumps({"cell": current, "audit_status": result["audit_status"],
                                  "derived_status": result["derived_status"],
                                  "completed": len(results), "cpu_seconds": time.process_time() - start_cpu}), flush=True)
                current = None
    except (Exception, KeyboardInterrupt) as error:
        failure = {"type": type(error).__name__, "message": str(error), "cell": current}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    completed = {result["spec"]["id"] for result in results}
    for spec in rows:
        if spec["id"] not in completed:
            results.append({"spec": spec, "status": "audit_unresolved", "audit_status": "audit_unresolved",
                            "derived_status": "not_verified_in_this_run", "feasible_lower": None,
                            "outside_upper": {"q_le_r": None, "q_ge_q0": None}})
    order = {spec["id"]: i for i, spec in enumerate(rows)}
    results.sort(key=lambda result: order[result["spec"]["id"]])
    complete = failure is None and len(completed) == 20
    status = "verified" if complete else "audit_failed" if failure and failure["type"] == "CertificateError" else "audit_unresolved"
    receipt = {"version": VERSION, "status": status, "passed": complete,
               "completed_cells": len(completed), "denominator": 20, "cells": results,
               "failure": failure, "backend": backend, "source_sha256": digest(__file__),
               "source_manifest_sha256": SOURCE_MANIFEST_SHA256, "historical_run_seal_sha256": RUN_SEAL_SHA256,
               "consumed_artifact_sha256": bindings,
               "binding_scope": "fixed manifest and result seal hashes plus every artifact consumed by a verified row",
               "all_archival_dependencies_rehashed": False,
               "cpu_seconds": time.process_time() - start_cpu, "cpu_limit_seconds": args.cpu_seconds,
               "wall_seconds": time.monotonic() - start_wall, "wall_limit_seconds": args.wall_seconds,
               "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               "memory_limit_bytes": memory, "original_outputs_changed": False,
               "new_optimization": False, "finite_noise_claim": False, "native_vg_claim": False}
    write_new_json(destination / "TERMINAL_AUDIT.json", receipt)
    write_new_json(destination / "SEAL.json", {"version": VERSION,
                   "files": {path.name: digest(path) for path in destination.iterdir() if path.is_file()}})
    print(json.dumps({"status": status, "completed_cells": len(completed), "failure": failure}), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
