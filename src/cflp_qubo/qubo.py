import math
from typing import Any

import sympy as sp
from tno.quantum.optimization.qubo.components import QUBO
from tno.quantum.utils import BitVector


class CFLP_QUBO:
    """Capacitated Facility Location Problem (CFLP) formulated as a QUBO."""

    def __init__(
        self,
        customers: list[str],
        facilities: list[str],
        customer_demands: dict[str, float],
        facility_capacities: dict[str, float],
        facility_costs: dict[str, float],
        service_costs: dict[tuple[str, str], float],
        normalize_qubo: bool = True,
        print_penalty_estimates: bool = False,
        debug: bool = False,
    ) -> None:
        """Initializes the CFLP_QUBO instance with problem data.

        Args:
            customers: List of customer identifiers.
            facilities: List of facility identifiers.
            customer_demands: Dictionary mapping customers to their demands.
            facility_capacities: Dictionary mapping facilities to their capacities.
            facility_costs: Dictionary mapping facilities to their fixed opening costs.
            service_costs: Dictionary mapping (customer, facility) tuples to service costs.
            normalize_qubo: Whether to normalize final QUBO coefficients.
            print_penalty_estimates: Whether to print estimated penalty coefficients.
            debug: Whether to enable debug printing.
        """
        self.customers = customers
        self.facilities = facilities
        self.di = customer_demands
        self.Qf = facility_capacities
        self.Ff = facility_costs
        self.cif = service_costs
        self.normalize_qubo = normalize_qubo
        self.print_penalty_estimates = print_penalty_estimates
        self.debug = debug

        self.num_customers = len(customers)
        self.num_facilities = len(facilities)

        self.y_vars = {
            f: sp.Symbol(f"y_{fdx}") for fdx, f in enumerate(self.facilities)
        }

        self.x_vars = {
            (c, f): sp.Symbol(f"x_{cdx}_{fdx}")
            for cdx, c in enumerate(self.customers)
            for fdx, f in enumerate(self.facilities)
        }

        self.slack_vars = {}
        self.slack_bits = {}

        for fdx, f in enumerate(self.facilities):
            capacity = self.Qf[f]
            num_slack_bits = math.ceil(math.log2(capacity + 1))
            self.slack_bits[f] = num_slack_bits
            self.slack_vars[f] = [
                sp.Symbol(f"s_{fdx}_{k}") for k in range(num_slack_bits)
            ]

        self.lam1: sp.Expr = sp.Symbol("λ1")
        self.lam2: sp.Expr = sp.Symbol("λ2")
        self.lam3: sp.Expr = sp.Symbol("λ3")
        self.all_vars: list[sp.Symbol] = []
        self.penalties_provided: bool | None = None
        self.normalization_factor: float | None = None

        if self.debug:
            print("\nCFLP_QUBO initialized with the following parameters:")
            print(f"Customers: {self.customers}")
            print(f"Facilities: {self.facilities}")
            print(f"Customer Demands: {self.di}")
            print(f"Facility Capacities: {self.Qf}")
            print(f"Facility Costs: {self.Ff}")
            print(f"Service Costs: {self.cif}")
            print(f"Slack Variables: {self.slack_vars}")

    def _get_qubo_symbolic(self) -> sp.Expr:
        """Builds the symbolic QUBO expression for the CFLP."""
        Q = 0

        # Cost terms
        for f in self.facilities:
            Q += self.Ff[f] * self.y_vars[f]

        for c in self.customers:
            for f in self.facilities:
                Q += self.cif[(c, f)] * self.x_vars[(c, f)]

        # Constraint #1: only one customer per facility
        for c in self.customers:
            assigned_sum = sum(self.x_vars[(c, f)] for f in self.facilities)
            Q += self.lam1 * (assigned_sum - 1) ** 2

        # Constraint #2: facility capacity does not exceed the sum of assigned customer demands
        for f in self.facilities:
            capacity = self.Qf[f] * self.y_vars[f]
            assigned_demand = sum(
                self.di[c] * self.x_vars[(c, f)] for c in self.customers
            )
            slack_sum = sum(
                (2**k) * self.slack_vars[f][k] for k in range(self.slack_bits[f])
            )
            Q += self.lam2 * (assigned_demand + slack_sum - capacity) ** 2

        # Constraint #3: does not assign customers to closed facilities
        for c in self.customers:
            for f in self.facilities:
                Q += self.lam3 * self.x_vars[(c, f)] * (1 - self.y_vars[f])

        if self.debug:
            print("\nSymbolic QUBO expression constructed.")
            print(Q)

        return sp.expand(Q)

    def get_qubo(
        self,
        substitute: dict[Any, Any] | None = None,
    ) -> QUBO:
        """Generates the QUBO matrix and offset for the CFLP.

        Args:
            substitute: Optional dictionary to substitute symbolic penalty coefficients with numeric values.

        Returns:
            QUBO class instance representing the QUBO matrix and offset.

        Raises:
            ValueError: If symbolic coefficients remain in the QUBO after substitution or if the polynomial is not quadratic.

        """
        Q_symbolic = self._get_qubo_symbolic()

        if not substitute:
            self.penalties_provided = False
        else:
            self.penalties_provided = True

        if self.print_penalty_estimates or not self.penalties_provided:
            est_lam1, est_lam2, est_lam3 = self._estimate_penalties_from_problem_data()

            if self.print_penalty_estimates:
                print("\nEstimated penalty coefficients based on problem data:")
                print(f"  λ₁ ≥ {est_lam1}")
                print(f"  λ₂ ≥ {est_lam2}")
                print(f"  λ₃ ≥ {est_lam3}")

            if not self.penalties_provided:
                substitute = {
                    sp.Symbol("λ1"): est_lam1,
                    sp.Symbol("λ2"): est_lam2,
                    sp.Symbol("λ3"): est_lam3,
                }

                print("\nPenalty coefficients were not provided, using estimates:")
                print(f"  λ₁ = {est_lam1}")
                print(f"  λ₂ = {est_lam2}")
                print(f"  λ₃ = {est_lam3}")

        Q_symbolic = Q_symbolic.subs(substitute)

        self.all_vars = list(self.x_vars.values()) + list(self.y_vars.values())

        for f in self.facilities:
            self.all_vars.extend(self.slack_vars[f])

        n = len(self.all_vars)

        Q_matrix: dict[tuple[int, int], float] = {
            (u, v): 0.0 for u in range(n) for v in range(n)
        }

        offset: float = 0.0

        # Polynomial expansion
        poly = sp.Poly(Q_symbolic, self.all_vars)

        # Fill Q matrix term-by-term
        for monom, coeff in poly.terms():
            # Extract non-zero exponents
            # nz has the form [(variable index 1, power 1), (variable index 2, power 2)]
            nz = [(i, p) for i, p in enumerate(monom) if p != 0]

            # Coerce numeric coefficient
            try:
                coeff_val = float(coeff)
            except (TypeError, ValueError) as error:
                error_msg = (
                    "Symbolic coefficients remain in QUBO. "
                    "Substitute numeric values for symbols (e.g. λ1, λ2, …) first."
                )
                raise ValueError(error_msg) from error

            if len(nz) == 0:
                offset += coeff_val
                continue

            # Linear or square term
            if len(nz) == 1:
                i, p = nz[0]

                if p == 1:
                    # Linear term: coeff * x_i
                    Q_matrix[(i, i)] += coeff_val

                elif p == 2:
                    # Square term: coeff * x_i^2 -> coeff * x_i (binary vars)
                    Q_matrix[(i, i)] += coeff_val

                else:
                    error_msg = "Power > 2 detected - not a QUBO."
                    raise ValueError(error_msg)

            # Quadratic terms
            elif len(nz) == 2:
                (i, p1), (j, p2) = nz

                if p1 == 1 and p2 == 1:
                    # Interaction term: coeff * x_i x_j
                    u, v = min(i, j), max(i, j)
                    Q_matrix[(u, v)] += coeff_val
                else:
                    error_msg = "Invalid quadratic powers for QUBO."
                    raise ValueError(error_msg)

            else:
                error_msg = "Polynomial contains higher-than-quadratic terms."
                raise ValueError(error_msg)

        if self.normalize_qubo:
            Q_matrix, offset = self._normalize_qubo(Q_matrix, offset)

        matrix = QUBO(Q_matrix, offset=offset)

        if self.debug:
            print("\nQUBO matrix constructed:")
            print(f"Size: {matrix.size}")
            print(f"Offset: {matrix.offset}")

        return matrix

    def _normalize_qubo(
        self,
        qubo_dict: dict[tuple[int, int], float],
        offset: float = 0.0,
    ) -> dict[tuple[int, int], float]:
        """Normalizes QUBO coefficients to the range [-1, 1].

        Args:
            qubo_dict: Dictionary representing the QUBO matrix.
            offset: The QUBO offset.

        Returns:
            Normalized QUBO dictionary.
        """
        vals = [abs(v) for v in qubo_dict.values()]
        vals.append(abs(offset))

        max_coeff = max(vals)

        if max_coeff > 0:
            scale = 1.0 / max_coeff
            for key in qubo_dict:
                qubo_dict[key] *= scale
            offset *= scale

            self.normalization_factor = max_coeff

        print(f"\nQUBO coefficients normalized by max |coef| = {max_coeff}.")

        return qubo_dict, offset

    def _estimate_penalties_from_problem_data(self) -> tuple[float, float, float]:
        """Prints estimates for penalty coefficients based on problem data."""
        max_cif = max(self.cif.values())
        max_Fcost = max(self.Ff.values())

        estimated_lam1 = max_cif + 1
        estimated_lam2 = len(self.customers) * max_cif + max_Fcost + 1
        estimated_lam3 = max_Fcost + max_cif + 1

        return estimated_lam1, estimated_lam2, estimated_lam3

    def decode_solution(
        self,
        bitstring: BitVector,
        *,
        check_constraints: bool = True,
    ) -> tuple[dict[str, int], dict[str, list[str]]]:
        """Decodes a bitvector solution into facility openings and customer assignments.

        Args:
            bitstring: The bitvector representing the solution.
            check_constraints: Whether to check for constraint violations.

        Returns:
            A tuple containing:
                - A dictionary mapping facility identifiers to their open (1) or closed (0) status.
                - A dictionary mapping customer identifiers to a list of assigned facility identifiers.

        Raises:
            ValueError: If the length of the bitvector does not match the number of variables.
        """
        bits = [int(b) for b in bitstring.bits]

        if len(bits) != len(self.all_vars):
            raise ValueError("Bitvector length mismatch")

        # Decode variables
        open_facilities = {}
        assignments = {c: [] for c in self.customers}

        for bit, var in zip(bits, self.all_vars):
            if bit != 1:
                continue

            name = str(var)

            if name.startswith("y_"):
                f = int(name.split("_")[1])
                open_facilities[self.facilities[f]] = 1

            elif name.startswith("x_"):
                _, c, f = name.split("_")
                assignments[self.customers[int(c)]].append(self.facilities[int(f)])

        # Fill closed facilities explicitly
        for f in self.facilities:
            open_facilities.setdefault(f, 0)

        if check_constraints:
            violations = self._check_constraints(open_facilities, assignments)
            if violations:
                print("\nConstraint violations detected in the solution:")
                for constraint, message in violations.items():
                    print(f" • {constraint}: {message}")
            elif self.debug:
                print("\nNo constraint violations detected in the solution.")

        return open_facilities, assignments

    def _check_constraints(
        self,
        open_facilities: dict[str, int],
        assignments: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Checks constraint violations in the decoded solution."""
        violations = {}

        # Constraint #1: each customer assigned to exactly one facility
        for c in self.customers:
            assigned_count = len(assignments[c])
            if assigned_count != 1:
                violations[f"Customer {c} assignment"] = (
                    f"Assigned to {assigned_count} facilities."
                )

        # Constraint #2: facility capacity not exceeded
        for f in self.facilities:
            if open_facilities[f] == 0:
                continue

            total_demand = sum(
                self.di[c] for c in self.customers if f in assignments[c]
            )
            eps = 1e-9
            if total_demand > self.Qf[f] + eps:
                violations[f"Facility {f} capacity"] = (
                    f"Total demand {total_demand} exceeds capacity {self.Qf[f]} (tol={eps})."
                )

        # Constraint #3: customers only assigned to open facilities
        for c in self.customers:
            for f in assignments[c]:
                if open_facilities[f] == 0:
                    violations[f"Customer {c} to Facility {f}"] = (
                        "Assigned to closed facility."
                    )

        return violations
