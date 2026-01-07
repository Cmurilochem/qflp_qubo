import numpy as np
import sympy as sp
from tno.quantum.optimization.qubo import SolverConfig
from tno.quantum.optimization.qubo.components import QUBO
from tno.quantum.utils import BitVector

import pulp
from cflp_qubo import CFLP_QUBO


def solve_cflp_milp(
    customers: list[str],
    facilities: list[str],
    customer_demands: dict[str, float],
    facility_capacities: dict[str, float],
    facility_costs: dict[str, float],
    service_costs: dict[tuple[str, str], float],
) -> dict[str, any]:
    """Helper function to solve CFLP using exact MILP formulation with PuLP."""

    model = pulp.LpProblem("CFLP", pulp.LpMinimize)

    # Decision variables
    y = pulp.LpVariable.dicts("y", facilities, 0, 1, cat="Binary")
    x = pulp.LpVariable.dicts(
        "x", ((c, f) for c in customers for f in facilities), 0, 1, cat="Binary"
    )

    # Objective
    facility_term = pulp.lpSum(facility_costs[f] * y[f] for f in facilities)
    service_term = pulp.lpSum(
        service_costs[(c, f)] * x[(c, f)] for c in customers for f in facilities
    )

    model += facility_term + service_term

    # Constraints
    for c in customers:
        model += pulp.lpSum(x[(c, f)] for f in facilities) == 1

    for f in facilities:
        model += (
            pulp.lpSum(customer_demands[c] * x[(c, f)] for c in customers)
            <= facility_capacities[f] * y[f]
        )

    for c in customers:
        for f in facilities:
            model += x[(c, f)] <= y[f]

    # Solve
    model.solve()

    return {
        "status": pulp.LpStatus[model.status],
        "objective": pulp.value(model.objective),
        "open_facilities": {f: int(y[f].value()) for f in facilities},
        "assignments": {
            c: [f for f in facilities if x[(c, f)].value() > 0.5] for c in customers
        },
    }


def test_cflp_qubo():
    """Test case for CFLP_QUBO class."""
    customers = ["C1", "C2"]
    facilities = ["F1", "F2", "F3"]
    customer_demands = {"C1": 5.0, "C2": 10.0}
    facility_capacities = {"F1": 10.0, "F2": 20.0, "F3": 25.0}
    facility_costs = {"F1": 100.0, "F2": 150.0, "F3": 200.0}
    service_costs = {
        ("C1", "F1"): 10.0,
        ("C1", "F2"): 20.0,
        ("C1", "F3"): 30.0,
        ("C2", "F1"): 15.0,
        ("C2", "F2"): 25.0,
        ("C2", "F3"): 35.0,
    }

    cflp_qubo = CFLP_QUBO(
        customers,
        facilities,
        customer_demands,
        facility_capacities,
        facility_costs,
        service_costs,
        normalize_qubo=True,
        print_penalty_estimates=True,
        debug=True,
    )

    assert cflp_qubo.customers == customers
    assert cflp_qubo.facilities == facilities
    assert cflp_qubo.di == customer_demands
    assert cflp_qubo.Qf == facility_capacities
    assert cflp_qubo.Ff == facility_costs
    assert cflp_qubo.cif == service_costs

    QUBO_symbolic = cflp_qubo._get_qubo_symbolic()

    assert QUBO_symbolic is not None
    assert isinstance(QUBO_symbolic, sp.Expr)

    QUBO_matrix = cflp_qubo.get_qubo(
        substitute={
            sp.Symbol("λ1"): 100.0,
            sp.Symbol("λ2"): 300.0,
            sp.Symbol("λ3"): 200.0,
        }
    )

    assert isinstance(QUBO_matrix, QUBO)
    assert QUBO_matrix.size == len(cflp_qubo.all_vars)
    assert isinstance(QUBO_matrix.matrix, np.ndarray)
    assert cflp_qubo.penalties_provided is True
    assert cflp_qubo.normalization_factor is not None

    solver = SolverConfig(
        name="simulated_annealing_solver",
        options={"num_reads": 500},
    ).get_instance()

    result = solver.solve(QUBO_matrix)

    assert isinstance(result.best_bitvector, BitVector)
    assert result.best_value is not None

    open_facilities, assignments = cflp_qubo.decode_solution(
        result.best_bitvector,
        check_constraints=True,
    )

    exact_solution = solve_cflp_milp(
        customers,
        facilities,
        customer_demands,
        facility_capacities,
        facility_costs,
        service_costs,
    )

    assert exact_solution["status"] == "Optimal"
    assert open_facilities == exact_solution["open_facilities"]
    assert assignments == exact_solution["assignments"]

    cflp_qubo_2 = CFLP_QUBO(
        customers,
        facilities,
        customer_demands,
        facility_capacities,
        facility_costs,
        service_costs,
    )
    cflp_qubo_2.get_qubo()

    assert cflp_qubo_2.penalties_provided is False
