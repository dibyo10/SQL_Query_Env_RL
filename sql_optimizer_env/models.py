# ─────────────────────────────────────────────────────────────────────────────
# models.py
#
# Type-safe contracts between the agent and the environment.
# These dataclasses define exactly what the agent sends and receives.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State


# ─────────────────────────────────────────────────────────────────────────────
# ACTION TIERS
#
# Tier 1 — Structural rewrites (always available)
#   Work on any Postgres database with no extensions required.
#   Modify the SQL AST directly via sqlglot.
#
#   4  push_predicate              Move WHERE filter into JOIN ON
#   5  replace_subquery_with_join  Rewrite IN (SELECT...) as JOIN
#   6  remove_redundant_join       Drop a JOIN whose columns are unused
#   7  replace_select_star         Expand SELECT * to explicit columns
#   8  materialize_cte             Add MATERIALIZED to WITH clause
#   9  submit                      End the episode, return current query
#
# Tier 2 — Hint-based rewrites (requires pg_hint_plan on user's database)
#   Add /*+ ... */ comments that Postgres reads via pg_hint_plan extension.
#   If pg_hint_plan is not installed, Postgres treats these as regular
#   comments and ignores them — producing 0.0 reward and corrupting training.
#   The environment probes for the extension at reset() time and only
#   includes these actions in legal_actions if the extension is present.
#
#   1  add_index_hint              /*+ IndexScan(table index) */
#   2  add_join_order_hint         /*+ Leading(t1 t2 t3) */
#   3  add_join_method_hint        /*+ HashJoin(t1 t2) */
# ─────────────────────────────────────────────────────────────────────────────

TIER1_ACTION_IDS = {4, 5, 6, 7, 8, 9}
TIER2_ACTION_IDS = {1, 2, 3}

ACTION_NAMES = {
    1: "add_index_hint",
    2: "add_join_order_hint",
    3: "add_join_method_hint",
    4: "push_predicate",
    5: "replace_subquery_with_join",
    6: "remove_redundant_join",
    7: "replace_select_star",
    8: "materialize_cte",
    9: "submit",
}


@dataclass
class SQLAction(Action):
    """
    What the agent sends to the environment each step.

    action_id — which rewrite to apply (see ACTION_NAMES above)
    params    — action-specific parameters, varies by action_id:

        action_id=1  {"table": "orders", "index": "idx_orders_status"}
        action_id=2  {"table_order": ["regions", "customers", "orders"]}
        action_id=3  {"table_a": "orders", "table_b": "customers", "method": "HashJoin"}
        action_id=4  {"target_table": "customers"}
        action_id=5  {}
        action_id=6  {"table": "regions"}
        action_id=7  {}
        action_id=8  {"cte_name": "us_customers"}
        action_id=9  {}
    """

    action_id: int = 9
    params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SQLObservation(Observation):
    """
    What the agent receives after every reset() and step().

    observation_vector — flat list of floats, fixed length, fixed order.
                         This is the input to the agent's neural network.
                         Names are for human reference only — the agent
                         only sees the numbers.

        Index  0  execution_time_ms          real execution time from EXPLAIN ANALYZE
        Index  1  total_plan_cost            highest cost node in the plan tree
        Index  2  has_seq_scan               1.0 if any Seq Scan node exists
        Index  3  has_subquery               1.0 if any Subquery Scan / InitPlan exists
        Index  4  max_rows_removed           largest Rows Removed by Filter seen
        Index  5  num_joins                  number of join nodes in the plan
        Index  6  has_redundant_join         1.0 if a join's table is unreferenced
        Index  7  has_cte                    1.0 if a WITH clause exists
        Index  8  has_select_star            1.0 if SELECT * is present
        Index  9  estimated_vs_actual_gap    abs(Plan Rows - Actual Rows) at worst node

    legal_actions — the ONLY actions the agent is allowed to pick from.
                    Actions not in this list are either inapplicable to
                    the current query or require an unavailable extension.
                    Each entry: {"action_id": int, "params": dict, "description": str}

    hints_available — whether pg_hint_plan is installed on the user's DB.
                      True  = Tier 1 + Tier 2 actions available
                      False = Tier 1 actions only
    """

    # Current SQL string (after rewrites applied so far this episode)
    current_query: str = ""

    # Flat feature vector for the neural network
    observation_vector: List[float] = field(default_factory=list)

    # Valid actions for this step
    legal_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Whether hint actions are available (pg_hint_plan detected)
    #hints_available: bool = False

    # Raw EXPLAIN ANALYZE JSON (for logging and debugging, not fed to agent)
    explain_plan: Dict[str, Any] = field(default_factory=dict)

    done: bool = False
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SQLState(State):
    """
    Episode metadata — tracks the full history of one optimization run.
    Returned by env.state() at any point during an episode.
    """

    episode_id: Optional[str] = None

    # The original slow query as provided by the user
    original_query: str = ""

    # The query after all rewrites applied so far
    current_query: str = ""

    # Execution time of the original query before any rewrites (ms)
    baseline_time_ms: float = 0.0

    # Execution time of the current rewritten query (ms)
    current_time_ms: float = 0.0

    # Names of rewrites applied so far this episode (for logging)
    rewrites_applied: List[str] = field(default_factory=list)

    # How many steps have been taken this episode
    step_count: int = 0

    # Cumulative reward across all steps this episode
    total_reward: float = 0.0

    # Percentage improvement so far: (baseline - current) / baseline * 100
    improvement_pct: float = 0.0

    # Whether pg_hint_plan was detected on the user's database
    #hints_available: bool = False