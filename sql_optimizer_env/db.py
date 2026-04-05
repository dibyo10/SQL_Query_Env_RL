# ─────────────────────────────────────────────────────────────────────────────
# server/db.py
#
# Database layer — all raw Postgres interaction lives here.
# The environment class calls these methods; it never touches psycopg2 directly.
#
# Responsibilities:
#   - Connect to the user's database using their db_url
#   - Probe for pg_hint_plan at connect time
#   - Run EXPLAIN ANALYZE and return structured JSON
#   - Run correctness checksums
#   - Read pg_catalog and information_schema for schema discovery
# ─────────────────────────────────────────────────────────────────────────────

import os
from typing import Dict, List, Optional, Tuple

import psycopg2
import psycopg2.extras


class PostgreSQLExecutor:
    """
    Manages one Postgres connection per optimization episode.

    Instantiated at reset() time with the user's db_url.
    Destroyed (connection closed) at episode end or on error.
    """

    def __init__(self, db_url: str):
        self.db_url = db_url
        self.conn: Optional[psycopg2.extensions.connection] = None

        # Set by _check_extension_installed() immediately after connect.
        # Controls whether Tier 2 hint actions appear in legal_actions.
        self.hints_enabled: bool = False

        self._query_timeout_ms = int(
            os.getenv("QUERY_TIMEOUT_MS", "30000")
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CONNECTION
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Open a connection to the user's Postgres database.
        Immediately probes for pg_hint_plan.
        Raises ConnectionError if the database cannot be reached.
        """
        try:
            self.conn = psycopg2.connect(
                self.db_url,
                connect_timeout=10,
                options=f"-c statement_timeout={self._query_timeout_ms}",
            )
            self.conn.autocommit = True
        except psycopg2.OperationalError as e:
            raise ConnectionError(
                f"Could not connect to database. "
                f"Check that your db_url is correct and the database is reachable.\n"
                f"Original error: {e}"
            )

        # Probe for pg_hint_plan immediately after connecting.
        # This single probe drives the entire Tier 1 / Tier 2 split.
        self.hints_enabled = self._check_extension_installed("pg_hint_plan")

    def close(self) -> None:
        """Close the connection. Called at episode end."""
        if self.conn and not self.conn.closed:
            self.conn.close()

    def _check_extension_installed(self, extension_name: str) -> bool:
        """
        Probe whether a Postgres extension is installed on the user's database.

        Queries pg_extension — the system catalog of installed extensions.
        Returns True only if the extension is present AND loaded.

        This is the gate for Tier 2 actions:

            if self.hints_enabled:
                # add index/join hint actions to legal_actions
            else:
                # skip hint actions entirely — they would produce 0.0 reward
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT 1 FROM pg_extension WHERE extname = %s",
                (extension_name,)
            )
            result = cursor.fetchone()
            installed = result is not None

            if installed:
                print(
                    f"[db] pg_hint_plan detected — "
                    f"Tier 2 hint actions enabled"
                )
            else:
                print(
                    f"[db] pg_hint_plan NOT detected — "
                    f"Tier 2 hint actions disabled. "
                    f"Structural rewrites (Tier 1) still available. "
                    f"To enable hints: https://github.com/ossc-db/pg_hint_plan"
                )

            return installed

        except Exception as e:
            # If the probe itself errors, assume not installed.
            # Never crash the environment over a missing extension.
            print(f"[db] pg_hint_plan probe failed ({e}) — assuming not installed")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # EXPLAIN ANALYZE
    # ─────────────────────────────────────────────────────────────────────────

    def get_explain_plan(self, sql: str) -> Dict:
        """
        Run EXPLAIN (FORMAT JSON, ANALYZE) against the user's database.

        Postgres generates all cost numbers, actual times, and row counts.
        psycopg2 auto-parses the JSON response into a Python dict.
        We unwrap the outer list Postgres wraps it in.

        Returns the plan dict, or an empty dict if execution fails.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"EXPLAIN (FORMAT JSON, ANALYZE) {sql}")
            result = cursor.fetchone()
            if result and result[0]:
                return result[0][0]   # unwrap [{ "Plan": {...} }]
            return {}
        except Exception as e:
            print(f"[db] EXPLAIN ANALYZE failed: {e}")
            return {}

    def measure_execution_time(self, sql: str) -> float:
        """
        Get actual execution time in milliseconds from EXPLAIN ANALYZE.
        Returns a large fallback value if the query fails.
        """
        plan = self.get_explain_plan(sql)
        return plan.get("Execution Time", 999999.0)

    # ─────────────────────────────────────────────────────────────────────────
    # CORRECTNESS VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────

    def verify_correctness(self, original_sql: str, rewritten_sql: str) -> bool:
        """
        Run both queries and compare result set hashes.

        Uses md5(array_agg(t.*)::text) — converts all rows into a text
        representation and hashes it. If the hashes match, the rewrite
        returns identical data.

        Returns False (and gives -1.0 reward) if:
          - hashes differ (rewrite changed the results)
          - either query fails to execute
          - the hash query itself errors
        """
        try:
            cursor = self.conn.cursor()

            cursor.execute(
                f"SELECT md5(array_agg(t.*)::text) FROM ({original_sql}) t"
            )
            hash_original = cursor.fetchone()[0]

            cursor.execute(
                f"SELECT md5(array_agg(t.*)::text) FROM ({rewritten_sql}) t"
            )
            hash_rewritten = cursor.fetchone()[0]

            return hash_original == hash_rewritten

        except Exception as e:
            print(f"[db] Correctness check failed: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # SCHEMA DISCOVERY
    # ─────────────────────────────────────────────────────────────────────────

    def get_available_indexes(self) -> Dict[str, List[str]]:
        """
        Read all user-defined indexes from pg_catalog.

        Returns a dict mapping table name → list of index names.
        Only includes indexes on user tables (excludes pg_* system tables).

        Used to populate params for Tier 2 index hint actions.
        The agent can only hint indexes that actually exist.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT
                    t.relname  AS table_name,
                    i.relname  AS index_name
                FROM
                    pg_class     t
                    JOIN pg_index  ix ON t.oid = ix.indrelid
                    JOIN pg_class  i  ON i.oid = ix.indexrelid
                    JOIN pg_namespace n ON t.relnamespace = n.oid
                WHERE
                    t.relkind = 'r'
                    AND n.nspname NOT IN ('pg_catalog', 'information_schema')
                    AND NOT ix.indisprimary
                ORDER BY
                    t.relname, i.relname
            """)

            indexes: Dict[str, List[str]] = {}
            for table_name, index_name in cursor.fetchall():
                if table_name not in indexes:
                    indexes[table_name] = []
                indexes[table_name].append(index_name)

            return indexes

        except Exception as e:
            print(f"[db] Index discovery failed: {e}")
            return {}

    def get_column_names(self, table_name: str) -> List[str]:
        """
        Read column names for a table from information_schema.
        Used by the replace_select_star action.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
                """,
                (table_name,)
            )
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"[db] Column discovery failed for {table_name}: {e}")
            return []

    def get_table_stats(self, table_name: str) -> Dict:
        """
        Read estimated row count and page count from pg_class.
        Used to enrich the observation vector.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT
                    reltuples::bigint AS estimated_rows,
                    relpages          AS pages
                FROM pg_class
                WHERE relname = %s
                """,
                (table_name,)
            )
            row = cursor.fetchone()
            if row:
                return {"estimated_rows": row[0], "pages": row[1]}
            return {}
        except Exception as e:
            print(f"[db] Table stats failed for {table_name}: {e}")
            return {}