# sql_optimizer_env/client.py
from typing import Any, Dict
from openenv.core import EnvClient
from openenv.core.env_server.types import StepResponse
from sql_optimizer_env.models import SQLAction, SQLObservation, SQLState


class SQLOptimizerEnv(EnvClient[SQLAction, SQLObservation, SQLState]):

    def _step_payload(self, action: SQLAction) -> Dict[str, Any]:
        return {
            "action_id": action.action_id,
            "params":    action.params,
            "metadata":  action.metadata,
        }

    def _parse_result(self, payload: Dict) -> StepResponse:
        obs = payload.get("observation", {})
        return StepResponse(
            observation=SQLObservation(
                current_query=obs.get("current_query", ""),
                observation_vector=obs.get("observation_vector", []),
                legal_actions=obs.get("legal_actions", []),
                explain_plan=obs.get("explain_plan", {}),
                #hints_available=obs.get("hint", False)
                done=obs.get("done", False),
                reward=obs.get("reward", 0.0),
                metadata=obs.get("metadata", {}),
            ),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )
    

    def _parse_state(self, payload: Dict) -> SQLState:
        return SQLState(
            episode_id=payload.get("episode_id"),
            original_query=payload.get("original_query", ""),
            current_query=payload.get("current_query", ""),
            baseline_time_ms=payload.get("baseline_time_ms", 0.0),
            current_time_ms=payload.get("current_time_ms", 0.0),
            rewrites_applied=payload.get("rewrites_applied", []),
            step_count=payload.get("step_count", 0),
            total_reward=payload.get("total_reward", 0.0),
            improvement_pct=payload.get("improvement_pct", 0.0),
            #hints_available=payload.get("hints_available", False)
        )