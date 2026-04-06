# sql_optimizer_env/server/app.py

import os
from openenv_core import create_fastapi_app
from sql_optimizer_env import SQLOptimizerEnvironment

# Database config from environment variables
# These are set in the Dockerfile or docker run command
DB_CONFIG = {
    "host":     os.getenv("POSTGRES_HOST", "localhost"),
    "port":     int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "tpch"),
    "user":     os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
}

env = SQLOptimizerEnvironment(db_config=DB_CONFIG, max_steps=10)
app = create_fastapi_app(env)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()