#!/usr/bin/env bash
set -e;

# ─── Guard: n8n history DB non-root user ─────────────────────────────────────
if [ -z "${POSTGRES_NON_ROOT_USER:-}" ] || [ -z "${POSTGRES_NON_ROOT_PASSWORD:-}" ]; then
  echo "SETUP ERROR: POSTGRES_NON_ROOT_USER and POSTGRES_NON_ROOT_PASSWORD must be set!"
  exit 1
fi

# ─── Guard: model repository DB ──────────────────────────────────────────────
if [ -z "${MODEL_DB_NAME:-}" ] || [ -z "${MODEL_DB_USER:-}" ] || [ -z "${MODEL_DB_PASSWORD:-}" ]; then
  echo "SETUP ERROR: MODEL_DB_NAME, MODEL_DB_USER, and MODEL_DB_PASSWORD must all be set!"
  exit 1
fi

# ─── Guard: agent runtime DB ─────────────────────────────────────────────────
if [ -z "${AGENT_RUNTIME_DB_NAME:-}" ] || [ -z "${AGENT_RUNTIME_DB_USER:-}" ] || [ -z "${AGENT_RUNTIME_DB_PASSWORD:-}" ]; then
  echo "SETUP ERROR: AGENT_RUNTIME_DB_NAME, AGENT_RUNTIME_DB_USER, and AGENT_RUNTIME_DB_PASSWORD must all be set!"
  exit 1
fi

# ─── Guard: agent memory DB ──────────────────────────────────────────────────
if [ -z "${AGENT_MEMORY_DB_NAME:-}" ] || [ -z "${AGENT_MEMORY_DB_USER:-}" ] || [ -z "${AGENT_MEMORY_DB_PASSWORD:-}" ]; then
  echo "SETUP ERROR: AGENT_MEMORY_DB_NAME, AGENT_MEMORY_DB_USER, and AGENT_MEMORY_DB_PASSWORD must all be set!"
  exit 1
fi

# ─── n8n history DB: non-root user ───────────────────────────────────────────
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
  CREATE USER ${POSTGRES_NON_ROOT_USER} WITH PASSWORD '${POSTGRES_NON_ROOT_PASSWORD}';
  GRANT ALL PRIVILEGES ON DATABASE ${POSTGRES_DB} TO ${POSTGRES_NON_ROOT_USER};
  GRANT CREATE ON SCHEMA public TO ${POSTGRES_NON_ROOT_USER};
EOSQL

# ─── Model repository: DB + user ─────────────────────────────────────────────
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
  CREATE DATABASE ${MODEL_DB_NAME};
  CREATE USER ${MODEL_DB_USER} WITH PASSWORD '${MODEL_DB_PASSWORD}';
  GRANT ALL PRIVILEGES ON DATABASE ${MODEL_DB_NAME} TO ${MODEL_DB_USER};
EOSQL

# ─── Model repository: schema + tables ───────────────────────────────────────
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$MODEL_DB_NAME" <<-EOSQL
  CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

  CREATE TABLE models (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    name            TEXT,
    source_workflow TEXT,
    status          TEXT DEFAULT 'draft',
    nn_taxonomy     JSONB NOT NULL,
    extra_meta      JSONB,
    session_id      TEXT
  );

  CREATE TABLE model_benchmarks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id        UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    benchmark_name  TEXT NOT NULL,
    dataset         TEXT,
    run_label       TEXT,
    metrics         JSONB NOT NULL,
    config          JSONB,
    model_remarks   JSONB
  );

  CREATE TABLE IF NOT EXISTS plateau_history (
    id                  BIGSERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    model_test_accuracy DOUBLE PRECISION,
    payload             JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
  );

  CREATE INDEX IF NOT EXISTS ix_plateau_history_session_created_at
    ON plateau_history(session_id, created_at DESC, id DESC);

  CREATE INDEX IF NOT EXISTS ix_plateau_history_session_accuracy
    ON plateau_history(session_id, model_test_accuracy)
    WHERE model_test_accuracy IS NOT NULL;

  GRANT CREATE, USAGE ON SCHEMA public TO ${MODEL_DB_USER};
  GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ${MODEL_DB_USER};
  GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ${MODEL_DB_USER};
  ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ${MODEL_DB_USER};
  ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO ${MODEL_DB_USER};
EOSQL

# ─── Agent runtime: DB + user ────────────────────────────────────────────────
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
  DO \$\$
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '${AGENT_RUNTIME_DB_NAME}') THEN
      CREATE DATABASE ${AGENT_RUNTIME_DB_NAME};
    END IF;
  END
  \$\$;

  DO \$\$
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${AGENT_RUNTIME_DB_USER}') THEN
      CREATE USER ${AGENT_RUNTIME_DB_USER} WITH PASSWORD '${AGENT_RUNTIME_DB_PASSWORD}';
    END IF;
  END
  \$\$;

  GRANT ALL PRIVILEGES ON DATABASE ${AGENT_RUNTIME_DB_NAME} TO ${AGENT_RUNTIME_DB_USER};
EOSQL

# ─── Agent memory: DB + user ─────────────────────────────────────────────────
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
  DO \$\$
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '${AGENT_MEMORY_DB_NAME}') THEN
      CREATE DATABASE ${AGENT_MEMORY_DB_NAME};
    END IF;
  END
  \$\$;

  DO \$\$
  BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${AGENT_MEMORY_DB_USER}') THEN
      CREATE USER ${AGENT_MEMORY_DB_USER} WITH PASSWORD '${AGENT_MEMORY_DB_PASSWORD}';
    END IF;
  END
  \$\$;

  GRANT ALL PRIVILEGES ON DATABASE ${AGENT_MEMORY_DB_NAME} TO ${AGENT_MEMORY_DB_USER};
EOSQL

# ─── Agent runtime: schema privileges for LangGraph + app tables ─────────────
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$AGENT_RUNTIME_DB_NAME" <<-EOSQL
  CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

  GRANT CREATE, USAGE ON SCHEMA public TO ${AGENT_RUNTIME_DB_USER};

  GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ${AGENT_RUNTIME_DB_USER};
  GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ${AGENT_RUNTIME_DB_USER};

  ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ${AGENT_RUNTIME_DB_USER};

  ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO ${AGENT_RUNTIME_DB_USER};
EOSQL

# ─── Agent memory: schema privileges for episodic/summary tables ─────────────
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$AGENT_MEMORY_DB_NAME" <<-EOSQL
  CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

  CREATE TABLE IF NOT EXISTS agent_session_states (
    state_id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    summary_text TEXT,
    summary_state TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
  );

  CREATE TABLE IF NOT EXISTS agent_session_episodes (
    episode_id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    agent TEXT NOT NULL,
    message TEXT NOT NULL,
    episode_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_agent_session_episodes_session_step UNIQUE (session_id, step_index)
  );

  CREATE INDEX IF NOT EXISTS ix_agent_session_states_session_id
    ON agent_session_states(session_id);

  CREATE INDEX IF NOT EXISTS ix_agent_session_states_created_at
    ON agent_session_states(created_at);

  CREATE INDEX IF NOT EXISTS ix_agent_session_episodes_session_id
    ON agent_session_episodes(session_id);

  CREATE INDEX IF NOT EXISTS ix_agent_session_episodes_step_index
    ON agent_session_episodes(step_index);

  CREATE INDEX IF NOT EXISTS ix_agent_session_episodes_agent
    ON agent_session_episodes(agent);

  CREATE INDEX IF NOT EXISTS ix_agent_session_episodes_created_at
    ON agent_session_episodes(created_at);

  CREATE OR REPLACE FUNCTION agent_session_episodes_assign_step_index()
  RETURNS trigger
  LANGUAGE plpgsql
  AS \$\$
  BEGIN
    IF NEW.step_index IS NULL THEN
      SELECT COALESCE(MAX(step_index), 0) + 1
      INTO NEW.step_index
      FROM agent_session_episodes
      WHERE session_id = NEW.session_id;
    END IF;

    RETURN NEW;
  END;
  \$\$;

  DROP TRIGGER IF EXISTS trg_agent_session_episodes_assign_step_index ON agent_session_episodes;

  CREATE TRIGGER trg_agent_session_episodes_assign_step_index
  BEFORE INSERT ON agent_session_episodes
  FOR EACH ROW
  EXECUTE FUNCTION agent_session_episodes_assign_step_index();

  GRANT CREATE, USAGE ON SCHEMA public TO ${AGENT_MEMORY_DB_USER};

  GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ${AGENT_MEMORY_DB_USER};
  GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ${AGENT_MEMORY_DB_USER};

  ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO ${AGENT_MEMORY_DB_USER};

  ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO ${AGENT_MEMORY_DB_USER};
EOSQL

echo "SETUP INFO: n8n, model_repository, agent_runtime, and agent_memory databases/users initialized successfully."