# ADR-0028: Indexer Project Auto-Detection from Mount Path

**Status:** PROPOSED
**Date:** 2025-09-12
**Context:** Neural GraphRAG System, Development Workflow

## 1. Problem Statement

The `l9-neural-indexer` container fails to index the correct project due to a flawed project identification mechanism. The core issues are:

1.  **Environment Variable Dependency:** The indexer relies on a `PROJECT_NAME` environment variable to determine which project to index and which Qdrant collection to use.
2.  **Broken Development Workflow:** Development scripts (`start-project-indexer.sh`) are not consistently passing this environment variable into the Docker container.
3.  **Data Misrouting:** As a result, the indexer defaults to `project_name="default"`, causing all indexed data from any project (e.g., `eventfully-yours`) to be incorrectly routed into the `project_default_code` collection.
4.  **Configuration Drift:** Attempts to fix the container's entrypoint script failed because the running container was using a pre-built Docker image with the old, buggy script. The development workflow of editing local files was not synchronized with the container's runtime.

This leads to a confusing and broken development experience where the indexer appears to run but does not correctly isolate or store project data.

## 2. Decision

We will re-architect the `l9-neural-indexer` to be **self-configuring** by **auto-detecting the project name from its mounted file system**.

This decision removes the fragile dependency on the `PROJECT_NAME` environment variable and makes the indexer's behavior intuitive and robust. The container will inspect its `/workspace` directory, identify the single project directory mounted there, and use that directory's name as the `PROJECT_NAME`.

This aligns the container's behavior with the user's intent, as expressed by the `docker run -v` command.

## 3. Architecture Design

The new architecture simplifies the interaction between the host and the container.

### 3.1. Host-Container Interaction

The host is responsible for mounting the target project directory as a subdirectory within the container's `/workspace`.

**Example `docker run` command:**
```bash
# The host mounts the local project into a subdirectory of /workspace
docker run -d \
  -v "/Users/mkr/local-coding/Novel-Projects/eventfully-yours:/workspace/eventfully-yours" \
  --name l9-project-indexer \
  l9-neural-indexer:latest
```

### 3.2. Container-Internal Logic

The container's entrypoint script will perform the following steps upon startup:

1.  Scan the `/workspace` directory.
2.  Identify the single subdirectory (e.g., `eventfully-yours`). This is the project directory.
3.  Extract the name of this directory (`eventfully-yours`) to use as the `PROJECT_NAME`.
4.  Start the `IncrementalIndexer` service, passing it the detected `project_name` and the full path (`/workspace/eventfully-yours`).

This makes the container self-sufficient and removes the need for external configuration via environment variables for project identification.

```
 Host Machine                                       Docker Container
┌───────────────────────────────────┐           ┌──────────────────────────────────────────┐
│                                   │           │                                          │
│ /Users/mkr/.../eventfully-yours/  │   ----->  │ /workspace/eventfully-yours/             │
│                                   │  (mount)  │           ▲                              │
└───────────────────────────────────┘           │           │ 2. Detects project dir & name│
                                                │           │                              │
                                                │ ┌─────────────────┐                    │
                                                │ │ entrypoint.py   │ 1. Starts & scans    │
                                                │ └─────────────────┘                    │
                                                │           │                              │
                                                │           │ 3. Starts Indexer Service    │
                                                │           ▼                              │
                                                │ ┌─────────────────┐                    │
                                                │ │ IndexerService  │                    │
                                                │ │ (project_name=  │                    │
                                                │ │ "eventfully-yours")                    │
                                                │ └─────────────────┘                    │
                                                └──────────────────────────────────────────┘
```

## 4. Implementation Plan

### Phase 1: Simplify `indexer-entrypoint.py` (Immediate)

The primary change is to replace the complex, fragile detection logic in `docker/scripts/indexer-entrypoint.py` with a simple, direct approach.

**New `run` method in `IndexerRunner` class:**
```python
# In docker/scripts/indexer-entrypoint.py

async def run(self):
    """Main entry point for the indexer container."""
    logger.info("Starting Neural Indexer Sidecar Container")

    workspace_path = Path("/workspace")

    # Find the single project directory mounted inside /workspace
    project_dirs = [d for d in workspace_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not project_dirs:
        logger.error("FATAL: No project directory found inside /workspace.")
        logger.error("Mount your project directory as a volume, e.g., -v /path/to/my-project:/workspace/my-project")
        sys.exit(1)

    if len(project_dirs) > 1:
        logger.warning(f"Multiple directories found in /workspace: {[d.name for d in project_dirs]}.")
        logger.warning(f"Using the first one found: {project_dirs[0].name}")

    project_path = project_dirs[0]
    project_name = project_path.name

    logger.info(f"Auto-detected PROJECT_NAME='{project_name}' from mounted directory '{project_path}'")

    initial_index = os.getenv('INITIAL_INDEX', 'true').lower() == 'true'

    self.setup_signal_handlers()
    await self.start_health_server()
    app.state.indexer_runner = self

    try:
        await self.run_indexer(str(project_path), project_name, initial_index)
        await self.shutdown_event.wait()
    except Exception as e:
        logger.error(f"Fatal error in indexer runner: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Neural Indexer Sidecar Container stopped")
```

This new logic is simpler, more robust, and directly addresses the problem by enforcing a clear operational contract.

### Phase 2: Rebuild Docker Image and Update Workflow

1.  **Update Script:** Replace the content of `docker/scripts/indexer-entrypoint.py` with the simplified version.
2.  **Rebuild Image:** The `l9-neural-indexer` Docker image **must** be rebuilt to include the updated entrypoint script.
    ```bash
    docker build -f docker/Dockerfile.indexer -t l9-neural-indexer:latest .
    ```
3.  **Update Start Script:** The `start-project-indexer.sh` script (or equivalent) should be modified to no longer pass the `PROJECT_NAME` environment variable. It only needs to handle the volume mount.

## 5. Benefits

1.  **Robustness:** The indexer is no longer dependent on external environment variables for its core configuration, eliminating a major source of error.
2.  **Simplified Workflow:** Developers can start the indexer for any project with a single, consistent `docker run` command without worrying about setting environment variables.
3.  **Guaranteed Data Isolation:** Data is automatically routed to the correct project-specific Qdrant collection based on the mounted directory name.
4.  **Intuitive Operation:** The container's behavior directly reflects the user's intent as specified by the `-v` volume mount.
5.  **Reduced Configuration:** Eliminates the need for project-specific Docker configurations or environment files for the indexer.

## 6. Consequences

### Positive
- Fixes the persistent data misrouting bug.
- Greatly improves the developer experience.
- Makes the indexer service more modular and self-contained.

### Negative
- Enforces a convention that the project directory must be mounted as a subdirectory of `/workspace`. This is a minor and reasonable constraint.

## 7. Migration Path

1.  **Update `indexer-entrypoint.py`:** Apply the code changes from Phase 1.
2.  **Rebuild Docker Image:** Create a new `l9-neural-indexer:latest` image with the updated script.
3.  **Stop and Remove Old Containers:** Ensure any running indexer containers are stopped and removed to avoid using the old image.
    ```bash
    docker stop l9-project-indexer && docker rm l9-project-indexer
    ```
4.  **Update Start Scripts:** Modify any scripts used to start the indexer to remove the `-e PROJECT_NAME=...` flag.
5.  **Data Migration (Manual):** Manually re-index projects to move data from the `project_default_code` collection to the correct, newly created project-specific collections.

## 8. Decision Outcome

**Approved.** We will implement the auto-detection mechanism in the `indexer-entrypoint.py` script. This change makes the indexer more robust and user-friendly, and it permanently fixes the data misrouting issue by removing the unreliable dependency on environment variables for project identification.
