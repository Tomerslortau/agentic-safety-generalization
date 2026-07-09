# SuiteCRM Setup Guide - Instance 2 (Port 8081)

This is the **second instance** for parallel trajectory generation.

## Prerequisites

- **Docker**: Required to run the application containers

## Installation Steps

1. **Start the Application**
   - Open a terminal in this directory (`suitecrm_setup_2`)
   - Run the following command:
     ```
     docker compose up
     ```

2. **Load Demo Data**
   - Open a new terminal
   - Navigate to this directory and run:
     ```
     cd suitecrm_setup_2
     docker exec -i suitecrm_setup_2-mariadb2-1 mysql -u bn_suitecrm -pbitnami123 bitnami_suitecrm < init-db/demo_data.sql
     ```
     > Ignore the message: `mysql: Deprecated program name. It will be removed in a future release,`

3. **Access the Application**
   - Open your browser and navigate to: http://localhost:8081
   - Login with `user` as username and `bitnami` as password.

## Running Parallel Generation

**Terminal 1 (Instance 1 - Port 8080):**
```bash
export WA_SUITECRM="http://localhost:8080"
# Run first set of tasks
```

**Terminal 2 (Instance 2 - Port 8081):**
```bash
export WA_SUITECRM="http://localhost:8081"
# Run second set of tasks
```

## Troubleshooting

- Ensure port 8081 is not in use by other applications
- Check Docker logs: `docker compose logs`
- To stop: `docker compose down`
