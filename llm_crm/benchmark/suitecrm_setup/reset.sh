#!/bin/bash
# Reset SuiteCRM database and environment to clean state

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Resetting SuiteCRM Database and Environment"
echo "=========================================="

# Step 1: Stop and remove containers and volumes
echo ""
echo "Step 1: Stopping containers and removing volumes..."
docker compose down -v

# Step 2: Start containers fresh
echo ""
echo "Step 2: Starting fresh containers..."
docker compose up -d

# Step 3: Wait for database to be ready
echo ""
echo "Step 3: Waiting for database to be ready..."
sleep 10

# Check if mariadb container is running
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker exec suitecrm_setup-mariadb-1 mysqladmin ping -h localhost --silent 2>/dev/null; then
        echo "Database is ready!"
        break
    fi
    echo "Waiting for database... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "ERROR: Database did not become ready in time"
    exit 1
fi

# Step 4: Wait for SuiteCRM to initialize and create tables
echo ""
echo "Step 4: Waiting for SuiteCRM to initialize database tables..."
echo "This may take 2-3 minutes. SuiteCRM needs to complete its setup first."
MAX_RETRIES=90
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Check if multiple critical tables exist (users, contacts, accounts, email_addr_bean_rel)
    USERS_EXISTS=$(docker exec suitecrm_setup-mariadb-1 mysql -u bn_suitecrm -pbitnami123 bitnami_suitecrm -e "SHOW TABLES LIKE 'users';" 2>/dev/null | grep -c "users" || echo "0")
    CONTACTS_EXISTS=$(docker exec suitecrm_setup-mariadb-1 mysql -u bn_suitecrm -pbitnami123 bitnami_suitecrm -e "SHOW TABLES LIKE 'contacts';" 2>/dev/null | grep -c "contacts" || echo "0")
    ACCOUNTS_EXISTS=$(docker exec suitecrm_setup-mariadb-1 mysql -u bn_suitecrm -pbitnami123 bitnami_suitecrm -e "SHOW TABLES LIKE 'accounts';" 2>/dev/null | grep -c "accounts" || echo "0")
    EMAIL_REL_EXISTS=$(docker exec suitecrm_setup-mariadb-1 mysql -u bn_suitecrm -pbitnami123 bitnami_suitecrm -e "SHOW TABLES LIKE 'email_addr_bean_rel';" 2>/dev/null | grep -c "email_addr_bean_rel" || echo "0")
    
    # Convert to integers for arithmetic
    USERS_EXISTS=${USERS_EXISTS:-0}
    CONTACTS_EXISTS=${CONTACTS_EXISTS:-0}
    ACCOUNTS_EXISTS=${ACCOUNTS_EXISTS:-0}
    EMAIL_REL_EXISTS=${EMAIL_REL_EXISTS:-0}
    
    TABLES_EXIST=$((USERS_EXISTS + CONTACTS_EXISTS + ACCOUNTS_EXISTS + EMAIL_REL_EXISTS))
    
    if [ "$TABLES_EXIST" -ge 4 ]; then
        echo "SuiteCRM tables are ready! (found all 4 critical tables)"
        break
    fi
    echo "Waiting for SuiteCRM to create tables... ($RETRY_COUNT/$MAX_RETRIES) (found $TABLES_EXIST/4 critical tables)"
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "WARNING: SuiteCRM tables may not be ready yet. Skipping demo data load."
    echo "You may need to access http://localhost:8080 first to complete SuiteCRM setup."
    echo "After SuiteCRM is fully initialized, you can manually load demo data with:"
    echo "  docker exec -i suitecrm_setup-mariadb-1 mysql -u bn_suitecrm -pbitnami123 bitnami_suitecrm < init-db/demo_data.sql"
    exit 0
fi

# Step 5: Wait additional 15 seconds before loading demo data
echo ""
echo "Step 5: Waiting 15 seconds before loading demo data..."
sleep 15

# Step 6: Load demo data
echo ""
echo "Step 6: Loading demo data..."
docker exec -i suitecrm_setup-mariadb-1 mysql -u bn_suitecrm -pbitnami123 bitnami_suitecrm < init-db/demo_data.sql

# echo ""
echo "=========================================="
echo "Reset complete!"
echo "=========================================="
# echo ""
# echo "SuiteCRM is available at: http://localhost:8080"
# echo "Login credentials:"
# echo "  Username: user"
# echo "  Password: bitnami"
# echo ""
# echo "To view logs: docker compose logs -f"
# echo "To stop: docker compose down"
# echo ""
