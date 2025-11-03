# Apache Airflow Setup

This directory contains the Apache Airflow setup for orchestrating ETL and analytics pipelines.

## Directory Structure

```
airflow/
├── dags/                    # DAG definitions
│   ├── ingestion_dag.py    # Data ingestion pipeline
│   └── analytics_dag.py    # Analytics and transformation pipeline
├── plugins/                 # Custom operators, hooks, sensors
├── config/                  # Configuration files
└── logs/                    # Airflow logs (auto-generated)
```

## Installation

### 1. Install Apache Airflow

```bash
# Set Airflow home (optional, defaults to ~/airflow)
export AIRFLOW_HOME=$(pwd)/airflow

# Install Airflow with constraints for your Python version
pip install apache-airflow==2.8.0 --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.0/constraints-3.11.txt"

# Install providers as needed
pip install apache-airflow-providers-postgres
pip install apache-airflow-providers-http
pip install apache-airflow-providers-sqlite
```

### 2. Initialize Airflow Database

```bash
airflow db init
```

### 3. Create Admin User

```bash
airflow users create \
    --username admin \
    --firstname <Your First Name> \
    --lastname <Your Last Name> \
    --role Admin \
    --email <your-email@example.com>
```

You'll be prompted to set a password.

## Running Airflow

### Start the Webserver (Terminal 1)

```bash
airflow webserver --port 8080
```

Access the UI at: http://localhost:8080

### Start the Scheduler (Terminal 2)

```bash
airflow scheduler
```

## Configuration

### Setting up Connections

1. Navigate to Admin > Connections in the Airflow UI
2. Add connections for your data sources:
   - Database connections (PostgreSQL, DuckDB, etc.)
   - API connections
   - Cloud storage connections

### Setting up Variables

1. Navigate to Admin > Variables in the Airflow UI
2. Add environment-specific variables:
   - Data paths
   - API keys (use Secrets for sensitive data)
   - Configuration parameters

## DAG Templates

### ingestion_dag.py

Template for data ingestion pipeline with:
- Data extraction from sources
- Data validation
- Data loading to destinations

**TODO items to complete:**
- Set owner and email
- Configure retry settings
- Implement extraction logic
- Implement validation logic
- Implement loading logic
- Set schedule interval
- Set start date

### analytics_dag.py

Template for analytics pipeline with:
- Data fetching from sources
- Data cleaning and preprocessing
- Data transformation
- Analytics computation
- Results saving
- Quality checks with branching logic

**TODO items to complete:**
- Set owner and email
- Configure retry settings
- Implement data fetching
- Implement cleaning logic
- Implement transformation logic
- Implement analytics computation
- Implement quality checks
- Set schedule interval
- Set start date

## Best Practices

1. **Task Design**
   - Keep tasks atomic and idempotent
   - Use XCom sparingly (for small data only)
   - Implement proper error handling

2. **DAG Design**
   - Use meaningful task_ids and dag_ids
   - Set appropriate retry policies
   - Use SLAs and timeouts
   - Add tags for organization

3. **Dependencies**
   - Keep your ETL code in src/ directory
   - Import functions into DAGs (don't write business logic in DAGs)
   - Use plugins/ for custom operators

4. **Testing**
   - Test DAGs locally before deploying
   - Use `airflow dags test` command
   - Validate task dependencies

5. **Monitoring**
   - Set up email alerts
   - Monitor task duration
   - Review logs regularly

## Useful Commands

```bash
# List all DAGs
airflow dags list

# Test a specific DAG
airflow dags test <dag_id> <execution_date>

# Test a specific task
airflow tasks test <dag_id> <task_id> <execution_date>

# Trigger a DAG manually
airflow dags trigger <dag_id>

# Pause/Unpause a DAG
airflow dags pause <dag_id>
airflow dags unpause <dag_id>

# View logs
airflow tasks logs <dag_id> <task_id> <execution_date>
```

## Docker Deployment (Optional)

For production deployment, consider using Docker Compose:

```yaml
version: '3'
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow

  webserver:
    image: apache/airflow:2.8.0
    depends_on:
      - postgres
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
    ports:
      - "8080:8080"
```

## Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [DAG Writing Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/howto/writing-a-dag.html)
