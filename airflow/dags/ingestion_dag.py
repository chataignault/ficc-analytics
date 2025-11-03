"""
Data Ingestion DAG
This DAG orchestrates the data ingestion pipeline
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# Add src to path to import your modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import your ingestion modules here
# from src.ingestion.your_module import your_function


default_args = {
    'owner': '___',  # TODO: Set owner name
    'depends_on_past': False,
    'email': ['___@___.com'],  # TODO: Set email for alerts
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': ___,  # TODO: Set number of retries (e.g., 2)
    'retry_delay': timedelta(minutes=___),  # TODO: Set retry delay (e.g., 5)
}


def extract_data(**context):
    """
    Extract data from source
    TODO: Implement your data extraction logic
    """
    # Example: Pull data from API, database, or files
    # data = your_extraction_function()
    # Push data to XCom for next task
    # context['task_instance'].xcom_push(key='raw_data', value=data)
    pass


def validate_data(**context):
    """
    Validate extracted data
    TODO: Implement data validation logic
    """
    # Example: Check data quality, schema validation
    # raw_data = context['task_instance'].xcom_pull(key='raw_data')
    # validated = validate(raw_data)
    # context['task_instance'].xcom_push(key='validated_data', value=validated)
    pass


def load_data(**context):
    """
    Load data to destination
    TODO: Implement data loading logic
    """
    # Example: Load to database, data lake, etc.
    # validated_data = context['task_instance'].xcom_pull(key='validated_data')
    # load_to_destination(validated_data)
    pass


with DAG(
    dag_id='ingestion_pipeline',
    default_args=default_args,
    description='___',  # TODO: Add DAG description
    schedule='___',  # TODO: Set schedule (e.g., '@daily', '0 0 * * *', None for manual)
    start_date=datetime(___),  # TODO: Set start date (year, month, day)
    catchup=False,  # Set to True if you want to backfill
    tags=['ingestion', 'etl'],
) as dag:

    # Task 1: Extract data
    extract_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        provide_context=True,
    )

    # Task 2: Validate data
    validate_task = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data,
        provide_context=True,
    )

    # Task 3: Load data
    load_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        provide_context=True,
    )

    # Optional: Add a notification task
    # notify_task = BashOperator(
    #     task_id='notify_completion',
    #     bash_command='echo "Ingestion pipeline completed successfully"',
    # )

    # Define task dependencies
    extract_task >> validate_task >> load_task
    # load_task >> notify_task  # Uncomment if using notification
