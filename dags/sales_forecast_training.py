import sys
import logging
import pandas as pd
from datetime import datetime, timedelta
from airflow.decorators import dag, task

# Include
sys.path.append('/usr/local/airflow/include')
from utils.data_generator import RealisticSalesDataGenerator

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'aniket',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 4),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'catchup': False,
    'schedule': '@weekly'
}

@dag(
    default_args=default_args,
    description='Sales Forecast Training DAG',
    tags=['ml', 'training', 'sales_forecast', 'sales']
)

def sales_forecast_training():
    @task()
    def extract_data_task():
        data_output_dir = '/tmp/sales_data'
        generator = RealisticSalesDataGenerator(
            start_date="2021-01-01",
            end_date="2025-11-04",
        )

        print("Generating realistic sales data...")
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        total_files = sum(len(file_path) for file_path in file_paths.values())
        print(f"Generated {total_files} files")

        for data_type, paths in file_paths.items():
            print(f"{data_type}: {len(paths)} files")

        return {
            "data_output_dir": data_output_dir,
            "files_paths": file_paths,
            "total_files": total_files,
        }

    extract_result = extract_data_task()

sales_forecast_training()