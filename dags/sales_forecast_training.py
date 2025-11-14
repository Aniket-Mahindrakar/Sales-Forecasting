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

    @task()
    def validate_data_task(extract_result):
        file_paths = extract_result['files_paths']
        total_rows = 0
        issues_found = []
        logger.info(f"Validate {len(file_paths["sales"])} sales files...")
        for i, sales_file in enumerate(file_paths["sales"][:10]):
            df = pd.read_parquet(sales_file)
            if i == 0:
                logger.info(f"Sales data columns: {df.columns.tolist()}")

            if df.empty:
                issues_found.append(f"Sales file {sales_file} is empty")
                continue

            required_columns = [
                "date",
                "store_id",
                "product_id",
                "quantity_sold",
                "revenue"
            ]

            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                issues_found.append(f"Sales file {sales_file} is missing columns: {missing_cols}")
                continue

            total_rows += len(df)
            if df["quantity_sold"].min() < 0:
                issues_found.append(f"Sales file {sales_file} has negative quantity sold")

            if df["revenue"].min() < 0:
                issues_found.append(f"Sales file {sales_file} has negative revenue")

            for data_type in ["promotions", "customer_traffic", "store_events"]:
                if data_type in file_paths and file_paths[data_type]: # Checking for data types
                    sample_file = file_paths[data_type][0]
                    df = pd.read_parquet(sample_file)
                    logger.info(f"{data_type} data shape: {df.shape}")
                    logger.info(f"{data_type} data columns: {df.columns.tolist()}")

            validation_summary = {
                "total_files_validated": len(file_paths["sales"][:10]),
                "total_rows": total_rows,
                "issues_found": issues_found,
                "issues_count": issues_found[:5],
                "file_paths": file_paths
            }

            if issues_found:
                logger.error(f"Validation summary: {validation_summary}")
                for issue in issues_found:
                    logger.error(issue)
                raise Exception("Validation issues found")
            else:
                logger.info(f"Validation summary: {validation_summary}")

            return validation_summary

    extract_result = extract_data_task()
    validation_summary = validate_data_task(extract_result)

sales_forecast_training()