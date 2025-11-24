# Sales Forecasting
## Project Overview
This project builds a Sales Forecasting system that predicts future sales volumes using historical data. The model provides forward-looking sales visibility that can improve revenue planning accuracy by 3–8%, reduce inventory misallocation by 10–25%, and lower supply-chain disruptions by 8–12%, resulting in significant annual financial impact for mid-sized businesses.

## Business Impact
### I. Inventory Cost Optimization
Accurate demand forecasting helps reduce excess stock, minimize storage costs, and prevent product expiry or obsolescence.
* 10–25% reduction in inventory holding costs
* For a company spending $5M/year on inventory, this translates to: $500,000 – $1,250,000 in annual cost saving

### II. Increase in Revenue Through Better Demand Fulfillment
With fewer stockouts and more accurate supply planning, companies are able to meet customer demand consistently and avoid missed sales.
* 3–8% increase in annual revenue
* For a business with $20M/year revenue, this results in: $600,000 – $1,600,000 in additional yearly revenue

### III. Supply Chain & Logistics Efficiency
Better predictions reduce emergency orders, lower transportation costs, and improve warehouse utilization.
* 8–12% reduction in logistics and operational costs
* For a company spending $2M/year on supply chain operations: $160,000 – $240,000 in yearly savings

## System Architecture
<img width="1201" height="521" alt="Sales Forecasting" src="https://github.com/user-attachments/assets/79678f5b-c696-4bc0-bb77-bb358336f2a4" />

## Technical Details
The Sales Forecasting solution is built using an end-to-end, production-grade ML pipeline designed for enterprise use — from data ingestion through inference and monitoring.

| Category | Description |
|----------|-------------|
| **Workflow Orchestration** | Managed via **Apache Airflow** for scheduling, monitoring, and automating all pipeline steps. |
| **Artifact & Model Storage** | Uses **S3-compatible storage** for model binaries, feature artifacts, and logs. |
| **Experiment Tracking** | Integrated with **MLflow** for experiment tracking, model registry, and reproducibility. |
| **Containerisation & Deployment** | Deployed using **Docker / Docker-Compose** to ensure consistent and portable environments. |
| **Data Preprocessing** | Cleans and aggregates historical sales data; handles missing values, parses dates, builds time-series index, and generates lag, rolling, seasonal, and holiday features. |
| **Model Ensemble** | Uses a mix of **XGBoost**, **LightGBM**, and **Prophet** to capture linear, nonlinear, and seasonal sales patterns. |
| **Hyperparameter Optimisation** | Tuned using **Optuna** across time-series cross-validation windows. |
| **Validation Strategy** | Employs **expanding-window** time-series train/validation splits to simulate real production forecasting conditions. |
| **Forecasting Capability** | Supports **multi-horizon weekly sales forecasting**. |
| **User Interface** | Web-based **Streamlit** dashboard for uploading data and generating what-if forecast scenarios. |
| **Visual Analytics** | Produces charts for actual vs predicted sales, residual patterns, feature importance, confidence intervals, and performance summaries. |
| **Performance Metrics** | Evaluated using **MAE**, **RMSE**, and **MAPE** for model selection and business impact estimation. |
| **Monitoring** | Logging and alerting mechanisms notify when forecast errors exceed thresholds, enabling retraining or human review. |

## Forecasted Sales Output
<img width="1490" height="831" alt="Screenshot 2025-11-22 at 1 14 11 AM" src="https://github.com/user-attachments/assets/6edc788a-2d9a-433d-95eb-900c39bdda85" />

## License
This project is licensed under the MIT License.
