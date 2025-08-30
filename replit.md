# Restaurant Sales Analytics Platform

## Overview

This is a comprehensive restaurant analytics platform that provides ML-powered sales forecasting, performance reporting, and business intelligence for restaurant chains. The system integrates data from multiple food delivery platforms (Grab, Gojek), combines it with weather data and external factors, and generates detailed analytical reports with actionable insights.

The platform supports both static dataset analysis (SQLite-based) and live API integration (PostgreSQL-based) for real-time analytics. It includes a FastAPI backend for data serving, a Streamlit frontend for visualization, and sophisticated ML models for sales prediction and anomaly detection.

## Recent Changes

### v0.3.0-live-db (stable) - August 30, 2025
- **Fixed MySQL/MariaDB compatibility**: Replaced SQLite `strftime()` with `DATE_FORMAT()` for live databases
- **Database dialect detection**: Added automatic detection based on `DATABASE_URL` environment variable  
- **SQL placeholder standardization**: All queries use `sa_text()` wrapper with named parameters (`:start`, `:end`, `:rid`)
- **Dual database support**: SQLite for demo mode, MySQL/PostgreSQL for production
- **Report generation**: Stable text and basic reports working with live database connections
- **API endpoints**: `/health`, `/report-text`, `/restaurants` fully functional with MySQL backend

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Data Layer Architecture
The system implements a flexible dual-mode data architecture:
- **Legacy Mode**: SQLite database with CSV datasets for offline analysis
- **Live Mode**: PostgreSQL database connected to external restaurant APIs for real-time data
- **Data Adapter Pattern**: `DataAdapter` class provides unified interface regardless of underlying storage

### ETL Pipeline Design
- **API Client**: Fetches restaurant statistics from live APIs with retry logic and chunked processing
- **Feature Engineering**: Combines sales data with weather, holidays, tourist flow, and operational metrics
- **Data Views**: Creates analytical views (`daily_facts`) for optimized reporting queries
- **Caching Strategy**: Weather data cached locally to minimize API calls

### ML Architecture
- **Multi-Model Training**: LightGBM and RandomForest regressors with champion selection based on MAE
- **SHAP Integration**: Explainable AI for factor attribution and business insights
- **Feature Store**: Comprehensive feature engineering including temporal, lag, and external factors
- **Artifact Management**: Versioned model artifacts with metadata tracking

### API Architecture
- **FastAPI Backend**: RESTful endpoints for health checks, ML predictions, and report generation
- **Report Generation**: Text-based reports with structured sections and business metrics
- **Error Handling**: Graceful degradation when ML models or databases unavailable

### Frontend Architecture
- **Streamlit Dashboard**: Interactive web interface for report visualization
- **API Integration**: Communicates with FastAPI backend for data and predictions
- **Fallback Mechanisms**: Works with demo data when live systems unavailable

### Database Schema Design
- **Restaurant Mapping**: Centralized restaurant metadata and ID management
- **Time-Series Storage**: Optimized storage for daily sales, operations, and marketing metrics
- **Analytical Views**: Pre-computed aggregations for fast reporting

### Reporting Engine
- **Multi-Language Support**: Reports generated in Russian with structured formatting
- **Section-Based Architecture**: Modular report sections (sales, marketing, operations, quality, ML insights)
- **Snapshot Testing**: Automated testing for report structure and content consistency

## External Dependencies

### Third-Party APIs
- **Restaurant Stats API**: Live data integration from `http://5.187.7.140:3000` for real-time metrics
- **Open-Meteo Weather API**: Historical and forecast weather data for external factor analysis
- **Nager.Date Holidays API**: Public holiday data for Indonesia and international observances

### Database Systems
- **PostgreSQL**: Primary production database for live API integration
- **SQLite**: Fallback database for offline analysis and development
- **SQLAlchemy ORM**: Database abstraction layer for cross-platform compatibility

### ML and Analytics Stack
- **LightGBM/RandomForest**: Primary ML models for sales forecasting
- **SHAP**: Model explainability and feature attribution
- **Pandas/NumPy**: Data processing and numerical computations
- **Scikit-learn**: ML pipeline and preprocessing utilities

### Delivery Platform Integration
- **Grab API**: Food delivery platform data integration
- **Gojek API**: Food delivery platform data integration
- **Platform-specific metrics**: Orders, ratings, delivery times, marketing spend

### Infrastructure Dependencies
- **FastAPI/Uvicorn**: High-performance async web framework
- **Streamlit**: Interactive dashboard framework
- **Requests/HTTPX**: HTTP client libraries for API communication
- **Python-dotenv**: Environment variable management