"""SQLAlchemy ORM models for PostgreSQL.

Tables:
- restaurants
- sales
- operations
- marketing
- weather
- holidays
"""

from __future__ import annotations

from datetime import date
from sqlalchemy import (
    Column,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    location_type = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_restaurants_name", "name"),
    )


class Sale(Base):
    __tablename__ = "sales"

    id = Column(Integer, primary_key=True, autoincrement=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    platform = Column(Text, nullable=True)
    total_sales = Column(Float, nullable=True)
    orders_count = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_sales_restaurant_date", "restaurant_id", "date"),
    )


class Operation(Base):
    __tablename__ = "operations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    accepting_time = Column(Float, nullable=True)
    delivery_time = Column(Float, nullable=True)
    preparation_time = Column(Float, nullable=True)
    rating = Column(Float, nullable=True)
    repeat_customers = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_operations_restaurant_date", "restaurant_id", "date"),
    )


class Marketing(Base):
    __tablename__ = "marketing"

    id = Column(Integer, primary_key=True, autoincrement=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    ads_spend = Column(Float, nullable=True)
    roas = Column(Float, nullable=True)
    impressions = Column(Integer, nullable=True)
    clicks = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_marketing_restaurant_date", "restaurant_id", "date"),
    )


class Weather(Base):
    __tablename__ = "weather"

    id = Column(Integer, primary_key=True, autoincrement=True)
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    temp = Column(Float, nullable=True)
    rain = Column(Float, nullable=True)
    wind = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_weather_restaurant_date", "restaurant_id", "date"),
    )


class Holiday(Base):
    __tablename__ = "holidays"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    holiday_name = Column(Text, nullable=True)
    region = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_holidays_date", "date"),
    )