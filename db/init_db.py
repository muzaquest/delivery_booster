"""Create all tables in the configured PostgreSQL database."""

from __future__ import annotations

from sqlalchemy import text

from db.models import Base
from db.session import get_engine


def main() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(text("SELECT 1"))
    print("PostgreSQL tables created.")


if __name__ == "__main__":
    main()