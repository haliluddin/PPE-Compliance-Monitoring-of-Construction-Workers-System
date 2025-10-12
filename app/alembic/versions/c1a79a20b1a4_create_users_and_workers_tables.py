"""create users and workers tables

Revision ID: c1a79a20b1a4
Revises: e7d382977836
Create Date: 2025-10-12 21:35:04.085593

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c1a79a20b1a4'
down_revision: Union[str, None] = 'e7d382977836'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
