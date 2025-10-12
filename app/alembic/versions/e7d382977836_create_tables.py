"""create tables

Revision ID: e7d382977836
Revises: b3ea0d252ba7
Create Date: 2025-10-12 21:27:50.514140

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7d382977836'
down_revision: Union[str, None] = 'b3ea0d252ba7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
