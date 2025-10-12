"""create tables

Revision ID: bae90cedade9
Revises: b3ea0d252ba7
Create Date: 2025-10-12 21:19:38.558577

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'bae90cedade9'
down_revision: Union[str, None] = 'b3ea0d252ba7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
