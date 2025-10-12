"""create_table

Revision ID: ebdc9aa75ee0
Revises: b3ea0d252ba7
Create Date: 2025-10-12 20:46:36.097040

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ebdc9aa75ee0'
down_revision: Union[str, None] = 'b3ea0d252ba7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
