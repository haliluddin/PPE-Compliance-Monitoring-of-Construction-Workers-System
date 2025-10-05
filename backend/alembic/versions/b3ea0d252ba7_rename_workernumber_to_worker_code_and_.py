"""Rename workerNumber to worker_code and add registered

Revision ID: b3ea0d252ba7
Revises: ba126c9d3f8e
Create Date: 2025-10-06 04:43:08.432717

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3ea0d252ba7'
down_revision: Union[str, None] = 'ba126c9d3f8e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Rename column
    op.alter_column('workers', 'workerNumber', new_column_name='worker_code')
    
    # Add new column
    op.add_column('workers', sa.Column('registered', sa.Boolean(), nullable=False, server_default=sa.false()))

def downgrade():
    # Revert changes
    op.alter_column('workers', 'worker_code', new_column_name='workerNumber')
    op.drop_column('workers', 'registered')

