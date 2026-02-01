"""Add activity_events quality columns (k_count, avg_pose_conf, frames_ok_ratio)

Revision ID: a1b2c3d4e5f6
Revises: f3eb52747698
Create Date: 2026-01-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'f3eb52747698'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add input quality columns to activity_events (nullable for backward compatibility)."""
    op.add_column('activity_events', sa.Column('k_count', sa.Integer(), nullable=True))
    op.add_column('activity_events', sa.Column('avg_pose_conf', sa.Float(), nullable=True))
    op.add_column('activity_events', sa.Column('frames_ok_ratio', sa.Float(), nullable=True))


def downgrade() -> None:
    """Remove input quality columns from activity_events."""
    op.drop_column('activity_events', 'frames_ok_ratio')
    op.drop_column('activity_events', 'avg_pose_conf')
    op.drop_column('activity_events', 'k_count')
