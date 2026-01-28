"""Initial migration: create devices and activity_events tables

Revision ID: f3eb52747698
Revises: 
Create Date: 2026-01-28 17:20:37.320243

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'f3eb52747698'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create devices table
    op.create_table(
        'devices',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('device_id', sa.String(), nullable=False, unique=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_devices_device_id', 'devices', ['device_id'])

    # Create activity_events table
    op.create_table(
        'activity_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('device_id', sa.String(), nullable=False),
        sa.Column('camera_id', sa.String(), nullable=False),
        sa.Column('track_id', sa.Integer(), nullable=False),
        sa.Column('ts_start_ms', sa.BigInteger(), nullable=False),
        sa.Column('ts_end_ms', sa.BigInteger(), nullable=False),
        sa.Column('fps', sa.Integer(), nullable=False),
        sa.Column('window_size', sa.Integer(), nullable=False),
        sa.Column('activity', sa.String(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['device_id'], ['devices.device_id']),
    )
    op.create_index('ix_activity_events_device_id', 'activity_events', ['device_id'])
    op.create_index('ix_activity_events_camera_id', 'activity_events', ['camera_id'])
    op.create_index('ix_activity_events_track_id', 'activity_events', ['track_id'])
    op.create_index('ix_activity_events_activity', 'activity_events', ['activity'])
    op.create_index('ix_activity_events_created_at', 'activity_events', ['created_at'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('activity_events')
    op.drop_table('devices')
