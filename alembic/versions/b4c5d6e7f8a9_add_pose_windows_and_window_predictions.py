"""Add pose_windows and window_predictions tables

Revision ID: b4c5d6e7f8a9
Revises: a1b2c3d4e5f6
Create Date: 2026-01-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "b4c5d6e7f8a9"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pose_windows",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("device_id", sa.String(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("ts_start_ms", sa.BigInteger(), nullable=False),
        sa.Column("ts_end_ms", sa.BigInteger(), nullable=False),
        sa.Column("fps", sa.Integer(), nullable=False),
        sa.Column("window_size", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_pose_windows_device_id", "pose_windows", ["device_id"])
    op.create_index("ix_pose_windows_created_at", "pose_windows", ["created_at"])
    op.create_index("ix_pose_windows_label", "pose_windows", ["label"])

    op.create_table(
        "window_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("window_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_key", sa.String(), nullable=False),
        sa.Column("pred_label", sa.String(), nullable=False),
        sa.Column("pred_conf", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["window_id"], ["pose_windows.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_window_predictions_window_id", "window_predictions", ["window_id"])
    op.create_index("ix_window_predictions_model_key", "window_predictions", ["model_key"])
    op.create_index("ix_window_predictions_created_at", "window_predictions", ["created_at"])


def downgrade() -> None:
    op.drop_table("window_predictions")
    op.drop_table("pose_windows")
