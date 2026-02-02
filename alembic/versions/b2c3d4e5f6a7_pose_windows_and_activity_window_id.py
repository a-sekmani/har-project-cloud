"""Phase 5: pose_windows table + activity_events.window_id

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "pose_windows",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("device_id", sa.String(), nullable=False),
        sa.Column("camera_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("ts_start_ms", sa.BigInteger(), nullable=False),
        sa.Column("ts_end_ms", sa.BigInteger(), nullable=False),
        sa.Column("fps", sa.Float(), nullable=False),
        sa.Column("window_size", sa.Integer(), nullable=False),
        sa.Column("coord_space", sa.String(), nullable=False, server_default="norm"),
        sa.Column("keypoints", postgresql.JSONB(), nullable=False),
        sa.Column("mean_pose_conf", sa.Float(), nullable=True),
        sa.Column("missing_ratio", sa.Float(), nullable=True),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column("label_source", sa.String(), nullable=True),
        sa.Column("labeled_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_pose_windows_device_id", "pose_windows", ["device_id"])
    op.create_index("ix_pose_windows_camera_id", "pose_windows", ["camera_id"])
    op.create_index("ix_pose_windows_session_id", "pose_windows", ["session_id"])
    op.create_index("ix_pose_windows_track_id", "pose_windows", ["track_id"])
    op.create_index("ix_pose_windows_label", "pose_windows", ["label"])
    op.create_index("ix_pose_windows_created_at", "pose_windows", ["created_at"])

    op.add_column(
        "activity_events",
        sa.Column("window_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "activity_events_window_id_fkey",
        "activity_events",
        "pose_windows",
        ["window_id"],
        ["id"],
    )
    op.create_index("ix_activity_events_window_id", "activity_events", ["window_id"])


def downgrade() -> None:
    op.drop_index("ix_activity_events_window_id", table_name="activity_events")
    op.drop_constraint(
        "activity_events_window_id_fkey", "activity_events", type_="foreignkey"
    )
    op.drop_column("activity_events", "window_id")

    op.drop_index("ix_pose_windows_created_at", table_name="pose_windows")
    op.drop_index("ix_pose_windows_label", table_name="pose_windows")
    op.drop_index("ix_pose_windows_track_id", table_name="pose_windows")
    op.drop_index("ix_pose_windows_session_id", table_name="pose_windows")
    op.drop_index("ix_pose_windows_camera_id", table_name="pose_windows")
    op.drop_index("ix_pose_windows_device_id", table_name="pose_windows")
    op.drop_table("pose_windows")
