"""Add face recognition tables and person columns to pose_windows

Revision ID: d6e7f8a9b0c1
Revises: c5d6e7f8a9b0
Create Date: 2026-02-24

This migration adds:
- persons table: stores registered persons for face recognition
- person_faces table: stores face images and embeddings
- gallery_versions table: tracks face gallery version changes
- person columns in pose_windows: links windows to identified persons
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "d6e7f8a9b0c1"
down_revision: Union[str, Sequence[str], None] = "c5d6e7f8a9b0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create persons table
    op.create_table(
        "persons",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_persons_is_active", "persons", ["is_active"])
    op.create_index("ix_persons_created_at", "persons", ["created_at"])

    # Create person_faces table
    op.create_table(
        "person_faces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("person_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("persons.id", ondelete="CASCADE"), nullable=False),
        sa.Column("image_path", sa.String(), nullable=False),
        sa.Column("embedding", postgresql.JSON(), nullable=False),
        sa.Column("det_score", sa.Float(), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_person_faces_person_id", "person_faces", ["person_id"])
    op.create_index("ix_person_faces_created_at", "person_faces", ["created_at"])

    # Create gallery_versions table
    op.create_table(
        "gallery_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("version", sa.String(), unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_gallery_versions_version", "gallery_versions", ["version"])
    op.create_index("ix_gallery_versions_created_at", "gallery_versions", ["created_at"])

    # Add person columns to pose_windows
    op.add_column("pose_windows", sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column("pose_windows", sa.Column("person_name", sa.String(), nullable=True))
    op.add_column("pose_windows", sa.Column("person_conf", sa.Float(), nullable=True))
    op.add_column("pose_windows", sa.Column("gallery_version", sa.String(), nullable=True))

    # Add foreign key and index for person_id
    op.create_foreign_key(
        "fk_pose_windows_person_id",
        "pose_windows",
        "persons",
        ["person_id"],
        ["id"],
        ondelete="SET NULL"
    )
    op.create_index("ix_pose_windows_person_id", "pose_windows", ["person_id"])


def downgrade() -> None:
    # Remove person columns from pose_windows
    op.drop_index("ix_pose_windows_person_id", table_name="pose_windows")
    op.drop_constraint("fk_pose_windows_person_id", "pose_windows", type_="foreignkey")
    op.drop_column("pose_windows", "gallery_version")
    op.drop_column("pose_windows", "person_conf")
    op.drop_column("pose_windows", "person_name")
    op.drop_column("pose_windows", "person_id")

    # Drop gallery_versions table
    op.drop_index("ix_gallery_versions_created_at", table_name="gallery_versions")
    op.drop_index("ix_gallery_versions_version", table_name="gallery_versions")
    op.drop_table("gallery_versions")

    # Drop person_faces table
    op.drop_index("ix_person_faces_created_at", table_name="person_faces")
    op.drop_index("ix_person_faces_person_id", table_name="person_faces")
    op.drop_table("person_faces")

    # Drop persons table
    op.drop_index("ix_persons_created_at", table_name="persons")
    op.drop_index("ix_persons_is_active", table_name="persons")
    op.drop_table("persons")
