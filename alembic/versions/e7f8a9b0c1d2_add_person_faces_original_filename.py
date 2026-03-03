"""Add original_filename to person_faces

Revision ID: e7f8a9b0c1d2
Revises: d6e7f8a9b0c1
Create Date: 2026-03-02

Used to skip re-uploading an image when the same filename already exists for the person.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e7f8a9b0c1d2"
down_revision: Union[str, Sequence[str], None] = "d6e7f8a9b0c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("person_faces", sa.Column("original_filename", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("person_faces", "original_filename")
