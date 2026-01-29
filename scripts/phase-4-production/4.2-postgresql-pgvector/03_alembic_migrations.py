"""
03 - Alembic Migrations
========================
Database schema versioning with Alembic.

Key concept: Alembic manages database schema changes over time,
enabling version control for your database structure.

Book reference: —

Note: This script demonstrates Alembic usage. In practice, you would:
1. Initialize: alembic init alembic
2. Configure: Edit alembic.ini and alembic/env.py
3. Generate: alembic revision --autogenerate -m "message"
4. Apply: alembic upgrade head
"""

import subprocess
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


class AlembicManager:
    """Manager for Alembic migration operations."""

    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.alembic_dir = self.project_dir / "alembic"

    def run_command(self, cmd: list[str]) -> tuple[bool, str]:
        """Run an Alembic command."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_dir,
                capture_output=True,
                text=True
            )
            return result.returncode == 0, result.stdout + result.stderr
        except Exception as e:
            return False, str(e)

    def init(self) -> bool:
        """Initialize Alembic in the project."""
        print("Initializing Alembic...")
        success, output = self.run_command(["alembic", "init", "alembic"])

        if success:
            print("✓ Alembic initialized")
            print("\nNext steps:")
            print("1. Edit alembic.ini - set sqlalchemy.url")
            print("2. Edit alembic/env.py - import your models")
            return True
        else:
            print(f"✗ Failed: {output}")
            return False

    def create_migration(self, message: str, autogenerate: bool = True) -> bool:
        """Create a new migration."""
        print(f"Creating migration: {message}")

        cmd = ["alembic", "revision"]
        if autogenerate:
            cmd.append("--autogenerate")
        cmd.extend(["-m", message])

        success, output = self.run_command(cmd)

        if success:
            print(f"✓ Migration created: {message}")
            return True
        else:
            print(f"✗ Failed: {output}")
            return False

    def upgrade(self, revision: str = "head") -> bool:
        """Apply migrations up to a revision."""
        print(f"Upgrading to: {revision}")
        success, output = self.run_command(["alembic", "upgrade", revision])

        if success:
            print(f"✓ Upgraded to {revision}")
            return True
        else:
            print(f"✗ Failed: {output}")
            return False

    def downgrade(self, revision: str = "-1") -> bool:
        """Rollback migrations to a revision."""
        print(f"Downgrading to: {revision}")
        success, output = self.run_command(["alembic", "downgrade", revision])

        if success:
            print(f"✓ Downgraded to {revision}")
            return True
        else:
            print(f"✗ Failed: {output}")
            return False

    def current(self) -> Optional[str]:
        """Get current migration revision."""
        success, output = self.run_command(["alembic", "current"])

        if success:
            print("Current revision:")
            print(output)
            return output
        else:
            print(f"✗ Failed: {output}")
            return None

    def history(self) -> Optional[str]:
        """Show migration history."""
        success, output = self.run_command(["alembic", "history"])

        if success:
            print("Migration history:")
            print(output)
            return output
        else:
            print(f"✗ Failed: {output}")
            return None


if __name__ == "__main__":
    print("Alembic Migrations Demo")
    print("=" * 50)

    manager = AlembicManager()

    print("\nAlembic Workflow:")
    print("1. Initialize:   alembic init alembic")
    print("2. Create:       alembic revision --autogenerate -m 'Add table'")
    print("3. Apply:        alembic upgrade head")
    print("4. Rollback:     alembic downgrade -1")
    print("5. Check:        alembic current")
    print("6. History:      alembic history")

    print("\nExample migration file structure:")
    print("""
    def upgrade():
        op.create_table(
            'documents',
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('title', sa.String(255), nullable=False),
            sa.Column('content', sa.Text(), nullable=False)
        )

    def downgrade():
        op.drop_table('documents')
    """)

    print("\n" + "=" * 50)
    print("Run these commands manually to use Alembic:")
    print('  cd "scripts/phase-4-production/4.2-postgresql-pgvector"')
    print("  alembic init alembic")
    print("  alembic revision --autogenerate -m 'Initial migration'")
    print("  alembic upgrade head")
