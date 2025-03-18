from app import db

def upgrade():
    """Add source_url column to question table."""
    db.engine.execute('ALTER TABLE question ADD COLUMN source_url TEXT')

def downgrade():
    """Remove source_url column from question table."""
    db.engine.execute('ALTER TABLE question DROP COLUMN source_url') 