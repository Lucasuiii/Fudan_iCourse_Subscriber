"""Explicit post-commit checkpoints scoped to one pipeline database instance."""
from src.data.database import Database


class CheckpointDatabase(Database):
    """Persist selected state transitions before returning to the caller.

    A checkpoint failure propagates after the SQLite commit. Other database
    instances and metadata writes do not inherit this callback.
    """
    def __init__(self, db_path=None, *, checkpoint=None):
        self.checkpoint = checkpoint
        super().__init__(db_path)

    def _committed(self, mutation, *args):
        result = mutation(*args)
        if self.checkpoint is not None:
            self.checkpoint()
        return result

    def update_transcript(self, sub_id, transcript):
        return self._committed(super().update_transcript, sub_id, transcript)

    def clear_transcript(self, sub_id):
        return self._committed(super().clear_transcript, sub_id)

    def update_summary(self, sub_id, summary, model):
        return self._committed(super().update_summary, sub_id, summary, model)

    def mark_processed(self, sub_id):
        return self._committed(super().mark_processed, sub_id)

    def clear_error(self, sub_id):
        return self._committed(super().clear_error, sub_id)

    def update_error(self, sub_id, stage, error_msg):
        return self._committed(super().update_error, sub_id, stage, error_msg)

    def mark_emailed_batch(self, sub_ids):
        return self._committed(super().mark_emailed_batch, sub_ids)

    def mark_failure_notified_batch(self, sub_ids):
        return self._committed(super().mark_failure_notified_batch, sub_ids)
