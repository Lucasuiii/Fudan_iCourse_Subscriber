"""Formal publishing stage, with explicit runtime services and frozen inputs."""
from __future__ import annotations
import os
from pathlib import Path


def publish_result(runtime):
    files = runtime.decode(runtime.root()/'inbox'/'state.enc', 'state')
    spec = runtime.read_json(files['specification.json'])
    course, sub_id = spec['course_id'], str(spec['lecture']['sub_id'])
    configured = runtime.configured_courses(os.environ['COURSE_IDS'])
    if course not in configured: raise ValueError('Publication course is no longer subscribed')
    delta = runtime.root()/'delta.db'; delta.write_bytes(files['database.db'])
    if os.environ.get('PUBLISH_RESULTS') == 'true':
        runtime.publish(delta, course, sub_id)
        print('Encrypted lecture state published', flush=True)
    else:
        target = runtime.root()/'isolated.db'
        runtime.load_remote(target)
        runtime.merge_lecture(delta, target, course, sub_id)
        db = runtime.Database(str(target))
        try: payload = runtime.lecture_snapshot(db, runtime.root()/'snapshot.db', course, sub_id)
        finally: db.conn.close()
        runtime.encode({'database.db': payload, 'specification.json': files['specification.json'],
                'review.json': files['review.json']}, 'published', runtime.out('published.enc'))
        print('Isolated publication merge validated; remote database unchanged', flush=True)


def deliver(runtime):
    """One batch mail outbox after durable publication, with saved send receipts."""
    if os.environ.get('PUBLISH_RESULTS') != 'true' or os.environ.get('SEND_EMAIL') != 'true':
        raise ValueError('Email requires explicit publish and send switches')
    from scripts.parallel_courses import deliver as send
    target = Path('data/icourse.db'); target.parent.mkdir(exist_ok=True)
    runtime.load_remote(target)
    queue = runtime.decode(runtime.root()/'inbox'/'queue.enc', 'queue')
    tasks = runtime.read_json(queue['queue.json'])
    configured = runtime.configured_courses(os.environ['COURSE_IDS'])
    baseline = runtime.Database(str(target))
    receipts_before = {r['sub_id']: (r['emailed_at'], r['failure_notified_at'])
                       for r in baseline.conn.execute('SELECT * FROM lectures').fetchall()}
    baseline.conn.close()
    if runtime.artifact('qwen-production-mail-receipts', runtime.root()/'previous'):
        saved = runtime.decode(runtime.root()/'previous'/'receipts.enc', 'mail')
        previous = runtime.root()/'receipts.db'; previous.write_bytes(saved['database.db'])
        prior = runtime.Database(str(previous))
        current = runtime.Database(str(target))
        try:
            for row in prior.conn.execute('SELECT * FROM lectures').fetchall():
                if row['course_id'] in configured and (row['emailed_at'] or row['failure_notified_at']):
                    current.conn.execute('''UPDATE lectures SET emailed_at=COALESCE(emailed_at,?),
                        failure_notified_at=COALESCE(failure_notified_at,?) WHERE sub_id=?''',
                        (row['emailed_at'], row['failure_notified_at'], row['sub_id']))
            current.conn.commit()
        finally:
            prior.conn.close(); current.conn.close()
    def mail_checkpoint(db):
        runtime.encode({'database.db': runtime.snapshot(db, runtime.root()/'mail-snapshot.db')}, 'mail', runtime.out('receipts.enc'))
    def mail_database(path):
        db = runtime.CheckpointDatabase(str(path))
        db.checkpoint = lambda: mail_checkpoint(db)
        return db
    try:
        send(database_factory=mail_database)
    finally:
        db = runtime.Database(str(target))
        try:
            mail_checkpoint(db)
            for row in db.conn.execute('SELECT * FROM lectures').fetchall():
                course, sub_id = row['course_id'], row['sub_id']
                if course in configured and (row['emailed_at'], row['failure_notified_at']) != receipts_before.get(sub_id):
                    delta = runtime.root()/'mail-delta.db'
                    runtime.lecture_snapshot(db, delta, course, sub_id)
                    runtime.publish(delta, course, sub_id)
        finally:
            db.conn.close()


def persist_pool_failures(runtime):
    """Keep failed lecture progress without running gather or creating quota state."""
    from scripts.production_pool import read_state, store_for
    store = store_for()
    try: _, state = read_state(store)
    finally: store.close()
    if any(t['status'] != 'completed' for t in state['tickets']):
        raise ValueError('Cannot persist failures while children are active')
    payload = {}
    for raw, course_state in state['courses'].items():
        if course_state['phase'] != 'failed': continue
        slot = int(raw)
        tickets = [t for t in state['tickets'] if t['slot'] == slot]
        phase = tickets[-1]['stage']
        with runtime.shards.environment({'COURSE_SLOT': raw}):
            target = runtime.root()/'failed-input'/raw
            if runtime.artifact(f'qwen-production-state-{slot}', target):
                files = runtime.decode(target/'state.enc', 'state')
            elif runtime.artifact(f'qwen-production-prepare-{slot}', target):
                files = runtime.decode(target/'prepared.enc', 'prepared')
            else:
                db, course, title, lecture = runtime.task_files()
                try:
                    files = {'database.db': runtime.lecture_snapshot(db, runtime.root()/'failure-source.db', course, str(lecture['sub_id'])),
                             'specification.json': runtime.shards.encoded({'course_id': course, 'lecture': lecture})}
                finally: db.conn.close()
            spec = runtime.read_json(files['specification.json'])
            course, sub_id = spec['course_id'], str(spec['lecture']['sub_id'])
            if course not in runtime.configured_courses(os.environ['COURSE_IDS']):
                raise ValueError('Failure publication course is no longer subscribed')
            delta = runtime.root()/'failure.db'; delta.write_bytes(files['database.db'])
            db = runtime.Database(str(delta))
            try:
                if not db.get_lecture(sub_id).get('error_stage'):
                    db.update_error(sub_id, 'pool_'+phase, 'stage_failure')
                payload[f'failure-{slot}.db'] = runtime.lecture_snapshot(db, runtime.root()/'failure-snapshot.db', course, sub_id)
            finally: db.conn.close()
            # No state.enc or review.json is manufactured here. Missing quota
            # checkpoints remain missing and continue to fence future review.
            if os.environ.get('PUBLISH_RESULTS') == 'true':
                runtime.publish(delta, course, sub_id)
            else:
                isolated = runtime.root()/'failure-isolated.db'
                runtime.load_remote(isolated)
                runtime.merge_lecture(delta, isolated, course, sub_id)
    if payload:
        with runtime.shards.environment({'COURSE_SLOT': '0'}):
            runtime.encode(payload, 'pool-failures', runtime.out('pool-failures.enc'))


def finalize(runtime):
    """Retain the ordinary catalog/subscription sync, including empty lecture queues."""
    if os.environ.get('POOL_ARTIFACTS') == 'true':
        runtime.persist_pool_failures()
    queue = runtime.decode(runtime.root()/'inbox'/'queue.enc', 'queue')
    delta = runtime.root()/'catalog.db'; delta.write_bytes(queue['database.db'])
    db = runtime.Database(str(delta))
    try:
        with db.conn:
            for table in ('ppt_pages', 'lectures', 'courses'):
                db.conn.execute('DELETE FROM '+table)
            db.conn.execute('DELETE FROM meta')
        runtime.snapshot(db, runtime.root()/'catalog-snapshot.db')
    finally:
        db.conn.close()
    if os.environ.get('PUBLISH_RESULTS') == 'true':
        runtime.publish(runtime.root()/'catalog-snapshot.db')
        if os.environ.get('SEND_EMAIL') == 'true':
            runtime.deliver()
    else:
        runtime.encode({'database.db': (runtime.root()/'catalog-snapshot.db').read_bytes()}, 'catalog', runtime.out('catalog.enc'))
    # Process successfully enumerated courses first, but do not report an
    # incomplete scan as a clean empty/successful run. Old queues stay readable.
    if runtime.read_json(queue.get('enumeration.json', b'{}')).get('failed_course_count', 0):
        from src.runtime.enumeration import CourseEnumerationError
        raise CourseEnumerationError()

