"""Regression checks on the actual Actions graph, without course or cloud access.

Requires PyYAML, also installed by the branch's synthetic validation job.
"""
import itertools
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class CourseScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.caller = yaml.load((ROOT / '.github/workflows/qwen_sharded_pilot.yml').read_text(),
                               Loader=yaml.BaseLoader)
        called = cls.caller['jobs']['course']['uses']
        cls.course = yaml.load((ROOT / called).read_text(), Loader=yaml.BaseLoader)

    def dependencies(self, job):
        value = job.get('needs', [])
        return [value] if isinstance(value, str) else value

    def gather_ready(self, current_course, states):
        # Expand dependencies within the actual reusable-workflow call's scope.
        terminal = {'success', 'failure', 'cancelled', 'skipped'}
        return all(states[current_course].get(dep) in terminal
                   for dep in self.dependencies(self.course['jobs']['gather']))

    def test_completed_course_can_gather_while_another_is_still_recognizing(self):
        states = {0: {'asr': 'success'}, 1: {'asr': 'in_progress'}}
        self.assertEqual(self.dependencies(self.course['jobs']['gather']), ['asr'])
        self.assertTrue(self.gather_ready(0, states))
        self.assertFalse(self.gather_ready(1, states))

    def test_failed_course_does_not_cancel_or_delay_another_course(self):
        states = {0: {'asr': 'failure'}, 1: {'asr': 'success'}}
        self.assertEqual(self.caller['jobs']['course']['strategy']['fail-fast'], 'false')
        self.assertEqual(self.course['jobs']['asr']['strategy']['fail-fast'], 'false')
        self.assertTrue(self.gather_ready(1, states))
        # Failed local ASR reaches the existing full-block validation, never a
        # success-dependent shortcut that might skip reporting missing blocks.
        self.assertIn('always()', self.course['jobs']['gather']['if'])
        self.assertIn('!cancelled()', self.course['jobs']['gather']['if'])

    def test_mixed_recognition_and_gather_never_exceed_fifteen_runners(self):
        course_limit = int(self.caller['jobs']['course']['strategy']['max-parallel'])
        worker_limit = int(self.course['jobs']['asr']['strategy']['max-parallel'])
        self.assertEqual(course_limit, 5)
        self.assertLessEqual(worker_limit, 3)
        self.assertNotIn('runs-on', self.caller['jobs']['course'])
        self.assertNotIn('strategy', self.course['jobs']['gather'])
        # Each course is in ASR (1..3 workers), gather (1), or idle (0);
        # local dependencies forbid ASR and gather from overlapping.
        for allocations in itertools.product(range(worker_limit + 1), repeat=course_limit):
            self.assertLessEqual(sum(allocations), 15)
        self.assertEqual(set(self.course['jobs']), {'asr', 'gather'})
        self.assertEqual(self.dependencies(self.caller['jobs']['course']), ['plan_workers'])

    def test_checkpoint_restore_still_precedes_decode_and_finalization(self):
        for job, command, artifact in [('asr', 'worker', 'qwen-shard-result-'),
                                        ('gather', 'gather', 'qwen-shard-final-')]:
            steps = self.course['jobs'][job]['steps']
            restore = next(i for i, step in enumerate(steps) if step.get('id') == 'restore')
            execute = next(i for i, step in enumerate(steps)
                           if step.get('run') == f'python -m scripts.sharded_qwen_pilot {command}')
            self.assertLess(restore, execute)
            self.assertTrue(steps[restore]['env']['RESTORE_ARTIFACT'].startswith(artifact))
            upload = next(step for step in steps if step.get('uses') == 'actions/upload-artifact@v4'
                          and step['with']['name'].startswith(artifact))
            self.assertIn("steps.restore.outcome == 'success'", upload['if'])

    def test_only_caller_holds_batch_lock_and_course_has_no_uis_secrets(self):
        self.assertEqual(self.caller['concurrency']['cancel-in-progress'], 'false')
        self.assertNotIn('concurrency', self.course)
        for job in self.course['jobs'].values():
            self.assertNotIn('concurrency', job)
        self.assertEqual(set(self.caller['jobs']['course']['secrets']),
                         {'QWEN_ASR_TEST_KEY', 'DEEPSEEK_API_KEY', 'DOUBAO_ASR_API_KEY'})


if __name__ == '__main__':
    unittest.main()
