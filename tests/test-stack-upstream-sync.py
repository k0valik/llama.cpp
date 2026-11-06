#!/usr/bin/env python3
"""Local-only tests of the rolling merge worker; creates disposable repositories."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/stack-upstream-sync.sh'
BASE = 'port/qwen3.8-27b-cuda-v0.9.0'
CANDIDATE = 'sync/upstream-v090'


def run(*args, cwd=None, env=None, check=True):
    return subprocess.run(args, cwd=cwd, env=env, check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


class SyncTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.origin = self.root / 'origin.git'
        self.upstream = self.root / 'upstream.git'
        self.seed = self.root / 'seed'
        run('git', 'init', '--bare', str(self.origin))
        run('git', 'init', '--bare', str(self.upstream))
        run('git', 'init', '-b', 'master', str(self.seed))
        self.git(self.seed, 'config', 'user.name', 'Test')
        self.git(self.seed, 'config', 'user.email', 'test@example.com')
        self.commit(self.seed, 'shared', 'initial\n')
        self.git(self.seed, 'remote', 'add', 'origin', str(self.origin))
        self.git(self.seed, 'remote', 'add', 'upstream', str(self.upstream))
        self.git(self.seed, 'push', 'upstream', 'master')
        self.git(self.seed, 'branch', BASE)
        self.git(self.seed, 'branch', CANDIDATE)
        self.git(self.seed, 'push', 'origin', BASE, CANDIDATE)
        self.worker = self.root / 'worker'
        run('git', 'clone', '--no-checkout', str(self.origin), str(self.worker))
        self.copy = self.root / 'worker-script.sh'
        shutil.copyfile(SCRIPT, self.copy)
        self.env = os.environ.copy()
        self.env.update(SYNC_REPORT=str(self.root / 'report.md'),
                        SYNC_UPSTREAM_URL=str(self.upstream))

    def git(self, cwd, *args):
        return run('git', *args, cwd=cwd).stdout.strip()

    def commit(self, cwd, filename, value):
        (cwd / filename).write_text(value)
        self.git(cwd, 'add', filename)
        self.git(cwd, 'commit', '-m', 'change ' + filename)

    def worker_run(self):
        return run('bash', str(self.copy), cwd=self.worker, env=self.env, check=False)

    def remote_tip(self):
        return self.git(self.seed, 'ls-remote', str(self.origin), 'refs/heads/' + CANDIDATE).split()[0]

    def test_clean_repeat_and_canonical_advance(self):
        self.commit(self.seed, 'upstream-only', 'upstream\n')
        upstream_tip = self.git(self.seed, 'rev-parse', 'HEAD')
        self.git(self.seed, 'push', 'upstream', 'master')
        result = self.worker_run()
        self.assertEqual(result.returncode, 0, result.stdout)
        first = self.remote_tip()
        self.assertEqual(self.worker_run().returncode, 0)
        self.assertEqual(first, self.remote_tip())
        self.git(self.seed, 'switch', BASE)
        self.commit(self.seed, 'base-only', 'base\n')
        base_tip = self.git(self.seed, 'rev-parse', 'HEAD')
        self.git(self.seed, 'push', 'origin', BASE)
        result = self.worker_run()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotEqual(first, self.remote_tip())
        self.git(self.worker, 'merge-base', '--is-ancestor', base_tip, 'HEAD')
        self.git(self.worker, 'merge-base', '--is-ancestor', upstream_tip, 'HEAD')

    def test_rejected_push_does_not_publish(self):
        self.commit(self.seed, 'upstream-only', 'upstream\n')
        self.git(self.seed, 'push', 'upstream', 'master')
        before = self.remote_tip()
        hook = self.origin / 'hooks/pre-receive'
        hook.write_text('#!/bin/sh\nexit 1\n')
        hook.chmod(0o755)
        result = self.worker_run()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(before, self.remote_tip())
        self.assertIn('push rejected', (self.root / 'report.md').read_text())

    def test_conflict_does_not_publish_partial_base_merge(self):
        self.commit(self.seed, 'shared', 'upstream\n')
        self.git(self.seed, 'push', 'upstream', 'master')
        self.git(self.seed, 'switch', CANDIDATE)
        self.commit(self.seed, 'shared', 'candidate\n')
        self.git(self.seed, 'push', 'origin', CANDIDATE)
        before = self.remote_tip()
        self.git(self.seed, 'switch', BASE)
        self.commit(self.seed, 'base-only', 'base\n')
        self.git(self.seed, 'push', 'origin', BASE)
        result = self.worker_run()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(before, self.remote_tip())
        report = (self.root / 'report.md').read_text()
        self.assertIn('Blocked: conflict', report)
        self.assertIn('Files changed on both sides', report)
        self.assertIn('change shared', report)
        self.assertIn('shared', report)


if __name__ == '__main__':
    unittest.main()
