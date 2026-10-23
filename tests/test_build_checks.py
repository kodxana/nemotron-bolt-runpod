from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.build_checks import check_dependencies


class BuildCheckTests(unittest.TestCase):
    def test_mixed_otel_version_remains_a_hard_failure(self):
        requires = {
            'exporter': ['opentelemetry-exporter-otlp-proto-common==1.40.0'],
            'opentelemetry-exporter-otlp-proto-common': [],
        }
        with patch('scripts.build_checks.distribution', side_effect=lambda name: SimpleNamespace(requires=requires[name])), \
             patch('scripts.build_checks.version', return_value='1.45.1'):
            with self.assertRaisesRegex(RuntimeError, 'common==1.40.0; installed 1.45.1'):
                check_dependencies('exporter')

    def test_reports_all_conflicts_and_missing_packages(self):
        requires = {'engine': ['one==1', 'two==2', 'missing'], 'one': [], 'two': []}
        def installed(name):
            if name == 'missing':
                raise PackageNotFoundError(name)
            return '9'
        with patch('scripts.build_checks.distribution', side_effect=lambda name: SimpleNamespace(requires=requires[name])), \
             patch('scripts.build_checks.version', side_effect=installed):
            with self.assertRaises(RuntimeError) as caught:
                check_dependencies('engine')
        for fragment in ('one==1; installed 9', 'two==2; installed 9', 'missing; not installed'):
            self.assertIn(fragment, str(caught.exception))

    def test_requested_extras_are_checked_and_cycles_terminate(self):
        requires = {'engine': ['exporter[http]==1'], 'exporter': ['engine==1', 'transport==1; extra == "http"', 'optional==1; extra == "gcp"'], 'transport': []}
        with patch('scripts.build_checks.distribution', side_effect=lambda name: SimpleNamespace(requires=requires[name])), \
             patch('scripts.build_checks.version', return_value='1') as installed:
            check_dependencies('engine')
        self.assertIn(('transport',), [call.args for call in installed.call_args_list])
        self.assertNotIn(('optional',), [call.args for call in installed.call_args_list])
