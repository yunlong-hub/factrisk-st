from types import SimpleNamespace

from factrisk.pipeline import environment_lock


def test_closure_tracks_extras_markers_and_conflicts(monkeypatch):
    packages = {
        'root': ('1.0', ['child[score]>=2', 'skipped; python_version < "2"']),
        'child': ('1.0', ['leaf; extra == "score"']),
        'leaf': ('3.0', []),
    }
    monkeypatch.setattr(environment_lock.metadata, 'distribution',
                        lambda name: SimpleNamespace(version=packages[name][0], requires=packages[name][1]))
    monkeypatch.setattr(environment_lock.metadata, 'version', lambda name: packages[name][0])
    versions, conflicts = environment_lock.installed_closure(['root'])
    assert versions == {'root': '1.0', 'child': '1.0', 'leaf': '3.0'}
    assert conflicts == [{'parent': 'root', 'requirement': 'child[score]>=2', 'installed': '1.0'}]
