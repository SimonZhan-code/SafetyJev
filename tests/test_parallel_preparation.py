import json
import os
import pytest
from test_semantic_package import definitions
from test_source_episodes import source_fixture
from safetyjev.predictor_judge_dataset import build_package


def test_parallel_and_reused_media_preserve_all_supervision(tmp_path, monkeypatch):
    raw=tmp_path/'raw';source_fixture(raw,semantic=True,observation_count=17)
    options=dict(group_splits={'jar_transport/task_0000':'train'},semantic_definitions=definitions(),train_episodes='all')
    serial=tmp_path/'serial';parallel=tmp_path/'parallel'
    a=build_package([raw],serial,input_contract='chunk_start_v2',**options)
    b=build_package([raw],parallel,input_contract='chunk_start_v2',media_workers=2,**options)
    for key in ('file_sha256','normalization','normalization_statistics','counts','label_reasons'):
        assert a[key]==b[key]
    assert json.loads((serial/'composition.json').read_text())==json.loads((parallel/'composition.json').read_text())
    baseline=build_package([raw],tmp_path/'classifier-base',semantic_task='classifier',**options)
    from safetyjev import media_preparation
    def forbidden(*args,**kwargs):raise AssertionError('Validated media was decoded again')
    monkeypatch.setattr(media_preparation,'validate_batch',forbidden)
    reused=build_package([raw],tmp_path/'classifier-reused',semantic_task='classifier',reuse_media=parallel,reuse_annotations=parallel/'annotations',**options)
    for key in ('file_sha256','normalization','counts','label_reasons'):
        assert baseline[key]==reused[key]


def test_reuse_rejects_source_edit_even_with_restored_size_and_mtime(tmp_path):
    raw=tmp_path/'raw';source_fixture(raw,semantic=True,observation_count=17)
    options=dict(group_splits={'jar_transport/task_0000':'train'},semantic_definitions=definitions(),train_episodes='all')
    package=tmp_path/'first';build_package([raw],package,**options)
    f=raw/'trajectory.hdf5';st=f.stat()
    import h5py
    with h5py.File(f,'r+') as h:h['robot/q'][0,0]+=1
    os.utime(f,ns=(st.st_atime_ns,st.st_mtime_ns));assert f.stat().st_size==st.st_size
    with pytest.raises(ValueError,match='[Ss]ource media changed'):
        build_package([raw],tmp_path/'second',reuse_media=package,**options)


def test_parallel_corrupt_media_is_excluded_not_accepted(tmp_path):
    from safetyjev.media_preparation import MediaValidator
    raw=tmp_path/'raw';raw.mkdir();(raw/'bad.jpg').write_bytes(b'not an image')
    with MediaValidator(2) as validator:
        with pytest.raises((OSError,ValueError)):
            validator.validate(raw,{'observations':[{'images':{'overview':'bad.jpg'}}]},'bad')


def test_parallel_annotations_equal_serial_and_resume(tmp_path):
    from safetyjev.annotation_preparation import prepare_annotations
    from safetyjev.semantic_data import prepare_semantic_annotation,write_annotation
    raw=tmp_path/'raw';source_fixture(raw,semantic=True,observation_count=17)
    d=definitions();baseline=tmp_path/'baseline.json'
    write_annotation(baseline,prepare_semantic_annotation(raw,'source0',d))
    output=tmp_path/'annotations'
    assert list(prepare_annotations([(raw,'source0')],output,d,workers=2))==['source0']
    files=list(output.glob('*.json'));assert len(files)==1
    assert files[0].read_bytes()==baseline.read_bytes()
    resumed=tmp_path/'resumed'
    assert list(prepare_annotations([(raw,'source0')],resumed,d,workers=2,reuse_dirs=[output]))==['source0']
    assert next(resumed.glob('*.json')).read_bytes()==baseline.read_bytes()


def test_parallel_prepared_records_preserve_bytes_and_reject_changes(tmp_path, monkeypatch):
    from safetyjev.annotation_preparation import prepare_annotations,load_prepared_record
    from safetyjev.source_episodes import load_source_record
    raw=tmp_path/'raw';source_fixture(raw,semantic=True,observation_count=17)
    out=tmp_path/'prepared'
    list(prepare_annotations([(raw,'source0')],out,definitions(),workers=2,prepare_records=True))
    assert load_prepared_record(raw,out)==load_source_record(raw)
    options=dict(group_splits={'jar_transport/task_0000':'train'},semantic_definitions=definitions(),train_episodes='all',input_contract='chunk_start_v2')
    baseline=build_package([raw],tmp_path/'baseline-package',**options)
    from safetyjev import predictor_judge_dataset
    def forbidden(*args,**kwargs):raise AssertionError('Prepared record was rebuilt')
    monkeypatch.setattr(predictor_judge_dataset,'load_source_record',forbidden)
    reused=build_package([raw],tmp_path/'reused-package',record_cache=out,reuse_annotations=out,**options)
    for key in ('file_sha256','normalization','normalization_statistics','counts'):
        assert baseline[key]==reused[key]
    events=raw/'events.jsonl';events.write_text(events.read_text()+'\n')
    with pytest.raises(ValueError,match='[Ss]ource files changed'):
        load_prepared_record(raw,out)
