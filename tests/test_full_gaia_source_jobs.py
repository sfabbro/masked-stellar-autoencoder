import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('correction_sources', Path(__file__).parents[1]/'scripts/run_full_gaia_correction_sources.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_join_keeps_exact_gaia_ids_and_no_xp_predicate():
    ids=np.array([100000000000000001,100000000000000002],dtype='i8')
    rows,pos=m.match_positions(ids,np.array([ids[1],ids[0],-1,20],dtype='i8'))
    np.testing.assert_array_equal(rows,[0,1]);np.testing.assert_array_equal(pos,[1,0])
    import pytest
    with pytest.raises(ValueError):m.match_positions(ids,ids.astype(float))


def test_stage_preserves_native_age_and_alpha_definitions_and_integer_ids():
    a=np.array([(100000000000000001,3.5,.2)],dtype=[('source_id','>i8'),('Agergb','>f4'),('alpha_m','>f4')])
    frame=m.table_frame(a,list(a.dtype.names),0,1)
    assert frame.source_id.iloc[0]==100000000000000001
    assert frame.Agergb.iloc[0]==3.5
    assert m.native_fields(['Agergb','e_Agerc','alpha_m','mg_fe','source_id'])==['Agergb','e_Agerc','alpha_m','mg_fe']
