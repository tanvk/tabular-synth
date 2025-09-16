import pandas as pd
from synth.copula import CopulaGenerator
from eval.fidelity import basic_fidelity_report

def test_copula_sample_shapes():
    df = pd.DataFrame({"age":[25,30,35,40], "workclass":["Private","Self","Private","Gov"]})
    gen = CopulaGenerator().fit(df)
    out = gen.sample(8)
    assert out.shape == (8, 2)

def test_fidelity_runs():
    df = pd.DataFrame({"x":[1,2,3,4,5], "y":["a","b","a","b","a"]})
    gen = CopulaGenerator().fit(df)
    syn = gen.sample(len(df))
    rpt = basic_fidelity_report(df, syn)
    assert "headline" in rpt and "univariate" in rpt