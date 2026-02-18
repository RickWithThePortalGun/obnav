# test_core.py
from core import sector_from_box
def test_sector():
    assert sector_from_box((0,0,100,100,0.9,1), 300)=="left"
    assert sector_from_box((100,0,200,100,0.9,1), 300)=="front"
    assert sector_from_box((210,0,290,100,0.9,1), 300)=="right"