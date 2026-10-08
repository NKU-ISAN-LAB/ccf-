import copy,unittest
from recognize import StabilityGate

class BoundaryTests(unittest.TestCase):
    def candidate(self,point=(171,265)):
        return dict(candidate_valid_2d=True,visible_lower_rim_uv=list(point))

    def test_three_new_frames(self):
        gate=StabilityGate()
        self.assertFalse(gate.update(self.candidate(),1,100.)['stable_2d'])
        self.assertFalse(gate.update(self.candidate(),2,100.2)['stable_2d'])
        self.assertTrue(gate.update(self.candidate(),3,100.4)['stable_2d'])

    def test_duplicate_and_absence_reset_history(self):
        gate=StabilityGate()
        for i in range(3):gate.update(self.candidate(),i,100+i*.2)
        self.assertFalse(gate.update(self.candidate(),2,100.5)['stable_2d'])
        self.assertFalse(gate.update(self.candidate(),3,100.6)['stable_2d'])
        self.assertFalse(gate.update(dict(candidate_valid_2d=False),4,100.8)['stable_2d'])
        self.assertFalse(gate.update(self.candidate(),5,101.)['stable_2d'])

    def test_gap_and_view_change_not_stable(self):
        gate=StabilityGate()
        gate.update(self.candidate(),1,100.)
        gate.update(self.candidate(),2,100.2)
        self.assertFalse(gate.update(self.candidate((220,190)),3,100.4)['stable_2d'])
        self.assertFalse(gate.update(self.candidate(),4,105.)['stable_2d'])

    def test_nonfinite_timestamp(self):
        self.assertFalse(StabilityGate().update(self.candidate(),1,float('nan'))['stable_2d'])

if __name__=='__main__':unittest.main()
