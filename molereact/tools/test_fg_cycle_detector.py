import unittest
import sys
import os

# Adjust path to enable importing from multistep
current_dir = os.path.dirname(os.path.abspath(__file__))
multistep_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if multistep_root not in sys.path:
    sys.path.insert(0, multistep_root)

from multistep.agent.tools.fg_cycle_detector import FGStateTracker, CycleDetector, FG_EQUIVALENCE_CLASSES

class TestFGCycleDetector(unittest.TestCase):
    def setUp(self):
        self.detector = CycleDetector()
        
    def test_signatures(self):
        # Acid
        smi_acid = "CC(=O)O"
        fgs_acid = ["Acid"]
        eq_sig = FGStateTracker.generate_equivalence_state_signature(smi_acid, fgs_acid)
        ex_sig = FGStateTracker.generate_exact_state_signature(smi_acid, fgs_acid)
        
        # Expect: Acyclic|carbonyl_acid_derivative|carbonyl_ox_state (Acid matches multiple?)
        # Let's check logic: "sorted(list(cls.get_active_equivalence_classes(present_fgs)))"
        # Acid is in 'carbonyl_acid_derivative' AND 'carbonyl_ox_state'.
        self.assertIn("Acyclic", eq_sig)
        self.assertIn("carbonyl_acid_derivative", eq_sig)
        self.assertIn("Acid", ex_sig)
        
        # Chloride
        smi_cl = "CC(=O)Cl"
        fgs_cl = ["AcidChloride"]
        eq_sig_cl = FGStateTracker.generate_equivalence_state_signature(smi_cl, fgs_cl)
        # AcidChloride is in 'carbonyl_acid_derivative' only?
        self.assertIn("carbonyl_acid_derivative", eq_sig_cl)
        
        # Verify Acid and Chloride share a class in signature
        self.assertTrue("carbonyl_acid_derivative" in eq_sig and "carbonyl_acid_derivative" in eq_sig_cl)

    def test_exact_cycle_detection(self):
        # Hist: Acid -> Ester
        # Curr: Acid
        hist = [
            {"exact_state_signature": "S|Acid", "fg_equivalence_state": "S|Deriv"},
            {"exact_state_signature": "S|Ester", "fg_equivalence_state": "S|Deriv"}
        ]
        curr_exact = "S|Acid"
        curr_eq = "S|Deriv"
        
        res = self.detector.detect_complex_cycle(hist, curr_eq, curr_exact)
        self.assertTrue(res["is_cycle"])
        self.assertEqual(res["loop_risk"], "HIGH")
        self.assertEqual(res["cycle_length"], 2)

    def test_equivalence_stagnation(self):
        # Acid -> Chloride (Different FG, Same Class)
        hist = [
            {"exact_state_signature": "S|Acid", "fg_equivalence_state": "S|Deriv"}
        ]
        curr_exact = "S|AcidChloride" 
        curr_eq = "S|Deriv" 
        
        res = self.detector.detect_complex_cycle(hist, curr_eq, curr_exact)
        self.assertFalse(res["is_cycle"]) 
        self.assertEqual(res["loop_risk"], "MED")
        self.assertIn("Stagnation", res["loop_reason"])

    def test_complex_cycle(self):
        # Acid -> Chloride -> Ester -> Amide -> Acid
        hist = [
            {"exact_state_signature": "S|Acid", "fg_equivalence_state": "S|Deriv"},
            {"exact_state_signature": "S|Chlor", "fg_equivalence_state": "S|Deriv"},
            {"exact_state_signature": "S|Ester", "fg_equivalence_state": "S|Deriv"},
            {"exact_state_signature": "S|Amide", "fg_equivalence_state": "S|Deriv"}
        ]
        curr_exact = "S|Acid"
        curr_eq = "S|Deriv"
        
        res = self.detector.detect_complex_cycle(hist, curr_eq, curr_exact)
        self.assertTrue(res["is_cycle"])
        self.assertEqual(res["loop_risk"], "HIGH")
        self.assertEqual(res["cycle_length"], 4)

    def test_progress_out_of_class(self):
        # Acid -> Alcohol (Reduction, exits class)
        hist = [
            {"exact_state_signature": "S|Acid", "fg_equivalence_state": "S|AcidDeriv"}
        ]
        curr_exact = "S|Alcohol"
        curr_eq = "S|AlcoholClass" 
        
        res = self.detector.detect_complex_cycle(hist, curr_eq, curr_exact)
        self.assertEqual(res["loop_risk"], "LOW")
        self.assertEqual(res["fg_effectiveness"], "HIGH")

if __name__ == '__main__':
    unittest.main()
