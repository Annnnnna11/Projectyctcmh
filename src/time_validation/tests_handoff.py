"""Safety checks for cross-machine predictions and receipts."""
import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import numpy as np
from common import F,sha
from handoff import validate_prediction,verify_received,validate_bundle,import_bundle,valid_relative

class Checks(unittest.TestCase):
    def test_prediction_reordered_ids_allowed_and_errors_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'p.csv';frame=pd.DataFrame(np.ones((2,28)),columns=F);frame.insert(0,'id',['b','a']);frame.to_csv(p,index=False)
            validate_prediction(p,['a','b'])
            for bad in ['duplicate','missing','nan','negative','column']:
                v=frame.copy()
                if bad=='duplicate':v['id']=['a','a']
                if bad=='missing':v=v.iloc[:1]
                if bad=='nan':v.loc[0,'F1']=np.nan
                if bad=='negative':v.loc[0,'F1']=-1
                if bad=='column':v=v.rename(columns={'F1':'F0'})
                v.to_csv(p,index=False)
                with self.assertRaises(AssertionError):validate_prediction(p,['a','b'])

    def test_receipt_rejects_corruption_and_different_contract(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);p=d/'value.csv';p.write_text('original')
            r=dict(contract={'origin':1857},phase='retrain',stage='test',store='CA_3',mode='recursive',files={'value.csv':sha(p)})
            (d/'received.json').write_text(json.dumps(r))
            with patch('handoff.contract',return_value={'origin':1857}):
                verify_received({},d,'retrain','test','CA_3','recursive')
                p.write_text('corrupted')
                with self.assertRaises(AssertionError):verify_received({},d,'retrain','test','CA_3','recursive')
            with patch('handoff.contract',return_value={'origin':1885}):
                with self.assertRaises(AssertionError):verify_received({},d,'retrain','test','CA_3','recursive')

    def test_paths_reject_escape(self):
        valid_relative('selection/cutoff_1857/recursive/CA_3/predictions_500.csv')
        for bad in ['../secret','/absolute','a/../../secret','C:/secret','a\\secret']:
            with self.assertRaises(AssertionError):valid_relative(bad)

    def test_bundle_rejects_code_or_input_mismatch(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);(d/'bundle.json').write_text(json.dumps(dict(version=1,contract={'inputs':'wrong'})))
            with patch('handoff.contract',return_value={'inputs':'expected'}):
                with self.assertRaises(AssertionError):validate_bundle({},d)

    def test_import_conflicts_do_not_partially_copy(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);bundle=base/'bundle';local=base/'local';bundle.mkdir();local.mkdir()
            (bundle/'first').write_text('first');(bundle/'second').write_text('new');(local/'second').write_text('old')
            r=dict(files={'first':sha(bundle/'first'),'second':sha(bundle/'second')})
            with patch('handoff.validate_bundle',return_value=r),patch('handoff.run_root',return_value=local):
                with self.assertRaises(AssertionError):import_bundle({},bundle)
            self.assertFalse((local/'first').exists());self.assertEqual((local/'second').read_text(),'old')

if __name__=='__main__':unittest.main(verbosity=2)
