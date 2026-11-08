"""Checks for approved research price supplementation; network disabled."""
import io
import sys
import unittest
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from entry_price_supplement_20261028 import BAR, DAY, dates, parse_zip, validate


class PriceSupplementTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(open_time_ms=i*BAR,close_time_ms=(i+1)*BAR-1,
                                  open=100.,high=101.,low=99.,close=100.) for i in range(145)])

    def event(self):
        return dict(bar_close_time_ms=BAR-1,source_price=100.,entry_price=100.)

    def test_exact_full_window_valid(self):
        frame,status=validate(self.frame(),self.event())
        self.assertEqual(status,'VALID'); self.assertEqual(len(frame),145)

    def test_missing_duplicate_and_wrong_time_not_imputed(self):
        f=self.frame(); self.assertIsNone(validate(f.drop(index=5),self.event())[0])
        self.assertEqual(validate(pd.concat([f,f.iloc[[3]]]),self.event())[1],'DUPLICATE_TIME')
        f.loc[1,'close_time_ms']-=1
        self.assertEqual(validate(f,self.event())[1],'INVALID_CLOSE_TIME')

    def test_reference_mismatch_is_not_rounded_away(self):
        f=self.frame(); f.loc[0,'close']=100.01
        self.assertEqual(validate(f,self.event())[1],'SIGNAL_PRICE_CONFLICT')
        f=self.frame(); f.loc[1,'open']=100.01
        self.assertEqual(validate(f,self.event())[1],'ENTRY_PRICE_CONFLICT')

    def test_bad_ohlc_rejected(self):
        f=self.frame(); f.loc[2,'high']=98.
        self.assertEqual(validate(f,self.event())[1],'INVALID_RANGE')
        f=self.frame(); f.loc[2,'close']=float('nan')
        self.assertEqual(validate(f,self.event())[1],'NONFINITE')

    def test_parse_binance_header_without_extracting_files(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:
            z.writestr('A-5m-1970-01-01.csv','open_time,open,high,low,close,volume,close_time\n0,100,101,99,100,1,299999\n')
        f=parse_zip(data.getvalue())
        self.assertEqual(f.iloc[0]['open_time_ms'],0)
        self.assertEqual(f.iloc[0]['close_time_ms'],BAR-1)

    def test_ambiguous_archive_not_accepted(self):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:
            z.writestr('a.csv','0,1,2,1,1,0,299999')
            z.writestr('b.csv','0,1,2,1,1,0,299999')
        with self.assertRaises(ValueError): parse_zip(data.getvalue())

    def test_only_required_utc_days(self):
        self.assertEqual(dates(dict(bar_close_time_ms=20*3600000+BAR-1)),['1970-01-01','1970-01-02'])
        self.assertEqual(dates(dict(bar_close_time_ms=BAR-1)),['1970-01-01'])


if __name__=='__main__':
    unittest.main()
