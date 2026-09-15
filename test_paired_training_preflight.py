import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from paired_training_preflight import inspect


def table(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class PreflightTests(unittest.TestCase):
    def test_manifest_contract_and_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=root/'plan';plan.mkdir();data=root/'data';data.mkdir();arm=data/'A0';arm.mkdir()
            row=dict(base_id='B00000',k='2',omega_rev_s='0.73',temporal_variant='point',sampling_weight='1')
            table(plan/'A0.csv',[row])
            video=data/'B00000_point.mp4';video.write_bytes(b'fixture-not-real-media')
            (arm/'videos.txt').write_text('../B00000_point.mp4\n')
            (arm/'prompts.txt').write_text('a grayscale wheel with 2 spokes rotating counterclockwise at exactly 0.73 revolutions per second, static camera, white background\n')
            table(data/'encoding_audit.csv',[dict(video=video.name,frames=49,fps=8,width=720,height=480,
                video_sha256=hashlib.sha256(video.read_bytes()).hexdigest())])
            output=root/'weights'
            self.assertTrue(inspect(plan,data,'A0',output)['data_preflight_passed'])
            output.mkdir()
            self.assertIn('training_output_already_exists',inspect(plan,data,'A0',output)['issues'])
            video.write_bytes(b'changed')
            self.assertTrue(any('video_hash_mismatch' in x for x in inspect(plan,data,'A0',output)['issues']))
            table(plan/'A0.csv',[row,dict(row,base_id='B00001')])
            self.assertIn('incomplete_or_wrong_weighted_dataset',inspect(plan,data,'A0',output)['issues'])


if __name__=='__main__':unittest.main()
