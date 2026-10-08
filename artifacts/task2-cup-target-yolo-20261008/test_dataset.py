"""Regression checks for cleaned labels and grouped development partition."""
import hashlib,json,unittest
from pathlib import Path
from prepare import ROOT

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.audit=json.loads((ROOT/'annotation-audit.json').read_text())
    def test_expected_counts(self):
        self.assertEqual(self.audit['labeled_images'],56)
        self.assertEqual(self.audit['class_counts'],{'cup':56,'target':56})
        self.assertEqual(self.audit['split_counts'],{'train':43,'val':13})
        self.assertEqual(self.audit['training_images_with_augmentation'],97)
    def test_episodes_do_not_cross_split(self):
        train={r['episode'] for r in self.audit['records'] if r['split']=='train' and r['episode']}
        val={r['episode'] for r in self.audit['records'] if r['split']=='val' and r['episode']}
        self.assertFalse(train&val)
    def test_no_missing_label_image_used_as_negative(self):
        all_files={p.name for p in (ROOT/'dataset/images').rglob('*.jpg')}
        self.assertFalse(set(self.audit['excluded_missing_labels'])&all_files)
    def test_only_degenerate_annotation_removed(self):
        removed=self.audit['removed_invalid_boxes'];self.assertEqual(len(removed),1)
        self.assertEqual(removed[0]['file'],'024_old_s02_episode_000115_f00287.txt')
        self.assertEqual(removed[0]['original'],'1 0.650257 0.625000 0.000642 0.000000')
    def test_original_images_and_clean_labels_match_audit(self):
        for r in self.audit['records']:
            image=ROOT/'dataset/images'/r['split']/r['filename']
            label=ROOT/'dataset/labels'/r['split']/(image.stem+'.txt')
            self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(),r['source_image_sha256'])
            self.assertEqual(label.read_text(),r['clean_label'])
    def test_augmentations_only_derive_from_training(self):
        train={r['filename'] for r in self.audit['records'] if r['split']=='train'}
        for r in self.audit['derived_training_images']:self.assertIn(r['source'],train)

if __name__=='__main__':unittest.main()
