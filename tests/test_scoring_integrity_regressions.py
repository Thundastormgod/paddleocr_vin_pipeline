"""
Regression tests for scoring-path integrity: split leakage (H1) and the
VIN-corrupting artifact strip (H2).

Both defects are instances of this repository's documented failure mode -
a fix applied in one copy of a concept and left broken in another:

- VIN-grouped splitting was implemented in utils/prepare_dataset.py while
  evaluation/evaluate.py:create_splits still shuffled image PATHS, so the
  SCORING path leaked plates between train and test. Reproduced before the
  fix: 5 VINs x 4 images put the same VIN in train, val and test at once.
- Artifact stripping was fixed in pipeline.VINPostProcessor while
  core.vin_utils.RuleBasedCorrector kept the pattern `^[*#XYT]+`, which
  deleted the first character of every VIN starting with X, Y or T.
  Reproduced before the fix: correct_vin("YV1MS390X72123456") returned the
  16-character "V1MS390X72123456".

The identity tests at the bottom pin the single-source-of-truth wiring so a
third copy cannot silently reappear.
"""

import os

import pytest

from src.vin_ocr.core import vin_utils
from src.vin_ocr.core.vin_utils import (
    ARTIFACT_CHARS,
    NON_VIN_RUN,
    RuleBasedCorrector,
    correct_vin,
)
from src.vin_ocr.evaluation.evaluate import create_splits
from src.vin_ocr.pipeline import vin_pipeline

# Five real-format VINs (all with valid ISO-3779 check digits where noted);
# what matters here is five DISTINCT plates with four images each.
VINS = [
    "1M8GDM9AXKP042788",
    "1HGCM82633A004352",
    "11111111111111111",
    "5YJ3E1EA7KF317231",
    "WVWZZZ1JZXW000010",
]


def _ground_truth(vins=VINS, images_per_vin=4):
    return {
        f"/data/{vin}_{k}.jpg": vin
        for vin in vins
        for k in range(images_per_vin)
    }


class TestCreateSplitsGroupsByVIN:
    """
    WAS BROKEN: create_splits shuffled ground_truth.keys() - image paths -
    so multiple images of one plate landed in different splits and the model
    was scored on plates it trained on.
    """

    def test_no_vin_appears_in_more_than_one_split(self):
        splits = create_splits(_ground_truth(), output_dir=None)

        vin_sets = {
            name: set(split.ground_truths.values())
            for name, split in splits.items()
            if name != 'all'
        }
        names = sorted(vin_sets)
        overlaps = {
            (a, b): vin_sets[a] & vin_sets[b]
            for i, a in enumerate(names)
            for b in names[i + 1:]
            if vin_sets[a] & vin_sets[b]
        }
        assert overlaps == {}, f"VINs leaked between splits: {overlaps}"

    def test_every_image_lands_in_exactly_one_split(self):
        gt = _ground_truth()
        splits = create_splits(gt, output_dir=None)

        parts = [
            set(splits[name].image_paths) for name in ('train', 'val', 'test')
        ]
        union = set().union(*parts)
        assert union == set(gt), "images lost or invented by splitting"
        total = sum(len(p) for p in parts)
        assert total == len(gt), "an image appears in more than one split"

    def test_all_images_of_a_vin_stay_together(self):
        gt = _ground_truth()
        splits = create_splits(gt, output_dir=None)

        for name in ('train', 'val', 'test'):
            split = splits[name]
            for vin in set(split.ground_truths.values()):
                images_of_vin = [p for p, v in gt.items() if v == vin]
                assert all(p in split.ground_truths for p in images_of_vin), (
                    f"images of {vin} split across sets"
                )

    def test_same_seed_gives_identical_partition(self):
        a = create_splits(_ground_truth(), seed=42, output_dir=None)
        b = create_splits(_ground_truth(), seed=42, output_dir=None)
        for name in ('train', 'val', 'test'):
            assert set(a[name].image_paths) == set(b[name].image_paths)

    def test_all_split_still_carries_everything(self):
        gt = _ground_truth()
        splits = create_splits(gt, output_dir=None)
        assert splits['all'].ground_truths == gt
        assert set(splits['all'].image_paths) == set(gt)

    def test_split_files_are_still_written(self, tmp_path):
        create_splits(_ground_truth(), output_dir=str(tmp_path))
        for name in ('train', 'val', 'test'):
            f = tmp_path / f"{name}_split.txt"
            assert f.is_file(), f"{name}_split.txt not written"
            for line in f.read_text().splitlines():
                fname, vin = line.split('\t')
                assert vin in VINS
                assert fname.startswith(vin)

    def test_partition_matches_the_canonical_splitter(self):
        """The scoring path and the dataset-prep path use ONE split."""
        from src.vin_ocr.utils.prepare_dataset import (
            create_splits as canonical,
        )

        gt = _ground_truth()
        train_gt, val_gt, test_gt = canonical(gt, seed=42)
        splits = create_splits(gt, seed=42, output_dir=None)

        assert splits['train'].ground_truths == train_gt
        assert splits['val'].ground_truths == val_gt
        assert splits['test'].ground_truths == test_gt


class TestCorrectorPreservesValidVINCharacters:
    """
    WAS BROKEN: RuleBasedCorrector.ARTIFACT_PATTERNS contained `^[*#XYT]+`,
    which stripped leading X/Y/T - valid VIN characters - and returned
    16-character results for legitimate VINs.
    """

    def test_volvo_vin_keeps_its_leading_y(self):
        vin = "YV1MS390X72123456"
        result = correct_vin(vin)
        assert result['vin'] == vin, (
            f"leading 'Y' eaten: {result['vin']!r} ({result['corrections']})"
        )

    @pytest.mark.parametrize("vin", [
        "XL9AA11G96Z363495",   # leading X (Spyker style WMI shape)
        "TRUZZZ8N841000000",   # leading T (Audi Hungaria WMI shape)
        "YS3DD55H1Y2000000",   # leading Y (Saab WMI shape)
    ])
    def test_leading_x_t_y_survive(self, vin):
        assert correct_vin(vin)['vin'] == vin

    def test_leading_i_is_corrected_not_deleted(self):
        """'I' is invalid IN a VIN but is evidence of a '1', not noise."""
        result = correct_vin("IM8GDM9AXKP042788")
        assert result['vin'] == "1M8GDM9AXKP042788"

    def test_real_artifacts_are_still_stripped(self):
        result = correct_vin("**YV1MS390X72123456##")
        assert result['vin'] == "YV1MS390X72123456"

    def test_artifacts_and_whitespace_around_plain_vin(self):
        result = correct_vin("* 1M8GDM9AXKP042788 #")
        assert result['vin'] == "1M8GDM9AXKP042788"

    def test_corrector_no_longer_carries_letter_stripping_patterns(self):
        """The defective class attributes are gone, not just unused."""
        assert not hasattr(RuleBasedCorrector, 'ARTIFACT_PATTERNS')


class TestArtifactDefinitionHasOneSource:
    """
    The Y-eating bug survived its first fix because the artifact rule
    existed in two places. These pin every consumer to the ONE definition
    in core.vin_utils, by object identity.
    """

    def test_pipeline_uses_the_canonical_regex(self):
        assert vin_pipeline._NON_VIN_RUN is NON_VIN_RUN

    def test_pipeline_reexports_the_canonical_charset(self):
        assert vin_pipeline.ARTIFACT_CHARS is ARTIFACT_CHARS

    def test_multi_model_evaluation_uses_the_canonical_regex(self):
        from src.vin_ocr.evaluation.multi_model_evaluation import (
            VINCharValidator,
        )
        assert VINCharValidator._NON_VIN_RUN is NON_VIN_RUN

    def test_no_artifact_definition_strips_letters(self):
        """Every valid VIN character must survive the canonical strip."""
        valid = "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"
        assert NON_VIN_RUN.sub('', valid) == valid
        assert not (set(valid) & ARTIFACT_CHARS)

    def test_postprocessor_and_corrector_agree_on_stripping(self):
        pp = vin_pipeline.VINPostProcessor()
        rc = vin_utils.get_corrector() if hasattr(vin_utils, 'get_corrector') \
            else RuleBasedCorrector()
        for noisy in [
            "**YV1MS390X72123456",
            "*#1M8GDM9AXKP042788#*",
            "TRU-ZZZ 8N8/41000000",
        ]:
            up = noisy.upper()
            assert pp._remove_artifacts(up) == rc._remove_artifacts(up)
