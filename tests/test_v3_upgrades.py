from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np


class TestV3Upgrades(unittest.TestCase):
    """Comprehensive verification suite for SlicerCBCTPlanner v3.0 upgrades."""

    def test_labels_vivid_colors_and_scheme_resolution(self):
        from CBCTPlannerLib.core.labels import (
            AnatomicalLabel,
            DENTALSEGMENTATOR_CANAL_ID,
            DENTALSEGMENTATOR_LOWER_TEETH_ID,
            DENTALSEGMENTATOR_MANDIBLE_ID,
            DENTALSEGMENTATOR_MAXILLA_ID,
            DENTALSEGMENTATOR_UPPER_TEETH_ID,
            LABELS_BY_ID,
            get_tooth_label,
            match_segment_to_label_and_precedence,
            resolve_combined_mask_scheme,
            safe_get_segment_tag,
        )

        # 1. Verify AnatomicalLabel has .color attribute on all label definitions
        for lid, lbl in LABELS_BY_ID.items():
            self.assertIsInstance(lbl, AnatomicalLabel)
            self.assertEqual(len(lbl.color), 3, f"Label {lid} missing 3-tuple RGB color")
            self.assertTrue(all(0.0 <= c <= 1.0 for c in lbl.color))

        for ds_id in (
            DENTALSEGMENTATOR_MAXILLA_ID,
            DENTALSEGMENTATOR_MANDIBLE_ID,
            DENTALSEGMENTATOR_UPPER_TEETH_ID,
            DENTALSEGMENTATOR_LOWER_TEETH_ID,
            DENTALSEGMENTATOR_CANAL_ID,
        ):
            self.assertIn(ds_id, LABELS_BY_ID)
            self.assertEqual(len(LABELS_BY_ID[ds_id].color), 3)

        # 2. Verify vivid FDI tooth color grading across quadrants
        lbl_11 = get_tooth_label("11")
        lbl_36 = get_tooth_label("36")
        self.assertIsNotNone(lbl_11)
        self.assertIsNotNone(lbl_36)
        self.assertNotEqual(lbl_11.color, lbl_36.color)

        # 3. Verify 32-tooth ToothFairy3 segmentation is NOT misclassified as 5-class
        tf3_segments = [
            ("seg_1", "Lower Jawbone", None),
            ("seg_2", "Upper Jawbone", None),
            ("seg_3", "Left Inferior Alveolar Canal", None),
            ("seg_11", "Tooth 11", "11"),
            ("seg_36", "Tooth 36", "36"),
            ("seg_46", "Tooth 46", "46"),
        ]
        tf3_records = [
            match_segment_to_label_and_precedence(sid, sname, fdi)
            for sid, sname, fdi in tf3_segments
        ]
        self.assertFalse(resolve_combined_mask_scheme(tf3_records))

        # 4. Verify 5-class DentalSegmentator segmentation IS classified as 5-class
        ds5_segments = [
            ("s1", "Maxilla & Upper Skull", None),
            ("s2", "Mandible", None),
            ("s3", "Upper Teeth", None),
            ("s4", "Lower Teeth", None),
            ("s5", "Mandibular canal", None),
        ]
        ds5_records = [
            match_segment_to_label_and_precedence(sid, sname, fdi)
            for sid, sname, fdi in ds5_segments
        ]
        self.assertTrue(resolve_combined_mask_scheme(ds5_records))

        # 5. Verify safe_get_segment_tag handles both 1-arg mock and 2-arg VTK C++ signature
        mock_seg_1arg = MagicMock()
        mock_seg_1arg.GetTag = lambda k: "36" if k == "CBCTPlanner.FDI" else ""
        self.assertEqual(safe_get_segment_tag(mock_seg_1arg, "CBCTPlanner.FDI"), "36")

    def test_rise_mar_engine_default_path_and_mar_pipeline(self):
        from CBCTPlannerLib.RiseMarEngine import (
            DEFAULT_RISEMAR_WEIGHTS_PATH,
            RiseMarConfig,
            RiseMarEngine,
            detect_metal_affected_slices,
            postprocess_slice_risemar,
            preprocess_slice_risemar,
        )

        # 1. Default checkpoint path matches user specification
        self.assertEqual(
            DEFAULT_RISEMAR_WEIGHTS_PATH,
            r"C:\Users\hamid\Downloads\risemar_dental_preview.pkl",
        )
        cfg = RiseMarConfig()
        self.assertEqual(cfg.weights_path, DEFAULT_RISEMAR_WEIGHTS_PATH)

        # 2. HU normalization [-1024, 3072] <-> [0, 1] roundtrip
        hu = np.array([[-1024.0, 0.0], [1024.0, 3072.0]], dtype=np.float32)
        norm = preprocess_slice_risemar(hu)
        np.testing.assert_allclose(norm, [[0.0, 0.25], [0.5, 1.0]], atol=1e-5)
        rt = postprocess_slice_risemar(norm)
        np.testing.assert_allclose(rt, hu, atol=1e-3)

        # 3. Synthetic dental volume with severe metal crown streak + photon starvation
        rng = np.random.default_rng(42)
        vol = rng.normal(350.0, 45.0, size=(8, 64, 64)).astype(np.float32)
        # Add bright radial streaks and dark starvation bands first, then place metal core (>2500 HU)
        vol[3:6, 31:33, 10:54] = 2150.0
        vol[3:6, 12:52, 31:33] = -850.0
        vol[3:6, 29:35, 29:35] = 3050.0

        metal_slices, metal_voxels = detect_metal_affected_slices(vol, metal_threshold_hu=2500.0)
        self.assertEqual(metal_slices, [3, 4, 5])
        self.assertGreater(metal_voxels, 0)

        engine = RiseMarEngine()
        res = engine.run_risemar_on_volume(vol, config=cfg)

        self.assertEqual(res.output_image_zyx.shape, vol.shape)
        self.assertEqual(res.processed_slices, 3)
        self.assertEqual(res.metal_slice_indices, [3, 4, 5])
        self.assertGreaterEqual(res.cqa_score_after, res.cqa_score_before)
        # High-density metal core (>2500 HU) is preserved
        self.assertTrue(np.all(res.output_image_zyx[3:6, 29:35, 29:35] >= 2500.0))
        # Non-metal slices remain untouched
        np.testing.assert_allclose(res.output_image_zyx[0], vol[0], atol=1e-5)

    def test_automated_dental_tools_engine_all_workflows(self):
        from CBCTPlannerLib.AutomatedDentalToolsEngine import (
            AutomatedDentalToolsEngine,
            CBCTVolume,
        )

        # Create realistic synthetic dental label mask (Z=40, Y=60, X=60)
        mask = np.zeros((40, 60, 60), dtype=np.int16)
        hu = np.full((40, 60, 60), 300.0, dtype=np.float32)

        # Mandible (1) and Maxilla (2)
        mask[5:20, 15:45, 10:50] = 1
        mask[22:35, 15:45, 12:48] = 2
        # Left IAN (3) & Right IAN (4)
        mask[10:14, 20:40, 15:20] = 3
        mask[10:14, 20:40, 40:45] = 4
        # Upper teeth: 11 (11), 21 (21), 13 impacted palatal (13), 23 normal (23)
        mask[20:26, 16:22, 26:30] = 11
        mask[20:26, 16:22, 31:35] = 21
        mask[28:34, 25:31, 22:26] = 13  # Impacted high & palatal right canine
        mask[20:26, 18:24, 36:40] = 23
        # Lower molars: 36 (36), 46 (46)
        mask[15:21, 28:34, 40:45] = 36
        mask[15:21, 28:34, 15:20] = 46

        aff = np.diag([0.5, 0.5, 0.5, 1.0])
        aff[:3, 3] = [-15.0, -15.0, -10.0]
        vol = CBCTVolume(
            image=hu,
            spacing=(0.5, 0.5, 0.5),
            origin=(-15.0, -15.0, -10.0),
            direction=np.eye(3, dtype=np.float64),
            affine=aff,
        )

        engine = AutomatedDentalToolsEngine()

        # 1. ALI landmarks
        landmarks = engine.identify_ali_landmarks(vol, mask=mask, is_5class=False)
        self.assertGreaterEqual(len(landmarks), 8)
        self.assertIn("Me", landmarks)
        self.assertIn("ANS", landmarks)

        # Test Markups JSON export
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "ALI_Landmarks.mrk.json"
            saved = engine.export_ali_landmarks_mrk_json(landmarks, json_path)
            self.assertTrue(Path(saved).exists())
            self.assertIn("@schema", Path(saved).read_text(encoding="utf-8"))

        # 2. ASO standardized orientation
        aso = engine.compute_aso_standardized_orientation(vol, mask=mask, is_5class=False, landmarks=landmarks)
        self.assertEqual(aso.matrix_4x4_ras.shape, (4, 4))

        # 3. CLI-C impaction report
        clic_list = engine.analyze_impacted_teeth_clic(vol, mask=mask, is_5class=False)
        self.assertGreaterEqual(len(clic_list), 1)
        fdis = [c.fdi for c in clic_list]
        self.assertIn("13", fdis)

        # 4. VFACE bilateral symmetry report
        vface = engine.compute_vface_asymmetry_analysis(vol, mask=mask, is_5class=False, landmarks=landmarks)
        self.assertIsNotNone(vface)
        self.assertTrue(len(vface.summary_text) > 0)

        # 5. AutoCrop3D
        center_ras, size_ras = engine.compute_autocrop_roi_bounds_ras(vol, mask=mask)
        self.assertEqual(center_ras.shape, (3,))
        self.assertEqual(size_ras.shape, (3,))
        self.assertTrue(np.all(size_ras > 0.0))

        # 6. AutoMatrix mirror
        mat = engine.build_sagittal_mirror_matrix_4x4(0.0)
        self.assertEqual(mat.shape, (4, 4))
        self.assertEqual(mat[0, 0], -1.0)

    def test_vr_visualization_vivid_presets(self):
        from CBCTPlannerLib.VRVisualization import (
            DENTAL_SEGMENT_PRESETS,
            VIVID_3D_STYLE_PRESETS,
            VRVisualizationLogic,
        )

        self.assertIn("DentalSegmentator Vivid (Teeth & Roots)", VIVID_3D_STYLE_PRESETS)
        self.assertEqual(len(DENTAL_SEGMENT_PRESETS), 5)
        self.assertIn("DentalSegmentator Vivid Roots (VR)", VRVisualizationLogic.SUPPORTED_VR_PRESETS)
        self.assertTrue(hasattr(VRVisualizationLogic, "apply_dental_segmentator_styling"))
        self.assertTrue(hasattr(VRVisualizationLogic, "set_jawbone_3d_opacity"))

    def test_print_cross_section_render_vertical_height_and_widths(self):
        from CBCTPlannerLib.core.measurements import CrossSectionData, ImplantSiteMeasurement
        from CBCTPlannerLib.core_print_render import render_cross_section

        rng = np.random.default_rng(7)
        slice_2d = rng.normal(400.0, 180.0, size=(80, 80)).astype(np.float32)
        seg_2d = np.zeros((80, 80), dtype=np.int16)
        seg_2d[30:65, 25:55] = 1   # Mandible bone
        seg_2d[12:32, 34:46] = 36  # Vivid Tooth #36 crown & root
        seg_2d[56:62, 37:43] = 3   # IAN nerve canal

        u_coords = np.linspace(-16.0, 16.0, 80)
        v_coords = np.linspace(16.0, -16.0, 80)
        frame = (
            np.zeros(3, dtype=np.float64),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

        cs = CrossSectionData(
            u_coords=u_coords,
            v_coords=v_coords,
            mask_cs=seg_2d,
            img_cs=slice_2d,
            frame=frame,
            crest_v=4.0,
            width_segments={
                0.0: (-3.6, 3.6),
                3.0: (-4.2, 4.2),
                5.0: (-4.5, 4.5),
            },
            canal_centroid_uv=(0.0, -6.4),
            basal_v=-10.0,
            height_u=0.0,
            is_mandible=True,
        )
        meas = ImplantSiteMeasurement(
            tooth_fdi="36",
            bone_height_mm=10.4,
            bone_width_at_depths_mm={0.0: 7.2, 3.0: 8.4, 5.0: 9.1},
            crest_width_mm=7.2,
            mid_root_width_mm=8.4,
            apical_width_mm=9.1,
            mesiodistal_space_mm=9.5,
            mean_density_hu=620.0,
            misch_density_class="D2",
            distance_to_vital_structure_mm=10.4,
            vital_structure_name="Left IAN",
            safety_margin_mm=8.4,
            suggested_diameter_mm=4.3,
            suggested_length_mm=10.0,
            suggested_angulation_deg=0.0,
        )
        rendered = render_cross_section(cs, meas, scale=2, dark_film=True)
        self.assertIsNotNone(rendered.image)
        arr = np.asarray(rendered.image)
        # Verify vivid green Height line and cyan Width lines are drawn on the rendered cross-section
        has_vivid_green = np.any((arr[:, :, 1] > 210) & (arr[:, :, 0] < 90))
        has_vivid_cyan = np.any((arr[:, :, 1] > 180) & (arr[:, :, 2] > 210) & (arr[:, :, 0] < 70))
        self.assertTrue(has_vivid_green, "Expected vivid green Bone Height H line in rendered cross-section")
        self.assertTrue(has_vivid_cyan, "Expected vivid cyan Width W0/W3/W5 lines in rendered cross-section")


if __name__ == "__main__":
    unittest.main()
