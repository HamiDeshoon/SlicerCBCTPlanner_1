# ECC Instincts & Karpathy Directives Registry (`SlicerCBCTPlanner`)

## 1. 3D Slicer VTK / MRML Python Binding Rules
- **Trigger**: Accessing `vtkSegment` tags or visibility on `vtkMRMLSegmentationNode`.
- **Observed Pattern**:
  - In 3D Slicer C++ VTK bindings, `segment.GetTag(name, vtk.reference(""))` requires 2 arguments, whereas test mocks may accept 1 argument. Always use a safe helper `safe_get_segment_tag(segment, key)` that tries 2-arg `vtk.reference("")` first and falls back to 1-arg for unit test mocks.
  - Segment visibility lives on `segmentationNode.GetDisplayNode().GetSegmentVisibility(seg_id)` and `SetSegmentVisibility(seg_id, bool)`, NOT on `vtkSegment` directly (keep fallback to `segmentationNode.SetSegmentVisibility` for unit test mocks).
- **Rationale**: Prevents runtime `TypeError` and `AttributeError` in real 3D Slicer that were masked by simplified unit test mocks.

## 2. ToothFairy3 vs. 5-Class DentalSegmentator Disambiguation
- **Trigger**: Combining segmentation segments into a unified label mask (`_extract_combined_label_mask`).
- **Observed Pattern**: Never set `is_5class = True` merely because a segment name contains `"Mandible"` or `"Maxilla"`. A segmentation is 5-class ONLY when it contains no individual FDI tooth labels (`11..48`), no split Left/Right IAN canals (`3, 4`), and at most 5 anatomical groups.
- **Rationale**: Prevents 32-tooth ToothFairy3 masks from being misinterpreted as 5-class masks.

## 3. Dual Pillow + Qt Rendering for Clinical Print & Measured Slices
- **Trigger**: Rendering cross-section slices (`render_cross_section`), panoramic strips (`render_panoramic_strip`), and printable PDF/PNG sheets (`PrintLayoutEngine`).
- **Observed Pattern**: Always draw all 5 clinical overlays (Green Bone Height `H`, Yellow Crest Width `W0`, Cyan `W3`, Orange/Blue `W5`, Red/Cyan Nerve Canal & Clearance, plus tooth/bone contours and HU/Misch badge), and provide a native `qt.QImage`/`qt.QPainter` fallback if `Pillow` is absent. Save print files (`.pdf`, `.png`, `.html`) to an explicit user-accessible folder as well as displaying an inline Measured Image Gallery in the UI.
- **Rationale**: Guarantees measured images and print files are always generated and visible in 3D Slicer.
