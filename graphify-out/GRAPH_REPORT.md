# Graphify Knowledge Graph Report — SlicerCBCTPlanner

Generated: `2026-11-27T09:27:28.813332+00:00`

## Summary
- **Nodes**: 1039 (40 source modules, 27 test modules, 3 harnesses, 15 MRML node types, 54 external/stdlib imports)
- **Edges**: 2897 (`EXTRACTED`, `INFERRED`, `AMBIGUOUS`)
- **Module Import Cycles (SCCs)**: 1

## Top 12 God Nodes (Centrality & Coupling)
| Module | Degree | PageRank | LOC |
|---|---:|---:|---:|
| `CBCTPlannerLib.core.labels` | 21 | 0.1133 | 762 |
| `CBCTPlannerLib.core.volume_loader` | 21 | 0.0894 | 270 |
| `CBCTPlanner` | 21 | 0.0205 | 9326 |
| `CBCTPlannerLib.core.measurements` | 15 | 0.0375 | 3376 |
| `CBCTPlannerLib.Phase7Tools` | 15 | 0.0187 | 810 |
| `CBCTPlannerLib.core.arch_detector` | 14 | 0.0382 | 1569 |
| `CBCTPlannerLib` | 14 | 0.0113 | 214 |
| `CBCTPlannerLib.PrintLayoutEngine` | 11 | 0.0160 | 1293 |
| `CBCTPlannerLib.CheckpointInference` | 10 | 0.0272 | 701 |
| `CBCTPlannerLib.CanalTracer` | 10 | 0.0176 | 389 |
| `CBCTPlannerLib.adt` | 10 | 0.0157 | 218 |
| `CBCTPlannerLib.PanoramicCurveLogic` | 9 | 0.0238 | 1279 |

## Architectural Communities (Greedy Modularity)
- **Cluster 1** (16 modules): `CBCTPlanner`, `CBCTPlannerLib.EndodonticEngine`, `CBCTPlannerLib.ImplantPlanner`, `CBCTPlannerLib.PrintLayoutEngine`, `CBCTPlannerLib.RiseMarEngine`, `CBCTPlannerLib.WorkflowUIHelpers`, `CBCTPlannerLib.core.agent_review`, `CBCTPlannerLib.core.annotations`, `CBCTPlannerLib.core.case_state`, `CBCTPlannerLib.core.contracts`, `CBCTPlannerLib.core.diagnostics`, `CBCTPlannerLib.core.labels`, `CBCTPlannerLib.core.measurements`, `CBCTPlannerLib.core.print_layout`, `CBCTPlannerLib.core.tangential`, `CBCTPlannerLib.core_print_render`
- **Cluster 2** (11 modules): `CBCTPlannerLib.AutomatedDentalToolsEngine`, `CBCTPlannerLib.adt`, `CBCTPlannerLib.adt.ali`, `CBCTPlannerLib.adt.aso`, `CBCTPlannerLib.adt.clic`, `CBCTPlannerLib.adt.doc_shapeaxi`, `CBCTPlannerLib.adt.geometry`, `CBCTPlannerLib.adt.vface`, `CBCTPlannerLib.core.orientation`, `CBCTPlannerLib.core.tooth_localizer`, `CBCTPlannerLib.core.volume_loader`
- **Cluster 3** (9 modules): `CBCTPlannerLib`, `CBCTPlannerLib.AdvancedTools`, `CBCTPlannerLib.CheckpointInference`, `CBCTPlannerLib.CrossSectionNavigator`, `CBCTPlannerLib.PanoramicCurveLogic`, `CBCTPlannerLib.Phase7Tools`, `CBCTPlannerLib.Utils`, `CBCTPlannerLib.VRVisualization`, `cli_batch`
- **Cluster 4** (4 modules): `CBCTPlannerLib.CanalTracer`, `CBCTPlannerLib.core`, `CBCTPlannerLib.core.arch_detector`, `CBCTPlannerLib.core.mask_centerline`

## Circular Dependencies
- `CBCTPlanner -> CBCTPlannerLib -> CBCTPlannerLib.Phase7Tools`

## MRML Node Ownership & Lifecycle
- `vtkMRMLLayoutNode`: `CBCTPlannerLib.Utils`
- `vtkMRMLLinearTransformNode`: `CBCTPlanner`
- `vtkMRMLMarkupsCurveNode`: `CBCTPlanner`, `CBCTPlannerLib.CanalTracer`, `CBCTPlannerLib.PanoramicCurveLogic`
- `vtkMRMLMarkupsDisplayNode`: `CBCTPlannerLib.CanalTracer`, `CBCTPlannerLib.CrossSectionNavigator`, `CBCTPlannerLib.PanoramicCurveLogic`, `tests.test_slicer_integration`
- `vtkMRMLMarkupsFiducialNode`: `CBCTPlanner`, `CBCTPlannerLib.AdvancedTools`, `analysis.mcp_slicer_check.verify_mcp_slicer`
- `vtkMRMLMarkupsLineNode`: `CBCTPlanner`, `CBCTPlannerLib.CrossSectionNavigator`, `CBCTPlannerLib.ImplantPlanner`
- `vtkMRMLMarkupsNode`: `CBCTPlanner`, `CBCTPlannerLib.CrossSectionNavigator`
- `vtkMRMLMarkupsROINode`: `CBCTPlanner`
- `vtkMRMLModelNode`: `CBCTPlannerLib.CrossSectionNavigator`, `CBCTPlannerLib.ImplantPlanner`
- `vtkMRMLScalarVolumeNode`: `CBCTPlanner`, `CBCTPlannerLib.PanoramicCurveLogic`, `CBCTPlannerLib.RiseMarEngine`, `CBCTPlannerLib.core.volume_loader`, `analysis.mcp_slicer_check.slicer_webserver_boot`, `tests.test_slicer_integration`
- `vtkMRMLScriptedModuleNode`: `tests.test_slicer_integration`
- `vtkMRMLSegmentationNode`: `CBCTPlanner`, `CBCTPlannerLib.AdvancedTools`, `CBCTPlannerLib.CheckpointInference`
- `vtkMRMLSliceNode`: `CBCTPlannerLib.Utils`, `tests.test_slicer_integration`
- `vtkMRMLViewNode`: `CBCTPlannerLib.Utils`
- `vtkMRMLVolumeRenderingDisplayNode`: `CBCTPlannerLib.VRVisualization`

## Isolated Source Modules
- None
