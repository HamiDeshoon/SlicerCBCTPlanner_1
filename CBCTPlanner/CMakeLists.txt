#-----------------------------------------------------------------------------
set(MODULE_NAME CBCTPlanner)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  cli_batch.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/Utils.py
  ${MODULE_NAME}Lib/VRVisualization.py
  ${MODULE_NAME}Lib/PanoramicCurveLogic.py
  ${MODULE_NAME}Lib/CrossSectionNavigator.py
  ${MODULE_NAME}Lib/CanalTracer.py
  ${MODULE_NAME}Lib/ImplantPlanner.py
  ${MODULE_NAME}Lib/AdvancedTools.py
  ${MODULE_NAME}Lib/CheckpointInference.py
  ${MODULE_NAME}Lib/Phase7Tools.py
  ${MODULE_NAME}Lib/core/__init__.py
  ${MODULE_NAME}Lib/core/arch_detector.py
  ${MODULE_NAME}Lib/core/labels.py
  ${MODULE_NAME}Lib/core/mask_centerline.py
  ${MODULE_NAME}Lib/core/measurements.py
  ${MODULE_NAME}Lib/core/tooth_localizer.py
  ${MODULE_NAME}Lib/core/volume_loader.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/generic_implant_library.json
  Resources/generic_prosthetic_library.json
  Resources/printer_presets.json
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
