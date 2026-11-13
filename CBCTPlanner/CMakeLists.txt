#-----------------------------------------------------------------------------
set(MODULE_NAME CBCTPlanner)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  ${MODULE_NAME}Lib/__init__.py
  ${MODULE_NAME}Lib/Utils.py
  ${MODULE_NAME}Lib/VRVisualization.py
  ${MODULE_NAME}Lib/PanoramicCurveLogic.py
  ${MODULE_NAME}Lib/CrossSectionNavigator.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
