cmake_minimum_required(VERSION 3.5)
project(ner_model)

find_package(catkin REQUIRED COMPONENTS rospy)

catkin_python_setup()

catkin_package()

# The weights (~433 MB) are not in git, so download them into data/, where ner_model looks for them.
# Skipped when they are already there, and in CI, which doesn't need them.
if(NOT "$ENV{CI}" STREQUAL "true")
  file(
    DOWNLOAD https://huggingface.co/PanosHuggingFace/bert-ner/resolve/main/model.pth ${PROJECT_SOURCE_DIR}/data/model.pth
    EXPECTED_HASH "SHA256=07298a9f6c874396cbb29d3b7bde3f0b17a9b6c39893bd82547b498677b165fb"
    SHOW_PROGRESS
  )
endif()

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test)
endif()
