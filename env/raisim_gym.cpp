//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisim;

#ifndef MANO_FIXED_NAME
  #define MANO_FIXED_NAME Mano_Fixed
#endif
#ifndef MANO_FLOATING_NAME
  #define MANO_FLOATING_NAME Mano_Floating
#endif
#ifndef MANO_TEST_NAME
  #define MANO_TEST_NAME Mano_Test
#endif


#ifndef SHADOW_FIXED_NAME
  #define SHADOW_FIXED_NAME Shadow_Fixed
#endif
#ifndef SHADOW_FLOATING_NAME
  #define SHADOW_FLOATING_NAME Shadow_Floating
#endif
#ifndef SHADOW_TEST_NAME
  #define SHADOW_TEST_NAME Shadow_Test
#endif


#ifndef ALLEGRO_FIXED_NAME
  #define ALLEGRO_FIXED_NAME Allegro_Fixed
#endif
#ifndef ALLEGRO_FLOATING_NAME
  #define ALLEGRO_FLOATING_NAME Allegro_Floating
#endif
#ifndef ALLEGRO_TEST_NAME
  #define ALLEGRO_TEST_NAME Allegro_Test
#endif


#ifndef LEAP_FIXED_NAME
  #define LEAP_FIXED_NAME Leap_Fixed
#endif
#ifndef LEAP_FLOATING_NAME
  #define LEAP_FLOATING_NAME Leap_Floating
#endif
#ifndef LEAP_TEST_NAME
  #define LEAP_TEST_NAME Leap_Test
#endif


#ifndef INSPIRE_FLOATING_NAME
  #define INSPIRE_FLOATING_NAME Inspire_Floating
#endif
#ifndef INSPIRE_TEST_NAME
  #define INSPIRE_TEST_NAME Inspire_Test
#endif

#ifndef ABILITY_FLOATING_NAME
  #define ABILITY_FLOATING_NAME Ability_Floating
#endif
#ifndef ABILITY_TEST_NAME
  #define ABILITY_TEST_NAME Ability_Test
#endif

#ifndef SCHUNK_FLOATING_NAME
  #define SCHUNK_FLOATING_NAME Schunk_Floating
#endif
#ifndef SCHUNK_TEST_NAME
  #define SCHUNK_TEST_NAME Schunk_Test
#endif

#ifndef BRAINCO_FLOATING_NAME
  #define BRAINCO_FLOATING_NAME Brainco_Floating
#endif
#ifndef BRAINCO_TEST_NAME
  #define BRAINCO_TEST_NAME Brainco_Test
#endif
#ifndef BARRETT_TEST_NAME
  #define BARRETT_TEST_NAME Barrett_Test
#endif
// environment.hpp -> env name:
// SHADOW_FIXED
// MANO_FIXED
// ALLEGRO_FIXED
// ALLEGRO_FLOATING
// SHADOW_FLOATING
// MANO_FLOATING
PYBIND11_MODULE(RAISIMGYM_TORCH_ENV_NAME, m) {
  py::class_<NormalSampler>(m, "NormalSampler")
    .def(py::init<int>(), py::arg("dim"))
    .def("seed", &NormalSampler::seed)
    .def("sample", &NormalSampler::sample);
  //for shadow
  py::class_<VectorizedEnvironment<SHADOW_FIXED>>(m, RSG_MAKE_STR(SHADOW_FIXED_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<SHADOW_FIXED>::init)
    .def("reset", &VectorizedEnvironment<SHADOW_FIXED>::reset)
    .def("add_stage", &VectorizedEnvironment<SHADOW_FIXED>::add_stage)
    .def("load_object", &VectorizedEnvironment<SHADOW_FIXED>::load_object)
    .def("load_articulated", &VectorizedEnvironment<SHADOW_FIXED>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<SHADOW_FIXED>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<SHADOW_FIXED>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<SHADOW_FIXED>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<SHADOW_FIXED>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<SHADOW_FIXED>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<SHADOW_FIXED>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<SHADOW_FIXED>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<SHADOW_FIXED>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<SHADOW_FIXED>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<SHADOW_FIXED>::observe)
    .def("get_global_state", &VectorizedEnvironment<SHADOW_FIXED>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<SHADOW_FIXED>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<SHADOW_FIXED>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<SHADOW_FIXED>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<SHADOW_FIXED>::control_switch_all)
    .def("step", &VectorizedEnvironment<SHADOW_FIXED>::step)
    .def("step2", &VectorizedEnvironment<SHADOW_FIXED>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<SHADOW_FIXED>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<SHADOW_FIXED>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<SHADOW_FIXED>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<SHADOW_FIXED>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<SHADOW_FIXED>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<SHADOW_FIXED>::close)
    .def("isTerminalState", &VectorizedEnvironment<SHADOW_FIXED>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<SHADOW_FIXED>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<SHADOW_FIXED>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<SHADOW_FIXED>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<SHADOW_FIXED>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<SHADOW_FIXED>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<SHADOW_FIXED>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<SHADOW_FIXED>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<SHADOW_FIXED>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<SHADOW_FIXED>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<SHADOW_FIXED>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<SHADOW_FIXED>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<SHADOW_FIXED>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<SHADOW_FIXED>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<SHADOW_FIXED>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<SHADOW_FIXED>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<SHADOW_FIXED> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<SHADOW_FIXED> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
    ));
  py::class_<VectorizedEnvironment<MANO_FIXED>>(m, RSG_MAKE_STR(MANO_FIXED_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<MANO_FIXED>::init)
    .def("reset", &VectorizedEnvironment<MANO_FIXED>::reset)
    .def("add_stage", &VectorizedEnvironment<MANO_FIXED>::add_stage)
    .def("load_object", &VectorizedEnvironment<MANO_FIXED>::load_object)
    .def("load_articulated", &VectorizedEnvironment<MANO_FIXED>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<MANO_FIXED>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<MANO_FIXED>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<MANO_FIXED>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<MANO_FIXED>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<MANO_FIXED>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<MANO_FIXED>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<MANO_FIXED>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<MANO_FIXED>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<MANO_FIXED>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<MANO_FIXED>::observe)
    .def("get_global_state", &VectorizedEnvironment<MANO_FIXED>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<MANO_FIXED>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<MANO_FIXED>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<MANO_FIXED>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<MANO_FIXED>::control_switch_all)
    .def("step", &VectorizedEnvironment<MANO_FIXED>::step)
    .def("step2", &VectorizedEnvironment<MANO_FIXED>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<MANO_FIXED>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<MANO_FIXED>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<MANO_FIXED>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<MANO_FIXED>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<MANO_FIXED>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<MANO_FIXED>::close)
    .def("isTerminalState", &VectorizedEnvironment<MANO_FIXED>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<MANO_FIXED>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<MANO_FIXED>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<MANO_FIXED>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<MANO_FIXED>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<MANO_FIXED>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<MANO_FIXED>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<MANO_FIXED>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<MANO_FIXED>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<MANO_FIXED>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<MANO_FIXED>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<MANO_FIXED>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<MANO_FIXED>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<MANO_FIXED>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<MANO_FIXED>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<MANO_FIXED>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<MANO_FIXED> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<MANO_FIXED> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
    ));

  py::class_<VectorizedEnvironment<ALLEGRO_FIXED>>(m, RSG_MAKE_STR(ALLEGRO_FIXED_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<ALLEGRO_FIXED>::init)
    .def("reset", &VectorizedEnvironment<ALLEGRO_FIXED>::reset)
    .def("add_stage", &VectorizedEnvironment<ALLEGRO_FIXED>::add_stage)
    .def("load_object", &VectorizedEnvironment<ALLEGRO_FIXED>::load_object)
    .def("load_articulated", &VectorizedEnvironment<ALLEGRO_FIXED>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<ALLEGRO_FIXED>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<ALLEGRO_FIXED>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<ALLEGRO_FIXED>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<ALLEGRO_FIXED>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<ALLEGRO_FIXED>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<ALLEGRO_FIXED>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<ALLEGRO_FIXED>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<ALLEGRO_FIXED>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<ALLEGRO_FIXED>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<ALLEGRO_FIXED>::observe)
    .def("get_global_state", &VectorizedEnvironment<ALLEGRO_FIXED>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<ALLEGRO_FIXED>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<ALLEGRO_FIXED>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<ALLEGRO_FIXED>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<ALLEGRO_FIXED>::control_switch_all)
    .def("step", &VectorizedEnvironment<ALLEGRO_FIXED>::step)
    .def("step2", &VectorizedEnvironment<ALLEGRO_FIXED>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<ALLEGRO_FIXED>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<ALLEGRO_FIXED>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<ALLEGRO_FIXED>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<ALLEGRO_FIXED>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<ALLEGRO_FIXED>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<ALLEGRO_FIXED>::close)
    .def("isTerminalState", &VectorizedEnvironment<ALLEGRO_FIXED>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ALLEGRO_FIXED>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ALLEGRO_FIXED>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<ALLEGRO_FIXED>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<ALLEGRO_FIXED>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<ALLEGRO_FIXED>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<ALLEGRO_FIXED>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ALLEGRO_FIXED>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<ALLEGRO_FIXED>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<ALLEGRO_FIXED>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<ALLEGRO_FIXED>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<ALLEGRO_FIXED>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<ALLEGRO_FIXED>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<ALLEGRO_FIXED>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<ALLEGRO_FIXED>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<ALLEGRO_FIXED>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<ALLEGRO_FIXED> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<ALLEGRO_FIXED> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
  ));
  py::class_<VectorizedEnvironment<LEAP_FIXED>>(m, RSG_MAKE_STR(LEAP_FIXED_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<LEAP_FIXED>::init)
    .def("reset", &VectorizedEnvironment<LEAP_FIXED>::reset)
    .def("add_stage", &VectorizedEnvironment<LEAP_FIXED>::add_stage)
    .def("load_object", &VectorizedEnvironment<LEAP_FIXED>::load_object)
    .def("load_articulated", &VectorizedEnvironment<LEAP_FIXED>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<LEAP_FIXED>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<LEAP_FIXED>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<LEAP_FIXED>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<LEAP_FIXED>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<LEAP_FIXED>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<LEAP_FIXED>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<LEAP_FIXED>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<LEAP_FIXED>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<LEAP_FIXED>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<LEAP_FIXED>::observe)
    .def("get_global_state", &VectorizedEnvironment<LEAP_FIXED>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<LEAP_FIXED>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<LEAP_FIXED>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<LEAP_FIXED>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<LEAP_FIXED>::control_switch_all)
    .def("step", &VectorizedEnvironment<LEAP_FIXED>::step)
    .def("step2", &VectorizedEnvironment<LEAP_FIXED>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<LEAP_FIXED>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<LEAP_FIXED>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<LEAP_FIXED>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<LEAP_FIXED>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<LEAP_FIXED>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<LEAP_FIXED>::close)
    .def("isTerminalState", &VectorizedEnvironment<LEAP_FIXED>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<LEAP_FIXED>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<LEAP_FIXED>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<LEAP_FIXED>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<LEAP_FIXED>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<LEAP_FIXED>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<LEAP_FIXED>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<LEAP_FIXED>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<LEAP_FIXED>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<LEAP_FIXED>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<LEAP_FIXED>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<LEAP_FIXED>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<LEAP_FIXED>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<LEAP_FIXED>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<LEAP_FIXED>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<LEAP_FIXED>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<LEAP_FIXED> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<LEAP_FIXED> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
  ));
  py::class_<VectorizedEnvironment<SHADOW_FLOATING>>(m, RSG_MAKE_STR(SHADOW_FLOATING_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<SHADOW_FLOATING>::init)
    .def("reset", &VectorizedEnvironment<SHADOW_FLOATING>::reset)
    .def("add_stage", &VectorizedEnvironment<SHADOW_FLOATING>::add_stage)
    .def("load_object", &VectorizedEnvironment<SHADOW_FLOATING>::load_object)
    .def("load_articulated", &VectorizedEnvironment<SHADOW_FLOATING>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<SHADOW_FLOATING>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<SHADOW_FLOATING>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<SHADOW_FLOATING>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<SHADOW_FLOATING>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<SHADOW_FLOATING>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<SHADOW_FLOATING>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<SHADOW_FLOATING>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<SHADOW_FLOATING>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<SHADOW_FLOATING>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<SHADOW_FLOATING>::observe)
    .def("get_global_state", &VectorizedEnvironment<SHADOW_FLOATING>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<SHADOW_FLOATING>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<SHADOW_FLOATING>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<SHADOW_FLOATING>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<SHADOW_FLOATING>::control_switch_all)
    .def("step", &VectorizedEnvironment<SHADOW_FLOATING>::step)
    .def("step2", &VectorizedEnvironment<SHADOW_FLOATING>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<SHADOW_FLOATING>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<SHADOW_FLOATING>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<SHADOW_FLOATING>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<SHADOW_FLOATING>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<SHADOW_FLOATING>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<SHADOW_FLOATING>::close)
    .def("isTerminalState", &VectorizedEnvironment<SHADOW_FLOATING>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<SHADOW_FLOATING>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<SHADOW_FLOATING>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<SHADOW_FLOATING>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<SHADOW_FLOATING>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<SHADOW_FLOATING>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<SHADOW_FLOATING>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<SHADOW_FLOATING>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<SHADOW_FLOATING>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<SHADOW_FLOATING>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<SHADOW_FLOATING>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<SHADOW_FLOATING>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<SHADOW_FLOATING>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<SHADOW_FLOATING>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<SHADOW_FLOATING>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<SHADOW_FLOATING>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<SHADOW_FLOATING> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<SHADOW_FLOATING> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
    ));

 

  py::class_<VectorizedEnvironment<MANO_FLOATING>>(m, RSG_MAKE_STR(MANO_FLOATING_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<MANO_FLOATING>::init)
    .def("reset", &VectorizedEnvironment<MANO_FLOATING>::reset)
    .def("add_stage", &VectorizedEnvironment<MANO_FLOATING>::add_stage)
    .def("load_object", &VectorizedEnvironment<MANO_FLOATING>::load_object)
    .def("load_articulated", &VectorizedEnvironment<MANO_FLOATING>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<MANO_FLOATING>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<MANO_FLOATING>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<MANO_FLOATING>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<MANO_FLOATING>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<MANO_FLOATING>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<MANO_FLOATING>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<MANO_FLOATING>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<MANO_FLOATING>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<MANO_FLOATING>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<MANO_FLOATING>::observe)
    .def("get_global_state", &VectorizedEnvironment<MANO_FLOATING>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<MANO_FLOATING>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<MANO_FLOATING>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<MANO_FLOATING>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<MANO_FLOATING>::control_switch_all)
    .def("step", &VectorizedEnvironment<MANO_FLOATING>::step)
    .def("step2", &VectorizedEnvironment<MANO_FLOATING>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<MANO_FLOATING>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<MANO_FLOATING>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<MANO_FLOATING>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<MANO_FLOATING>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<MANO_FLOATING>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<MANO_FLOATING>::close)
    .def("isTerminalState", &VectorizedEnvironment<MANO_FLOATING>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<MANO_FLOATING>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<MANO_FLOATING>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<MANO_FLOATING>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<MANO_FLOATING>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<MANO_FLOATING>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<MANO_FLOATING>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<MANO_FLOATING>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<MANO_FLOATING>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<MANO_FLOATING>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<MANO_FLOATING>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<MANO_FLOATING>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<MANO_FLOATING>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<MANO_FLOATING>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<MANO_FLOATING>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<MANO_FLOATING>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<MANO_FLOATING> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<MANO_FLOATING> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
    ));

 
    //for shadow
  py::class_<VectorizedEnvironment<ALLEGRO_FLOATING>>(m, RSG_MAKE_STR(ALLEGRO_FLOATING_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<ALLEGRO_FLOATING>::init)
    .def("reset", &VectorizedEnvironment<ALLEGRO_FLOATING>::reset)
    .def("add_stage", &VectorizedEnvironment<ALLEGRO_FLOATING>::add_stage)
    .def("load_object", &VectorizedEnvironment<ALLEGRO_FLOATING>::load_object)
    .def("load_articulated", &VectorizedEnvironment<ALLEGRO_FLOATING>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<ALLEGRO_FLOATING>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<ALLEGRO_FLOATING>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<ALLEGRO_FLOATING>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<ALLEGRO_FLOATING>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<ALLEGRO_FLOATING>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<ALLEGRO_FLOATING>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<ALLEGRO_FLOATING>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<ALLEGRO_FLOATING>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<ALLEGRO_FLOATING>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<ALLEGRO_FLOATING>::observe)
    .def("get_global_state", &VectorizedEnvironment<ALLEGRO_FLOATING>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<ALLEGRO_FLOATING>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<ALLEGRO_FLOATING>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<ALLEGRO_FLOATING>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<ALLEGRO_FLOATING>::control_switch_all)
    .def("step", &VectorizedEnvironment<ALLEGRO_FLOATING>::step)
    .def("step2", &VectorizedEnvironment<ALLEGRO_FLOATING>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<ALLEGRO_FLOATING>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<ALLEGRO_FLOATING>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<ALLEGRO_FLOATING>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<ALLEGRO_FLOATING>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<ALLEGRO_FLOATING>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<ALLEGRO_FLOATING>::close)
    .def("isTerminalState", &VectorizedEnvironment<ALLEGRO_FLOATING>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ALLEGRO_FLOATING>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ALLEGRO_FLOATING>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<ALLEGRO_FLOATING>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<ALLEGRO_FLOATING>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<ALLEGRO_FLOATING>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<ALLEGRO_FLOATING>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ALLEGRO_FLOATING>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<ALLEGRO_FLOATING>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<ALLEGRO_FLOATING>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<ALLEGRO_FLOATING>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<ALLEGRO_FLOATING>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<ALLEGRO_FLOATING>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<ALLEGRO_FLOATING>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<ALLEGRO_FLOATING>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<ALLEGRO_FLOATING>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<ALLEGRO_FLOATING> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<ALLEGRO_FLOATING> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
  ));
  py::class_<VectorizedEnvironment<LEAP_FLOATING>>(m, RSG_MAKE_STR(LEAP_FLOATING_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<LEAP_FLOATING>::init)
    .def("reset", &VectorizedEnvironment<LEAP_FLOATING>::reset)
    .def("add_stage", &VectorizedEnvironment<LEAP_FLOATING>::add_stage)
    .def("load_object", &VectorizedEnvironment<LEAP_FLOATING>::load_object)
    .def("load_articulated", &VectorizedEnvironment<LEAP_FLOATING>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<LEAP_FLOATING>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<LEAP_FLOATING>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<LEAP_FLOATING>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<LEAP_FLOATING>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<LEAP_FLOATING>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<LEAP_FLOATING>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<LEAP_FLOATING>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<LEAP_FLOATING>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<LEAP_FLOATING>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<LEAP_FLOATING>::observe)
    .def("get_global_state", &VectorizedEnvironment<LEAP_FLOATING>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<LEAP_FLOATING>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<LEAP_FLOATING>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<LEAP_FLOATING>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<LEAP_FLOATING>::control_switch_all)
    .def("step", &VectorizedEnvironment<LEAP_FLOATING>::step)
    .def("step2", &VectorizedEnvironment<LEAP_FLOATING>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<LEAP_FLOATING>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<LEAP_FLOATING>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<LEAP_FLOATING>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<LEAP_FLOATING>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<LEAP_FLOATING>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<LEAP_FLOATING>::close)
    .def("isTerminalState", &VectorizedEnvironment<LEAP_FLOATING>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<LEAP_FLOATING>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<LEAP_FLOATING>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<LEAP_FLOATING>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<LEAP_FLOATING>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<LEAP_FLOATING>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<LEAP_FLOATING>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<LEAP_FLOATING>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<LEAP_FLOATING>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<LEAP_FLOATING>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<LEAP_FLOATING>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<LEAP_FLOATING>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<LEAP_FLOATING>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<LEAP_FLOATING>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<LEAP_FLOATING>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<LEAP_FLOATING>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<LEAP_FLOATING> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<LEAP_FLOATING> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
  ));
  py::class_<VectorizedEnvironment<LEAP_TEST>>(m, RSG_MAKE_STR(LEAP_TEST_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<LEAP_TEST>::init)
    .def("reset", &VectorizedEnvironment<LEAP_TEST>::reset)
    .def("add_stage", &VectorizedEnvironment<LEAP_TEST>::add_stage)
    .def("load_object", &VectorizedEnvironment<LEAP_TEST>::load_object)
    .def("load_articulated", &VectorizedEnvironment<LEAP_TEST>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<LEAP_TEST>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<LEAP_TEST>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<LEAP_TEST>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<LEAP_TEST>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<LEAP_TEST>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<LEAP_TEST>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<LEAP_TEST>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<LEAP_TEST>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<LEAP_TEST>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<LEAP_TEST>::observe)
    .def("get_global_state", &VectorizedEnvironment<LEAP_TEST>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<LEAP_TEST>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<LEAP_TEST>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<LEAP_TEST>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<LEAP_TEST>::control_switch_all)
    .def("step", &VectorizedEnvironment<LEAP_TEST>::step)
    .def("step2", &VectorizedEnvironment<LEAP_TEST>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<LEAP_TEST>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<LEAP_TEST>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<LEAP_TEST>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<LEAP_TEST>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<LEAP_TEST>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<LEAP_TEST>::close)
    .def("isTerminalState", &VectorizedEnvironment<LEAP_TEST>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<LEAP_TEST>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<LEAP_TEST>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<LEAP_TEST>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<LEAP_TEST>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<LEAP_TEST>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<LEAP_TEST>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<LEAP_TEST>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<LEAP_TEST>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<LEAP_TEST>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<LEAP_TEST>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<LEAP_TEST>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<LEAP_TEST>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<LEAP_TEST>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<LEAP_TEST>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<LEAP_TEST>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<LEAP_TEST> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<LEAP_TEST> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
    }
  ));
  py::class_<VectorizedEnvironment<ALLEGRO_TEST>>(m, RSG_MAKE_STR(ALLEGRO_TEST_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<ALLEGRO_TEST>::init)
    .def("reset", &VectorizedEnvironment<ALLEGRO_TEST>::reset)
    .def("add_stage", &VectorizedEnvironment<ALLEGRO_TEST>::add_stage)
    .def("load_object", &VectorizedEnvironment<ALLEGRO_TEST>::load_object)
    .def("load_articulated", &VectorizedEnvironment<ALLEGRO_TEST>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<ALLEGRO_TEST>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<ALLEGRO_TEST>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<ALLEGRO_TEST>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<ALLEGRO_TEST>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<ALLEGRO_TEST>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<ALLEGRO_TEST>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<ALLEGRO_TEST>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<ALLEGRO_TEST>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<ALLEGRO_TEST>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<ALLEGRO_TEST>::observe)
    .def("get_global_state", &VectorizedEnvironment<ALLEGRO_TEST>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<ALLEGRO_TEST>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<ALLEGRO_TEST>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<ALLEGRO_TEST>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<ALLEGRO_TEST>::control_switch_all)
    .def("step", &VectorizedEnvironment<ALLEGRO_TEST>::step)
    .def("step2", &VectorizedEnvironment<ALLEGRO_TEST>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<ALLEGRO_TEST>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<ALLEGRO_TEST>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<ALLEGRO_TEST>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<ALLEGRO_TEST>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<ALLEGRO_TEST>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<ALLEGRO_TEST>::close)
    .def("isTerminalState", &VectorizedEnvironment<ALLEGRO_TEST>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ALLEGRO_TEST>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ALLEGRO_TEST>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<ALLEGRO_TEST>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<ALLEGRO_TEST>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<ALLEGRO_TEST>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<ALLEGRO_TEST>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ALLEGRO_TEST>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<ALLEGRO_TEST>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<ALLEGRO_TEST>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<ALLEGRO_TEST>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<ALLEGRO_TEST>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<ALLEGRO_TEST>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<ALLEGRO_TEST>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<ALLEGRO_TEST>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<ALLEGRO_TEST>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<ALLEGRO_TEST> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<ALLEGRO_TEST> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
    }
  ));
  py::class_<VectorizedEnvironment<MANO_TEST>>(m, RSG_MAKE_STR(MANO_TEST_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<MANO_TEST>::init)
    .def("reset", &VectorizedEnvironment<MANO_TEST>::reset)
    .def("add_stage", &VectorizedEnvironment<MANO_TEST>::add_stage)
    .def("load_object", &VectorizedEnvironment<MANO_TEST>::load_object)
    .def("load_articulated", &VectorizedEnvironment<MANO_TEST>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<MANO_TEST>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<MANO_TEST>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<MANO_TEST>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<MANO_TEST>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<MANO_TEST>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<MANO_TEST>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<MANO_TEST>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<MANO_TEST>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<MANO_TEST>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<MANO_TEST>::observe)
    .def("get_global_state", &VectorizedEnvironment<MANO_TEST>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<MANO_TEST>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<MANO_TEST>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<MANO_TEST>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<MANO_TEST>::control_switch_all)
    .def("step", &VectorizedEnvironment<MANO_TEST>::step)
    .def("step2", &VectorizedEnvironment<MANO_TEST>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<MANO_TEST>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<MANO_TEST>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<MANO_TEST>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<MANO_TEST>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<MANO_TEST>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<MANO_TEST>::close)
    .def("isTerminalState", &VectorizedEnvironment<MANO_TEST>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<MANO_TEST>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<MANO_TEST>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<MANO_TEST>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<MANO_TEST>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<MANO_TEST>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<MANO_TEST>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<MANO_TEST>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<MANO_TEST>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<MANO_TEST>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<MANO_TEST>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<MANO_TEST>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<MANO_TEST>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<MANO_TEST>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<MANO_TEST>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<MANO_TEST>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<MANO_TEST> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<MANO_TEST> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
    }
  ));
  py::class_<VectorizedEnvironment<SHADOW_TEST>>(m, RSG_MAKE_STR(SHADOW_TEST_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<SHADOW_TEST>::init)
    .def("reset", &VectorizedEnvironment<SHADOW_TEST>::reset)
    .def("add_stage", &VectorizedEnvironment<SHADOW_TEST>::add_stage)
    .def("load_object", &VectorizedEnvironment<SHADOW_TEST>::load_object)
    .def("load_articulated", &VectorizedEnvironment<SHADOW_TEST>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<SHADOW_TEST>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<SHADOW_TEST>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<SHADOW_TEST>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<SHADOW_TEST>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<SHADOW_TEST>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<SHADOW_TEST>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<SHADOW_TEST>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<SHADOW_TEST>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<SHADOW_TEST>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<SHADOW_TEST>::observe)
    .def("get_global_state", &VectorizedEnvironment<SHADOW_TEST>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<SHADOW_TEST>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<SHADOW_TEST>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<SHADOW_TEST>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<SHADOW_TEST>::control_switch_all)
    .def("step", &VectorizedEnvironment<SHADOW_TEST>::step)
    .def("step2", &VectorizedEnvironment<SHADOW_TEST>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<SHADOW_TEST>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<SHADOW_TEST>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<SHADOW_TEST>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<SHADOW_TEST>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<SHADOW_TEST>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<SHADOW_TEST>::close)
    .def("isTerminalState", &VectorizedEnvironment<SHADOW_TEST>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<SHADOW_TEST>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<SHADOW_TEST>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<SHADOW_TEST>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<SHADOW_TEST>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<SHADOW_TEST>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<SHADOW_TEST>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<SHADOW_TEST>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<SHADOW_TEST>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<SHADOW_TEST>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<SHADOW_TEST>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<SHADOW_TEST>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<SHADOW_TEST>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<SHADOW_TEST>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<SHADOW_TEST>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<SHADOW_TEST>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<SHADOW_TEST> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<SHADOW_TEST> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
    }
  ));



  py::class_<VectorizedEnvironment<INSPIRE_FLOATING>>(m, RSG_MAKE_STR(INSPIRE_FLOATING_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<INSPIRE_FLOATING>::init)
    .def("reset", &VectorizedEnvironment<INSPIRE_FLOATING>::reset)
    .def("add_stage", &VectorizedEnvironment<INSPIRE_FLOATING>::add_stage)
    .def("load_object", &VectorizedEnvironment<INSPIRE_FLOATING>::load_object)
    .def("load_articulated", &VectorizedEnvironment<INSPIRE_FLOATING>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<INSPIRE_FLOATING>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<INSPIRE_FLOATING>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<INSPIRE_FLOATING>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<INSPIRE_FLOATING>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<INSPIRE_FLOATING>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<INSPIRE_FLOATING>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<INSPIRE_FLOATING>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<INSPIRE_FLOATING>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<INSPIRE_FLOATING>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<INSPIRE_FLOATING>::observe)
    .def("get_global_state", &VectorizedEnvironment<INSPIRE_FLOATING>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<INSPIRE_FLOATING>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<INSPIRE_FLOATING>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<INSPIRE_FLOATING>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<INSPIRE_FLOATING>::control_switch_all)
    .def("step", &VectorizedEnvironment<INSPIRE_FLOATING>::step)
    .def("step2", &VectorizedEnvironment<INSPIRE_FLOATING>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<INSPIRE_FLOATING>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<INSPIRE_FLOATING>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<INSPIRE_FLOATING>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<INSPIRE_FLOATING>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<INSPIRE_FLOATING>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<INSPIRE_FLOATING>::close)
    .def("isTerminalState", &VectorizedEnvironment<INSPIRE_FLOATING>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<INSPIRE_FLOATING>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<INSPIRE_FLOATING>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<INSPIRE_FLOATING>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<INSPIRE_FLOATING>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<INSPIRE_FLOATING>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<INSPIRE_FLOATING>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<INSPIRE_FLOATING>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<INSPIRE_FLOATING>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<INSPIRE_FLOATING>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<INSPIRE_FLOATING>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<INSPIRE_FLOATING>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<INSPIRE_FLOATING>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<INSPIRE_FLOATING>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<INSPIRE_FLOATING>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<INSPIRE_FLOATING>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<INSPIRE_FLOATING> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<INSPIRE_FLOATING> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
    ));
  py::class_<VectorizedEnvironment<INSPIRE_TEST>>(m, RSG_MAKE_STR(INSPIRE_TEST_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<INSPIRE_TEST>::init)
    .def("reset", &VectorizedEnvironment<INSPIRE_TEST>::reset)
    .def("add_stage", &VectorizedEnvironment<INSPIRE_TEST>::add_stage)
    .def("load_object", &VectorizedEnvironment<INSPIRE_TEST>::load_object)
    .def("load_articulated", &VectorizedEnvironment<INSPIRE_TEST>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<INSPIRE_TEST>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<INSPIRE_TEST>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<INSPIRE_TEST>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<INSPIRE_TEST>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<INSPIRE_TEST>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<INSPIRE_TEST>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<INSPIRE_TEST>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<INSPIRE_TEST>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<INSPIRE_TEST>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<INSPIRE_TEST>::observe)
    .def("get_global_state", &VectorizedEnvironment<INSPIRE_TEST>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<INSPIRE_TEST>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<INSPIRE_TEST>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<INSPIRE_TEST>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<INSPIRE_TEST>::control_switch_all)
    .def("step", &VectorizedEnvironment<INSPIRE_TEST>::step)
    .def("step2", &VectorizedEnvironment<INSPIRE_TEST>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<INSPIRE_TEST>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<INSPIRE_TEST>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<INSPIRE_TEST>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<INSPIRE_TEST>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<INSPIRE_TEST>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<INSPIRE_TEST>::close)
    .def("isTerminalState", &VectorizedEnvironment<INSPIRE_TEST>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<INSPIRE_TEST>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<INSPIRE_TEST>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<INSPIRE_TEST>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<INSPIRE_TEST>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<INSPIRE_TEST>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<INSPIRE_TEST>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<INSPIRE_TEST>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<INSPIRE_TEST>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<INSPIRE_TEST>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<INSPIRE_TEST>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<INSPIRE_TEST>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<INSPIRE_TEST>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<INSPIRE_TEST>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<INSPIRE_TEST>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<INSPIRE_TEST>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<INSPIRE_TEST> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<INSPIRE_TEST> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
    }
    ));

  py::class_<VectorizedEnvironment<ABILITY_FLOATING>>(m, RSG_MAKE_STR(ABILITY_FLOATING_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<ABILITY_FLOATING>::init)
    .def("reset", &VectorizedEnvironment<ABILITY_FLOATING>::reset)
    .def("add_stage", &VectorizedEnvironment<ABILITY_FLOATING>::add_stage)
    .def("load_object", &VectorizedEnvironment<ABILITY_FLOATING>::load_object)
    .def("load_articulated", &VectorizedEnvironment<ABILITY_FLOATING>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<ABILITY_FLOATING>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<ABILITY_FLOATING>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<ABILITY_FLOATING>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<ABILITY_FLOATING>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<ABILITY_FLOATING>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<ABILITY_FLOATING>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<ABILITY_FLOATING>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<ABILITY_FLOATING>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<ABILITY_FLOATING>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<ABILITY_FLOATING>::observe)
    .def("get_global_state", &VectorizedEnvironment<ABILITY_FLOATING>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<ABILITY_FLOATING>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<ABILITY_FLOATING>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<ABILITY_FLOATING>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<ABILITY_FLOATING>::control_switch_all)
    .def("step", &VectorizedEnvironment<ABILITY_FLOATING>::step)
    .def("step2", &VectorizedEnvironment<ABILITY_FLOATING>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<ABILITY_FLOATING>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<ABILITY_FLOATING>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<ABILITY_FLOATING>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<ABILITY_FLOATING>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<ABILITY_FLOATING>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<ABILITY_FLOATING>::close)
    .def("isTerminalState", &VectorizedEnvironment<ABILITY_FLOATING>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ABILITY_FLOATING>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ABILITY_FLOATING>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<ABILITY_FLOATING>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<ABILITY_FLOATING>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<ABILITY_FLOATING>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<ABILITY_FLOATING>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ABILITY_FLOATING>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<ABILITY_FLOATING>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<ABILITY_FLOATING>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<ABILITY_FLOATING>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<ABILITY_FLOATING>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<ABILITY_FLOATING>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<ABILITY_FLOATING>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<ABILITY_FLOATING>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<ABILITY_FLOATING>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<ABILITY_FLOATING> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }
    
            /* Create a new C++ instance */
            VectorizedEnvironment<ABILITY_FLOATING> p(t[0].cast<std::string>(), t[1].cast<std::string>());
    
            return p;
        }
    ));
  py::class_<VectorizedEnvironment<ABILITY_TEST>>(m, RSG_MAKE_STR(ABILITY_TEST_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<ABILITY_TEST>::init)
    .def("reset", &VectorizedEnvironment<ABILITY_TEST>::reset)
    .def("add_stage", &VectorizedEnvironment<ABILITY_TEST>::add_stage)
    .def("load_object", &VectorizedEnvironment<ABILITY_TEST>::load_object)
    .def("load_articulated", &VectorizedEnvironment<ABILITY_TEST>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<ABILITY_TEST>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<ABILITY_TEST>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<ABILITY_TEST>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<ABILITY_TEST>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<ABILITY_TEST>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<ABILITY_TEST>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<ABILITY_TEST>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<ABILITY_TEST>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<ABILITY_TEST>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<ABILITY_TEST>::observe)
    .def("get_global_state", &VectorizedEnvironment<ABILITY_TEST>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<ABILITY_TEST>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<ABILITY_TEST>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<ABILITY_TEST>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<ABILITY_TEST>::control_switch_all)
    .def("step", &VectorizedEnvironment<ABILITY_TEST>::step)
    .def("step2", &VectorizedEnvironment<ABILITY_TEST>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<ABILITY_TEST>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<ABILITY_TEST>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<ABILITY_TEST>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<ABILITY_TEST>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<ABILITY_TEST>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<ABILITY_TEST>::close)
    .def("isTerminalState", &VectorizedEnvironment<ABILITY_TEST>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ABILITY_TEST>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ABILITY_TEST>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<ABILITY_TEST>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<ABILITY_TEST>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<ABILITY_TEST>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<ABILITY_TEST>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ABILITY_TEST>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<ABILITY_TEST>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<ABILITY_TEST>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<ABILITY_TEST>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<ABILITY_TEST>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<ABILITY_TEST>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<ABILITY_TEST>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<ABILITY_TEST>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<ABILITY_TEST>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<ABILITY_TEST> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }
    
            /* Create a new C++ instance */
            VectorizedEnvironment<ABILITY_TEST> p(t[0].cast<std::string>(), t[1].cast<std::string>());
    
            return p;
    }
    ));

  py::class_<VectorizedEnvironment<SCHUNK_FLOATING>>(m, RSG_MAKE_STR(SCHUNK_FLOATING_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<SCHUNK_FLOATING>::init)
    .def("reset", &VectorizedEnvironment<SCHUNK_FLOATING>::reset)
    .def("add_stage", &VectorizedEnvironment<SCHUNK_FLOATING>::add_stage)
    .def("load_object", &VectorizedEnvironment<SCHUNK_FLOATING>::load_object)
    .def("load_articulated", &VectorizedEnvironment<SCHUNK_FLOATING>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<SCHUNK_FLOATING>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<SCHUNK_FLOATING>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<SCHUNK_FLOATING>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<SCHUNK_FLOATING>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<SCHUNK_FLOATING>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<SCHUNK_FLOATING>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<SCHUNK_FLOATING>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<SCHUNK_FLOATING>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<SCHUNK_FLOATING>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<SCHUNK_FLOATING>::observe)
    .def("get_global_state", &VectorizedEnvironment<SCHUNK_FLOATING>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<SCHUNK_FLOATING>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<SCHUNK_FLOATING>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<SCHUNK_FLOATING>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<SCHUNK_FLOATING>::control_switch_all)
    .def("step", &VectorizedEnvironment<SCHUNK_FLOATING>::step)
    .def("step2", &VectorizedEnvironment<SCHUNK_FLOATING>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<SCHUNK_FLOATING>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<SCHUNK_FLOATING>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<SCHUNK_FLOATING>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<SCHUNK_FLOATING>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<SCHUNK_FLOATING>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<SCHUNK_FLOATING>::close)
    .def("isTerminalState", &VectorizedEnvironment<SCHUNK_FLOATING>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<SCHUNK_FLOATING>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<SCHUNK_FLOATING>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<SCHUNK_FLOATING>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<SCHUNK_FLOATING>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<SCHUNK_FLOATING>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<SCHUNK_FLOATING>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<SCHUNK_FLOATING>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<SCHUNK_FLOATING>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<SCHUNK_FLOATING>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<SCHUNK_FLOATING>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<SCHUNK_FLOATING>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<SCHUNK_FLOATING>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<SCHUNK_FLOATING>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<SCHUNK_FLOATING>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<SCHUNK_FLOATING>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<SCHUNK_FLOATING> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<SCHUNK_FLOATING> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
        }
    ));
  py::class_<VectorizedEnvironment<SCHUNK_TEST>>(m, RSG_MAKE_STR(SCHUNK_TEST_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<SCHUNK_TEST>::init)
    .def("reset", &VectorizedEnvironment<SCHUNK_TEST>::reset)
    .def("add_stage", &VectorizedEnvironment<SCHUNK_TEST>::add_stage)
    .def("load_object", &VectorizedEnvironment<SCHUNK_TEST>::load_object)
    .def("load_articulated", &VectorizedEnvironment<SCHUNK_TEST>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<SCHUNK_TEST>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<SCHUNK_TEST>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<SCHUNK_TEST>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<SCHUNK_TEST>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<SCHUNK_TEST>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<SCHUNK_TEST>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<SCHUNK_TEST>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<SCHUNK_TEST>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<SCHUNK_TEST>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<SCHUNK_TEST>::observe)
    .def("get_global_state", &VectorizedEnvironment<SCHUNK_TEST>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<SCHUNK_TEST>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<SCHUNK_TEST>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<SCHUNK_TEST>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<SCHUNK_TEST>::control_switch_all)
    .def("step", &VectorizedEnvironment<SCHUNK_TEST>::step)
    .def("step2", &VectorizedEnvironment<SCHUNK_TEST>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<SCHUNK_TEST>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<SCHUNK_TEST>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<SCHUNK_TEST>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<SCHUNK_TEST>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<SCHUNK_TEST>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<SCHUNK_TEST>::close)
    .def("isTerminalState", &VectorizedEnvironment<SCHUNK_TEST>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<SCHUNK_TEST>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<SCHUNK_TEST>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<SCHUNK_TEST>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<SCHUNK_TEST>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<SCHUNK_TEST>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<SCHUNK_TEST>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<SCHUNK_TEST>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<SCHUNK_TEST>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<SCHUNK_TEST>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<SCHUNK_TEST>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<SCHUNK_TEST>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<SCHUNK_TEST>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<SCHUNK_TEST>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<SCHUNK_TEST>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<SCHUNK_TEST>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<SCHUNK_TEST> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<SCHUNK_TEST> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
    }
    ));
      py::class_<VectorizedEnvironment<BRAINCO_FLOATING>>(m, RSG_MAKE_STR(BRAINCO_FLOATING_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<BRAINCO_FLOATING>::init)
    .def("reset", &VectorizedEnvironment<BRAINCO_FLOATING>::reset)
    .def("add_stage", &VectorizedEnvironment<BRAINCO_FLOATING>::add_stage)
    .def("load_object", &VectorizedEnvironment<BRAINCO_FLOATING>::load_object)
    .def("load_articulated", &VectorizedEnvironment<BRAINCO_FLOATING>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<BRAINCO_FLOATING>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<BRAINCO_FLOATING>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<BRAINCO_FLOATING>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<BRAINCO_FLOATING>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<BRAINCO_FLOATING>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<BRAINCO_FLOATING>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<BRAINCO_FLOATING>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<BRAINCO_FLOATING>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<BRAINCO_FLOATING>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<BRAINCO_FLOATING>::observe)
    .def("get_global_state", &VectorizedEnvironment<BRAINCO_FLOATING>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<BRAINCO_FLOATING>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<BRAINCO_FLOATING>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<BRAINCO_FLOATING>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<BRAINCO_FLOATING>::control_switch_all)
    .def("step", &VectorizedEnvironment<BRAINCO_FLOATING>::step)
    .def("step2", &VectorizedEnvironment<BRAINCO_FLOATING>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<BRAINCO_FLOATING>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<BRAINCO_FLOATING>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<BRAINCO_FLOATING>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<BRAINCO_FLOATING>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<BRAINCO_FLOATING>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<BRAINCO_FLOATING>::close)
    .def("isTerminalState", &VectorizedEnvironment<BRAINCO_FLOATING>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<BRAINCO_FLOATING>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<BRAINCO_FLOATING>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<BRAINCO_FLOATING>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<BRAINCO_FLOATING>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<BRAINCO_FLOATING>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<BRAINCO_FLOATING>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<BRAINCO_FLOATING>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<BRAINCO_FLOATING>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<BRAINCO_FLOATING>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<BRAINCO_FLOATING>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<BRAINCO_FLOATING>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<BRAINCO_FLOATING>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<BRAINCO_FLOATING>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<BRAINCO_FLOATING>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<BRAINCO_FLOATING>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<BRAINCO_FLOATING> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }
    
            /* Create a new C++ instance */
            VectorizedEnvironment<BRAINCO_FLOATING> p(t[0].cast<std::string>(), t[1].cast<std::string>());
    
            return p;
        }
    ));
  py::class_<VectorizedEnvironment<BRAINCO_TEST>>(m, RSG_MAKE_STR(BRAINCO_TEST_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<BRAINCO_TEST>::init)
    .def("reset", &VectorizedEnvironment<BRAINCO_TEST>::reset)
    .def("add_stage", &VectorizedEnvironment<BRAINCO_TEST>::add_stage)
    .def("load_object", &VectorizedEnvironment<BRAINCO_TEST>::load_object)
    .def("load_articulated", &VectorizedEnvironment<BRAINCO_TEST>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<BRAINCO_TEST>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<BRAINCO_TEST>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<BRAINCO_TEST>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<BRAINCO_TEST>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<BRAINCO_TEST>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<BRAINCO_TEST>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<BRAINCO_TEST>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<BRAINCO_TEST>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<BRAINCO_TEST>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<BRAINCO_TEST>::observe)
    .def("get_global_state", &VectorizedEnvironment<BRAINCO_TEST>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<BRAINCO_TEST>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<BRAINCO_TEST>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<BRAINCO_TEST>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<BRAINCO_TEST>::control_switch_all)
    .def("step", &VectorizedEnvironment<BRAINCO_TEST>::step)
    .def("step2", &VectorizedEnvironment<BRAINCO_TEST>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<BRAINCO_TEST>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<BRAINCO_TEST>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<BRAINCO_TEST>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<BRAINCO_TEST>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<BRAINCO_TEST>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<BRAINCO_TEST>::close)
    .def("isTerminalState", &VectorizedEnvironment<BRAINCO_TEST>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<BRAINCO_TEST>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<BRAINCO_TEST>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<BRAINCO_TEST>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<BRAINCO_TEST>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<BRAINCO_TEST>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<BRAINCO_TEST>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<BRAINCO_TEST>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<BRAINCO_TEST>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<BRAINCO_TEST>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<BRAINCO_TEST>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<BRAINCO_TEST>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<BRAINCO_TEST>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<BRAINCO_TEST>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<BRAINCO_TEST>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<BRAINCO_TEST>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<BRAINCO_TEST> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }
    
            /* Create a new C++ instance */
            VectorizedEnvironment<BRAINCO_TEST> p(t[0].cast<std::string>(), t[1].cast<std::string>());
    
            return p;
    }
    ));


  py::class_<VectorizedEnvironment<BARRETT_TEST>>(m, RSG_MAKE_STR(BARRETT_TEST_NAME))
    .def(py::init<std::string, std::string>(), py::arg("resourceDir"), py::arg("cfg"))
    .def("init", &VectorizedEnvironment<BARRETT_TEST>::init)
    .def("reset", &VectorizedEnvironment<BARRETT_TEST>::reset)
    .def("add_stage", &VectorizedEnvironment<BARRETT_TEST>::add_stage)
    .def("load_object", &VectorizedEnvironment<BARRETT_TEST>::load_object)
    .def("load_articulated", &VectorizedEnvironment<BARRETT_TEST>::load_articulated)
    .def("load_multi_articulated", &VectorizedEnvironment<BARRETT_TEST>::load_multi_articulated)
    .def("reset_state", &VectorizedEnvironment<BARRETT_TEST>::reset_state)
    .def("set_goals_r", &VectorizedEnvironment<BARRETT_TEST>::set_goals_r)
    .def("set_goals_r2", &VectorizedEnvironment<BARRETT_TEST>::set_goals_r2)
    .def("set_obj_goal", &VectorizedEnvironment<BARRETT_TEST>::set_obj_goal)
    .def("set_goals", &VectorizedEnvironment<BARRETT_TEST>::set_goals)
    .def("set_imitation_goals", &VectorizedEnvironment<BARRETT_TEST>::set_imitation_goals)
    .def("set_ext", &VectorizedEnvironment<BARRETT_TEST>::set_ext)
    .def("set_pregrasp", &VectorizedEnvironment<BARRETT_TEST>::set_pregrasp)
    .def("observe", &VectorizedEnvironment<BARRETT_TEST>::observe)
    .def("get_global_state", &VectorizedEnvironment<BARRETT_TEST>::get_global_state)
    .def("set_rootguidance", &VectorizedEnvironment<BARRETT_TEST>::set_rootguidance)
    .def("switch_root_guidance", &VectorizedEnvironment<BARRETT_TEST>::switch_root_guidance)
    .def("control_switch", &VectorizedEnvironment<BARRETT_TEST>::control_switch)
    .def("control_switch_all", &VectorizedEnvironment<BARRETT_TEST>::control_switch_all)
    .def("step", &VectorizedEnvironment<BARRETT_TEST>::step)
    .def("step2", &VectorizedEnvironment<BARRETT_TEST>::step2)
    .def("reset_right_hand", &VectorizedEnvironment<BARRETT_TEST>::reset_right_hand)
    .def("step_imitate", &VectorizedEnvironment<BARRETT_TEST>::step_imitate)
    .def("setSeed", &VectorizedEnvironment<BARRETT_TEST>::setSeed)
    .def("rewardInfoRight", &VectorizedEnvironment<BARRETT_TEST>::getRewardInfoRight)
    .def("rewardInfoLeft", &VectorizedEnvironment<BARRETT_TEST>::getRewardInfoLeft)
    .def("close", &VectorizedEnvironment<BARRETT_TEST>::close)
    .def("isTerminalState", &VectorizedEnvironment<BARRETT_TEST>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<BARRETT_TEST>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<BARRETT_TEST>::setControlTimeStep)
    .def("getRightObDim", &VectorizedEnvironment<BARRETT_TEST>::getRightObDim)
    .def("getLeftObDim", &VectorizedEnvironment<BARRETT_TEST>::getLeftObDim)
    .def("getActionDim", &VectorizedEnvironment<BARRETT_TEST>::getActionDim)
    .def("getGSDim", &VectorizedEnvironment<BARRETT_TEST>::getGSDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<BARRETT_TEST>::getNumOfEnvs)
    .def("turnOnVisualization", &VectorizedEnvironment<BARRETT_TEST>::turnOnVisualization)
    .def("turnOffVisualization", &VectorizedEnvironment<BARRETT_TEST>::turnOffVisualization)
    .def("stopRecordingVideo", &VectorizedEnvironment<BARRETT_TEST>::stopRecordingVideo)
    .def("startRecordingVideo", &VectorizedEnvironment<BARRETT_TEST>::startRecordingVideo)
    .def("curriculumUpdate", &VectorizedEnvironment<BARRETT_TEST>::curriculumUpdate)
    .def("switch_arctic", &VectorizedEnvironment<BARRETT_TEST>::switch_arctic)
    .def("set_pd_wrist", &VectorizedEnvironment<BARRETT_TEST>::set_pd_wrist)
    .def("set_joint_sensor_visual", &VectorizedEnvironment<BARRETT_TEST>::set_joint_sensor_visual)
    .def(py::pickle(
        [](const VectorizedEnvironment<BARRETT_TEST> &p) { // __getstate__ --> Pickling to Python
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.getResourceDir(), p.getCfgString());
        },
        [](py::tuple t) { // __setstate__ - Pickling from Python
            if (t.size() != 2) {
              throw std::runtime_error("Invalid state!");
            }

            /* Create a new C++ instance */
            VectorizedEnvironment<BARRETT_TEST> p(t[0].cast<std::string>(), t[1].cast<std::string>());

            return p;
    }
    ));

  
  }
