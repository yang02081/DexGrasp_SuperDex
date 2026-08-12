//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include "raisim/World.hpp"
#include <vector>
#include "raisim/math.hpp"
#include <math.h>

namespace raisim {


    class SHADOW_FLOATING : public RaisimGymEnv {

        public:

            explicit SHADOW_FLOATING(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                visualizable_ = cfg["visualize"].As<bool>();
                root_guided = cfg["root_guided"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/shadow/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();


                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);

                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i = 0; i < num_contacts; i++){
                    const std::string& name = contact_bodies_r_[i];
                    if (name.find("distal") != std::string::npos) {
                        finger_weights_contact(i) *= 3.0;
                    }
                    if (name.rfind("th", 0) == 0) {  // thumb
                        finger_weights_contact(i) *= 2.0;
                    }
                }
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                finger_weights_aff.setOnes(num_bodyparts);
                const std::string tip_names[5] = {"FFtip_back", "MFtip_back", "RFtip_back", "LFtip_back", "THtip_back"};
                for(int t = 0; t < 5; t++){
                    for(int i = 0; i < num_bodyparts; i++){
                        if(body_parts_r_[i] == tip_names[t]){
                            finger_weights_aff[i] *= 4.0;
                            break;
                        }
                    }
                }
                finger_weights_aff /= finger_weights_aff.sum();
                finger_weights_aff *= num_bodyparts;

                for(int i = 0; i < num_contacts; i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                    if (contact_bodies_r_[i].rfind("th", 0) == 0) {
                        impulse_high[i] = 0.2;
                    }
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                /// MUST BE DONE FOR ALL ENVIRONMENTS
                /// Shadow: obDim = 12 + 2*gcDim + 4*num_contacts; gsDim = 10 + 6*num_bodyparts
                obDim_r_ = 136; //12 + 2 * gcDim_ + 4 * num_contacts; //132+4=136
                obDim_l_ = 1;
                gsDim_ = 142 ;//10 + 6 * num_bodyparts; 136+6=142
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                                   table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                                   table_top->setPosition(1.25, 0, 0.475);
                                   leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                                   leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                                   leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                                   leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                                   leg1->setPosition(0.2625,0.4675,0.2375);
                                   leg2->setPosition(2.2275,0.4875,0.2375);
                                   leg3->setPosition(0.2625,-0.4675,0.2375);
                                   leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();

                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);

                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);

                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                /// set initial hand pose (20 DoF) and velocity (20 DoF)
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                obj_weight = arctic->getTotalMass();
                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;
                        // Shadow has an extra LFJ5 node; use name-based indices when available (fallback to legacy indexing otherwise)
                        int mid_ids[4] = {-1, -1, -1, -1};
                        int thumb_tip_id = -1;
                        for(int i = 0; i < num_bodyparts; i++){
                            if(body_parts_r_[i] == "FFJ2") mid_ids[0] = i;
                            else if(body_parts_r_[i] == "MFJ2") mid_ids[1] = i;
                            else if(body_parts_r_[i] == "RFJ2") mid_ids[2] = i;
                            else if(body_parts_r_[i] == "LFJ2") mid_ids[3] = i;
                            else if(body_parts_r_[i] == "THtip_back") thumb_tip_id = i;
                        }
                        if(mid_ids[0] >= 0 && mid_ids[1] >= 0 && mid_ids[2] >= 0 && mid_ids[3] >= 0){
                            for(int k = 0; k < 4; k++){
                                mano_r_->getFramePosition(body_parts_r_[mid_ids[k]], sphere_pos);
                                hand_center_temp[0] += sphere_pos.e()[0];
                                hand_center_temp[1] += sphere_pos.e()[1];
                                hand_center_temp[2] += sphere_pos.e()[2];
                            }
                        } else {
                            for(int i = 0; i < 4 ; i++){
                                mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                                hand_center_temp[0] += sphere_pos.e()[0];
                                hand_center_temp[1] += sphere_pos.e()[1];
                                hand_center_temp[2] += sphere_pos.e()[2];
                            }
                        }
                        hand_center_temp /= 4;
                        mano_r_->getFramePosition(body_parts_r_[(thumb_tip_id >= 0) ? thumb_tip_id : 20], sphere_pos);


                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                //            Eigen::Vector3d target_center;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }

                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }


                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();


                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);


                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame

                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);

                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),      // (mirror) , generalized coordinate
                            right_hand_torque,
                            wrist_vel_in_wrist,
                            wrist_qvel_in_wrist,
                            contacts_r_af,
                            impulses_r_af,
                            obj_vel_in_wrist,
                            obj_qvel_in_wrist,
                            contacts_r_non_af,
                            impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
                Eigen::Vector3d current_across_axis_w;

                    current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

                Eigen::Vector3d current_grasp_axis_w;
                current_grasp_axis_w = wrist_mat_r.e() * hand_center;
                current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

                Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
                current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
                current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

                float across_mul = 0;
                across_mul = current_across_axis_o.dot(across_axis_o);

                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
                raisim::Vec<3> obj_current_pos;
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(top_id, obj_current_pos); //Obj_Position: object base position at this time

                double displacement = (obj_current_pos.e() - obj_pos_init_.head(3)).squaredNorm();
                if (displacement > 0.1){
                    terminalReward = -10;
                    return true;
                }

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 17;
            int num_bodyparts = 22;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;

            std::string body_parts_r_[22] =  {"z_rotation_joint",    // 0
                "FFJ3", "FFJ2", "FFJ1","FFtip_back", // 0.100568，0.045，0.025，0.026
                "MFJ3", "MFJ2", "MFJ1","MFtip_back", // 0.0996092，0.045，0.025，0.026
                "RFJ3", "RFJ2", "RFJ1","RFtip_back", // 0.0956347，0.045，0.025，0.026
                "LFJ5","LFJ3", "LFJ2", "LFJ1","LFtip_back", // 0.0925597，0.045，0.025，0.026
                "THJ4", "THJ2", "THJ1","THtip_back"  // 0.045489，0.038，0.032，0.0275
                
            };

            std::string contact_bodies_r_[17] =  {"palm",  // base_link
                                "ffproximal", "ffmiddle", "ffdistal", // link_1.0 link_2.0   link_3.0_tip
                                "mfproximal", "mfmiddle", "mfdistal", // link_5.0 link_6.0   link_7.0_tip
                                "rfproximal", "rfmiddle", "rfdistal", // link_9.0 link_10.0   link_11.0_tip
                                "lfmetacarpal","lfproximal", "lfmiddle", "lfdistal",
                                "thproximal", "thmiddle", "thdistal"  // link_14.0 link_15.0   link_15.0_tip
            };

            Eigen::VectorXd mean_pose;

            raisim::PolyLine *lines[22];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[22];
            raisim::Visuals *sphere[22];
            raisim::Visuals *joints_sphere[22];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
        };

    class MANO_FLOATING : public RaisimGymEnv {

        public:

            explicit MANO_FLOATING(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                visualizable_ = cfg["visualize"].As<bool>();
                root_guided = cfg["root_guided"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "table", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/mano_double/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();

                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "table", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);

                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i=1; i < 6; i++){
                    finger_weights_contact(3*i) *= 3;
                }
                finger_weights_contact.segment(13,3) *= 2;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                finger_weights_aff.setOnes(num_bodyparts);
                for(int i=0; i < 5; i++){
                    finger_weights_aff[4 * i+4] *= 4.0;
                }
                finger_weights_aff /= finger_weights_aff.sum();
                finger_weights_aff *= num_bodyparts;

                for(int i=0; i< (num_contacts - 3); i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                for(int i=13; i< 16; i++){
                    impulse_high[i] = 0.2;
                    impulse_low[i] = -0.0;
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                /// MUST BE DONE FOR ALL ENVIRONMENTS
                obDim_r_ = 178;
                obDim_l_ = 1;
                gsDim_ = 136;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                    //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    //                table_top->setPosition(1.25, 0, 0.475);
                    //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg1->setPosition(0.2625,0.4675,0.2375);
                    //                leg2->setPosition(2.2275,0.4875,0.2375);
                    //                leg3->setPosition(0.2625,-0.4675,0.2375);
                    //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();

                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
                }
            }

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);

                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);

                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                /// set initial hand pose (20 DoF) and velocity (20 DoF)
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_ = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                obj_weight = arctic->getTotalMass();

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;
                        // Shadow has an extra LFJ5 node; use name-based indices when available (fallback to legacy indexing otherwise)
                        int mid_ids[4] = {-1, -1, -1, -1};
                        int thumb_tip_id = -1;
                        for(int i = 0; i < num_bodyparts; i++){
                            if(body_parts_r_[i] == "FFJ2") mid_ids[0] = i;
                            else if(body_parts_r_[i] == "MFJ2") mid_ids[1] = i;
                            else if(body_parts_r_[i] == "RFJ2") mid_ids[2] = i;
                            else if(body_parts_r_[i] == "LFJ2") mid_ids[3] = i;
                            else if(body_parts_r_[i] == "THtip_back") thumb_tip_id = i;
                        }
                        if(mid_ids[0] >= 0 && mid_ids[1] >= 0 && mid_ids[2] >= 0 && mid_ids[3] >= 0){
                            for(int k = 0; k < 4; k++){
                                mano_r_->getFramePosition(body_parts_r_[mid_ids[k]], sphere_pos);
                                hand_center_temp[0] += sphere_pos.e()[0];
                                hand_center_temp[1] += sphere_pos.e()[1];
                                hand_center_temp[2] += sphere_pos.e()[2];
                            }
                        } else {
                            for(int i = 0; i < 4 ; i++){
                                mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                                hand_center_temp[0] += sphere_pos.e()[0];
                                hand_center_temp[1] += sphere_pos.e()[1];
                                hand_center_temp[2] += sphere_pos.e()[2];
                            }
                        }
                        hand_center_temp /= 4;
                        mano_r_->getFramePosition(body_parts_r_[(thumb_tip_id >= 0) ? thumb_tip_id : 20], sphere_pos);
                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }

                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();


                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);

                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame


                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);
                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);

                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),      // (mirror) 45, generalized coordinate
                            right_hand_torque,
                            wrist_vel_in_wrist,
                            wrist_qvel_in_wrist,
                            contacts_r_af,
                            impulses_r_af,
                            obj_vel_in_wrist,
                            obj_qvel_in_wrist,
                            contacts_r_non_af,
                            impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
            Eigen::Vector3d current_across_axis_w;
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);

                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
                raisim::Vec<3> obj_current_pos;
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(top_id, obj_current_pos); //Obj_Position: object base position at this time

                double displacement = (obj_current_pos.e() - obj_pos_init_.head(3)).squaredNorm();
                if (displacement > 0.1 && !visualizable_){
                    terminalReward = -10;
                    return true;
                }

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 16;
            int num_bodyparts = 21;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;

            std::string body_parts_r_[21] = {"right_wrist_0rz",
                                            "right_index1_x", "right_index2_x", "right_index3_x", "right_index_tip",
                                            "right_middle1_x", "right_middle2_x", "right_middle3_x", "right_middle_tip",
                                            "right_pinky1_x", "right_pinky2_x", "right_pinky3_x", "right_pinky_tip",
                                            "right_ring1_x", "right_ring2_x", "right_ring3_x", "right_ring_tip",
                                            "right_thumb1_x", "right_thumb2_x", "right_thumb3_x", "right_thumb_tip",
            };
            std::string body_parts_l_[21] = {"left_wrist_0rz",
                                            "left_index1_x", "left_index2_x", "left_index3_x",
                                            "left_middle1_x", "left_middle2_x", "left_middle3_x",
                                            "left_pinky1_x", "left_pinky2_x", "left_pinky3_x",
                                            "left_ring1_x", "left_ring2_x", "left_ring3_x",
                                            "left_thumb1_x", "left_thumb2_x", "left_thumb3_x",
                                            "left_thumb_tip", "left_index_tip", "left_middle_tip", "left_ring_tip", "left_pinky_tip",
            };

            std::string contact_bodies_r_[16] = {"right_wrist_rz",
                                                "right_index1_z",  "right_index2_z",  "right_index3_z",
                                                "right_middle1_z", "right_middle2_z", "right_middle3_z",
                                                "right_pinky1_z",  "right_pinky2_z",  "right_pinky3_z",
                                                "right_ring1_z",   "right_ring2_z",   "right_ring3_z",
                                                "right_thumb1_z",  "right_thumb2_z",  "right_thumb3_z"
            };

            std::string contact_bodies_l_[16] = {"left_wrist_rz",
                                                "left_index1_z",  "left_index2_z",  "left_index3_z",
                                                "left_middle1_z", "left_middle2_z", "left_middle3_z",
                                                "left_pinky1_z",  "left_pinky2_z",  "left_pinky3_z",
                                                "left_ring1_z",   "left_ring2_z",   "left_ring3_z",
                                                "left_thumb1_z",  "left_thumb2_z",  "left_thumb3_z"
            };

            raisim::PolyLine *lines[21];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[21];
            raisim::Visuals *sphere[21];
            raisim::Visuals *joints_sphere[21];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
        };

    class ALLEGRO_FLOATING : public RaisimGymEnv {

        public:

            explicit ALLEGRO_FLOATING(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
                root_guided = cfg["root_guided"].As<bool>();
                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/allegro/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();


                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);


                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i=1; i < 5; i++){
                    finger_weights_contact(3*i) *= 3;
                }
                finger_weights_contact.segment(10,3) *= 2;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;


                for(int i=0; i< (num_contacts - 3); i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                for(int i=(num_contacts - 3); i< num_contacts; i++){
                    impulse_high[i] = 0.2;
                    impulse_low[i] = -0.0;
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                obDim_r_ = 108;
                obDim_l_ = 1;
                gsDim_ = 112;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                    //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    //                table_top->setPosition(1.25, 0, 0.475);
                    //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg1->setPosition(0.2625,0.4675,0.2375);
                    //                leg2->setPosition(2.2275,0.4875,0.2375);
                    //                leg3->setPosition(0.2625,-0.4675,0.2375);
                    //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();

                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);

                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);

                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>();
                gv_set_r_ = init_vel_r.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                obj_weight = arctic->getTotalMass();

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;
                        for(int i = 0; i < 3 ; i++){
                            mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                            hand_center_temp[0] += sphere_pos.e()[0];
                            hand_center_temp[1] += sphere_pos.e()[1];
                            hand_center_temp[2] += sphere_pos.e()[2];
                        }
                        hand_center_temp /= 3;
                        mano_r_->getFramePosition(body_parts_r_[16], sphere_pos);


                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }
                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();


                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);


                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame


                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),
                            right_hand_torque,
                            wrist_vel_in_wrist,
                            wrist_qvel_in_wrist,
                            contacts_r_af,
                            impulses_r_af,
                            obj_vel_in_wrist,
                            obj_qvel_in_wrist,
                            contacts_r_non_af,
                            impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
            Eigen::Vector3d current_across_axis_w;

                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);

                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
                raisim::Vec<3> obj_current_pos;
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(top_id, obj_current_pos); //Obj_Position: object base position at this time

                double displacement = (obj_current_pos.e() - obj_pos_init_.head(3)).squaredNorm();
                if (displacement > 0.1){
                    terminalReward = -10;
                    return true;
                }

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 13;
            int num_bodyparts = 17;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;


            std::string body_parts_r_[17] =  {"z_rotation_joint",
                                            "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
                                            "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
                                            "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
                                            "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"};

            std::string contact_bodies_r_[13] =  {"base_link",
                                                "link_1.0", "link_2.0", "link_3.0",    // right
                                                "link_5.0", "link_6.0", "link_7.0",
                                                "link_9.0", "link_10.0", "link_11.0",
                                                "link_13.0", "link_14.0", "link_15.0"};

            Eigen::VectorXd mean_pose;

            raisim::PolyLine *lines[17];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[17];
            raisim::Visuals *sphere[17];
            raisim::Visuals *joints_sphere[17];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
        };

    class SHADOW_FIXED : public RaisimGymEnv {
        public:

            explicit SHADOW_FIXED(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/shadow/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();

                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();

                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);

                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);


                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i = 0; i < num_contacts; i++){
                    const std::string& name = contact_bodies_r_[i];
                    if (name.find("distal") != std::string::npos) {
                        finger_weights_contact(i) *= 3.0;
                    }
                    if (name.rfind("th", 0) == 0) {  // thumb
                        finger_weights_contact(i) *= 2.0;
                    }
                }
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                finger_weights_aff.setOnes(num_bodyparts);
                const std::string tip_names[5] = {"FFtip_back", "MFtip_back", "RFtip_back", "LFtip_back", "THtip_back"};
                for(int t = 0; t < 5; t++){
                    for(int i = 0; i < num_bodyparts; i++){
                        if(body_parts_r_[i] == tip_names[t]){
                            finger_weights_aff[i] *= 4.0;
                            break;
                        }
                    }
                }
                finger_weights_aff /= finger_weights_aff.sum();
                finger_weights_aff *= num_bodyparts;

                for(int i = 0; i < num_contacts; i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                    if (contact_bodies_r_[i].rfind("th", 0) == 0) {
                        impulse_high[i] = 0.2;
                    }
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                /// MUST BE DONE FOR ALL ENVIRONMENTS
                /// Shadow: obDim = 12 + 2*gcDim + 4*num_contacts; gsDim = 10 + 6*num_bodyparts
                obDim_r_ = 136; //12 + 2 * gcDim_ + 4 * num_contacts;
                obDim_l_ = 1;
                gsDim_ = 142; //10 + 6 * num_bodyparts;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                    //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    //                table_top->setPosition(1.25, 0, 0.475);
                    //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg1->setPosition(0.2625,0.4675,0.2375);
                    //                leg2->setPosition(2.2275,0.4875,0.2375);
                    //                leg3->setPosition(0.2625,-0.4675,0.2375);
                    //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
                arctic->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_obj));
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
                }
            }

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(Eigen::VectorXd::Zero(gvDim_obj), Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                /// set initial hand pose (20 DoF) and velocity (20 DoF)
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord[0] = obj_pose[7];

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;
                        // Shadow has an extra LFJ5 node; use name-based indices when available (fallback to legacy indexing otherwise)
                        int mid_ids[4] = {-1, -1, -1, -1};
                        int thumb_tip_id = -1;
                        for(int i = 0; i < num_bodyparts; i++){
                            if(body_parts_r_[i] == "FFJ2") mid_ids[0] = i;
                            else if(body_parts_r_[i] == "MFJ2") mid_ids[1] = i;
                            else if(body_parts_r_[i] == "RFJ2") mid_ids[2] = i;
                            else if(body_parts_r_[i] == "LFJ2") mid_ids[3] = i;
                            else if(body_parts_r_[i] == "THtip_back") thumb_tip_id = i;
                        }
                        if(mid_ids[0] >= 0 && mid_ids[1] >= 0 && mid_ids[2] >= 0 && mid_ids[3] >= 0){
                            for(int k = 0; k < 4; k++){
                                mano_r_->getFramePosition(body_parts_r_[mid_ids[k]], sphere_pos);
                                hand_center_temp[0] += sphere_pos.e()[0];
                                hand_center_temp[1] += sphere_pos.e()[1];
                                hand_center_temp[2] += sphere_pos.e()[2];
                            }
                        } else {
                            for(int i = 0; i < 4 ; i++){
                                mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                                hand_center_temp[0] += sphere_pos.e()[0];
                                hand_center_temp[1] += sphere_pos.e()[1];
                                hand_center_temp[2] += sphere_pos.e()[2];
                            }
                        }
                        hand_center_temp /= 4;
                        mano_r_->getFramePosition(body_parts_r_[(thumb_tip_id >= 0) ? thumb_tip_id : 20], sphere_pos);
                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }

                Eigen::Vector3d wrist_bias_w, wrist_bias;
                wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                wrist_bias = init_or_r_.e() * wrist_bias_w;
                actionMean_r_.head(3) += wrist_bias;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)


                /// Clip targets to limits
                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = obj_vel_in_wrist.squaredNorm();
                obj_qvel_reward_r = obj_qvel_in_wrist.squaredNorm();

                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);

                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame

                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);


                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);


                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),      //  generalized coordinate
                            right_hand_torque,         //  joint torque
                            wrist_vel_in_wrist,
                            wrist_qvel_in_wrist,
                            contacts_r_af,
                            impulses_r_af,
                            obj_vel_in_wrist,
                            obj_qvel_in_wrist,
                            contacts_r_non_af,
                            impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
            Eigen::Vector3d current_across_axis_w;
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);
                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double close_finger_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 17;
            int num_bodyparts = 22;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;

            std::string body_parts_r_[22] =  {"z_rotation_joint",    // 0
                "FFJ3", "FFJ2", "FFJ1","FFtip_back", // 0.100568，0.045，0.025，0.026
                "MFJ3", "MFJ2", "MFJ1","MFtip_back", // 0.0996092，0.045，0.025，0.026
                "RFJ3", "RFJ2", "RFJ1","RFtip_back", // 0.0956347，0.045，0.025，0.026
                "LFJ5","LFJ3", "LFJ2", "LFJ1","LFtip_back", // 0.0925597，0.045，0.025，0.026
                "THJ4", "THJ2", "THJ1","THtip_back"  // 0.045489，0.038，0.032，0.0275
                
            };

            std::string contact_bodies_r_[17] =  {"palm",  // base_link
                                "ffproximal", "ffmiddle", "ffdistal", // link_1.0 link_2.0   link_3.0_tip
                                "mfproximal", "mfmiddle", "mfdistal", // link_5.0 link_6.0   link_7.0_tip
                                "rfproximal", "rfmiddle", "rfdistal", // link_9.0 link_10.0   link_11.0_tip
                                "lfmetacarpal","lfproximal", "lfmiddle", "lfdistal",
                                "thproximal", "thmiddle", "thdistal"  // link_14.0 link_15.0   link_15.0_tip
            };
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[22];
            raisim::Visuals *sphere[22];
            raisim::Visuals *joints_sphere[22];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
        };

    class MANO_FIXED : public RaisimGymEnv {

        public:

            explicit MANO_FIXED(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/mano_double/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();


                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);

                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i=1; i < 6; i++){
                    finger_weights_contact(3*i) *= 3;
                }
                finger_weights_contact.segment(13,3) *= 2;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                // finger_weights_aff.setOnes(num_bodyparts);
                // for(int i=0; i < 5; i++){
                //     finger_weights_aff[4 * i+4] *= 4.0;
                // }
                // finger_weights_aff /= finger_weights_aff.sum();
                // finger_weights_aff *= num_bodyparts;

                for(int i=0; i< (num_contacts - 3); i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                for(int i=13; i< 16; i++){
                    impulse_high[i] = 0.2;
                    impulse_low[i] = -0.0;
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                /// MUST BE DONE FOR ALL ENVIRONMENTS
                obDim_r_ = 178;
                obDim_l_ = 1;
                gsDim_ = 196;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                    //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    //                table_top->setPosition(1.25, 0, 0.475);
                    //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg1->setPosition(0.2625,0.4675,0.2375);
                    //                leg2->setPosition(2.2275,0.4875,0.2375);
                    //                leg3->setPosition(0.2625,-0.4675,0.2375);
                    //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
                arctic->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_obj));
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
                }

            }

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(Eigen::VectorXd::Zero(gvDim_obj), Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord[0] = obj_pose[7];

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                updateObservation();
            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;

                        for(int i = 0; i < 4 ; i++){
                            mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                            hand_center_temp[0] += sphere_pos.e()[0];
                            hand_center_temp[1] += sphere_pos.e()[1];
                            hand_center_temp[2] += sphere_pos.e()[2];
                        }
                        hand_center_temp /= 4;
                        mano_r_->getFramePosition(body_parts_r_[20], sphere_pos);
                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }

                Eigen::Vector3d wrist_bias_w, wrist_bias;
                wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                wrist_bias = init_or_r_.e() * wrist_bias_w;
                actionMean_r_.head(3) += wrist_bias;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;
                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = obj_vel_in_wrist.squaredNorm();
                obj_qvel_reward_r = obj_qvel_in_wrist.squaredNorm();

                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);

                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame

                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);


                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),      // (mirror) 45, generalized coordinate
                                right_hand_torque,         // (mirror) 51, joint torque
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }

            Eigen::Vector3d current_across_axis_w;
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);

                raisim::Vec<3> trans_r;
                raisim::Mat<3,3> rot_r;
                raisim::Vec<4> quat_r;

                Eigen::VectorXd obj_state;
                obj_state.setZero(8);

                mano_r_->getFramePosition(body_parts_r_[0], trans_r); // Position: body position of a body in mano
                mano_r_->getFrameOrientation(body_parts_r_[0], rot_r); // Body_orientation:
                raisim::rotMatToQuat(rot_r, quat_r);

                obj_state = obj_pos_init_;

                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul,
                                trans_r.e(),
                                quat_r.e(),
                                gc_r_.tail(gcDim_ - 6),
                                obj_state;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 16;
            int num_bodyparts = 21;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;

            std::string body_parts_r_[21] = {"right_wrist_0rz",
                                            "right_index1_x", "right_index2_x", "right_index3_x", "right_index_tip",
                                            "right_middle1_x", "right_middle2_x", "right_middle3_x", "right_middle_tip",
                                            "right_pinky1_x", "right_pinky2_x", "right_pinky3_x", "right_pinky_tip",
                                            "right_ring1_x", "right_ring2_x", "right_ring3_x", "right_ring_tip",
                                            "right_thumb1_x", "right_thumb2_x", "right_thumb3_x", "right_thumb_tip",
            };
            // std::string body_parts_l_[21] = {"left_wrist_0rz",
            //                                  "left_index1_x", "left_index2_x", "left_index3_x",
            //                                  "left_middle1_x", "left_middle2_x", "left_middle3_x",
            //                                  "left_pinky1_x", "left_pinky2_x", "left_pinky3_x",
            //                                  "left_ring1_x", "left_ring2_x", "left_ring3_x",
            //                                  "left_thumb1_x", "left_thumb2_x", "left_thumb3_x",
            //                                  "left_thumb_tip", "left_index_tip", "left_middle_tip", "left_ring_tip", "left_pinky_tip",
            // };

            std::string contact_bodies_r_[16] = {"right_wrist_rz",
                                                "right_index1_z",  "right_index2_z",  "right_index3_z",
                                                "right_middle1_z", "right_middle2_z", "right_middle3_z",
                                                "right_pinky1_z",  "right_pinky2_z",  "right_pinky3_z",
                                                "right_ring1_z",   "right_ring2_z",   "right_ring3_z",
                                                "right_thumb1_z",  "right_thumb2_z",  "right_thumb3_z"
            };

            // std::string contact_bodies_l_[16] = {"left_wrist_rz",
            //                                      "left_index1_z",  "left_index2_z",  "left_index3_z",
            //                                      "left_middle1_z", "left_middle2_z", "left_middle3_z",
            //                                      "left_pinky1_z",  "left_pinky2_z",  "left_pinky3_z",
            //                                      "left_ring1_z",   "left_ring2_z",   "left_ring3_z",
            //                                      "left_thumb1_z",  "left_thumb2_z",  "left_thumb3_z"
            // };

            // raisim::PolyLine *lines[21];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[21];
            raisim::Visuals *sphere[21];
            raisim::Visuals *joints_sphere[21];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
        };

    class ALLEGRO_FIXED : public RaisimGymEnv {

        public:

            explicit ALLEGRO_FIXED(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/allegro/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();

                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);

                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);


                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i=1; i < 5; i++){
                    finger_weights_contact(3*i) *= 3;
                }
                finger_weights_contact.segment(10,3) *= 2;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                for(int i=0; i< (num_contacts - 3); i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                for(int i=(num_contacts - 3); i< num_contacts; i++){
                    impulse_high[i] = 0.2;
                    impulse_low[i] = -0.0;
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                obDim_r_ = 108;
                obDim_l_ = 1;
                gsDim_ = 112;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                    //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    //                table_top->setPosition(1.25, 0, 0.475);
                    //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg1->setPosition(0.2625,0.4675,0.2375);
                    //                leg2->setPosition(2.2275,0.4875,0.2375);
                    //                leg3->setPosition(0.2625,-0.4675,0.2375);
                    //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
                arctic->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_obj));
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
            }

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(Eigen::VectorXd::Zero(gvDim_obj), Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>();
                gv_set_r_ = init_vel_r.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord[0] = obj_pose[7];

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state
                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;
                        for(int i = 0; i < 3 ; i++){
                            mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                            hand_center_temp[0] += sphere_pos.e()[0];
                            hand_center_temp[1] += sphere_pos.e()[1];
                            hand_center_temp[2] += sphere_pos.e()[2];
                        }
                        hand_center_temp /= 3;
                        mano_r_->getFramePosition(body_parts_r_[16], sphere_pos);
                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }

                Eigen::Vector3d wrist_bias_w, wrist_bias;
                wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                wrist_bias = init_or_r_.e() * wrist_bias_w;
                actionMean_r_.head(3) += wrist_bias;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)


                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = obj_vel_in_wrist.squaredNorm();
                obj_qvel_reward_r = obj_qvel_in_wrist.squaredNorm();

                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;


                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);

                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame

                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);


                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                //108total
                obDouble_r_ << target_center_dif, //3
                            gc_r_.segment(3, 3) - wrist_target_euler.e(), //3
                            gc_r_.tail(gcDim_ - 6),  //16  16metadatajoint
                            right_hand_torque,    //22 metadataact_dim
                            wrist_vel_in_wrist,   //3
                            wrist_qvel_in_wrist,   //3
                            contacts_r_af,     //13
                            impulses_r_af,     //13
                            obj_vel_in_wrist,   //3
                            obj_qvel_in_wrist,  //3
                            contacts_r_non_af, //13
                            impulses_r_non_af; //13

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
            Eigen::Vector3d current_across_axis_w;
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);
                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double close_finger_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 13;
            int num_bodyparts = 17;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;

            std::string body_parts_r_[17] =  {"z_rotation_joint",
                                            "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
                                            "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
                                            "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
                                            "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"};

            std::string contact_bodies_r_[13] =  {"base_link",
                                                "link_1.0", "link_2.0", "link_3.0",    // right
                                                "link_5.0", "link_6.0", "link_7.0",
                                                "link_9.0", "link_10.0", "link_11.0",
                                                "link_13.0", "link_14.0", "link_15.0"};

            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[17];
            raisim::Visuals *sphere[17];
            raisim::Visuals *joints_sphere[17];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
        };


    class LEAP_FIXED : public RaisimGymEnv {
        public:

            explicit LEAP_FIXED(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;

                mano_r_ = world_->addArticulatedSystem(resourceDir+"/leap_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();

                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();

                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);

                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);


                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                // For now, setting a uniform weight. Example of old Allegro hand specific weights:
                for(int i=1; i < 5; i++){
                    finger_weights_contact(3*i) *= 3;
                }
                finger_weights_contact.segment(10,3) *= 2;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                for(int i=0; i< (num_contacts - 3); i++){ // The last 3 contacts (thumb links) likely have different impulse limits.
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                for(int i=(num_contacts - 3); i< num_contacts; i++){
                    impulse_high[i] = 0.2;
                    impulse_low[i] = -0.0;
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                obDim_r_ = 108;
                obDim_l_ = 1;
                gsDim_ = 112;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                //                table_top->setPosition(1.25, 0, 0.475);
                //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg1->setPosition(0.2625,0.4675,0.2375);
                //                leg2->setPosition(2.2275,0.4875,0.2375);
                //                leg3->setPosition(0.2625,-0.4675,0.2375);
                //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();

                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));

                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
                arctic->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_obj));
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
            }

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(Eigen::VectorXd::Zero(gvDim_obj), Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                                const Eigen::Ref<EigenVec>& init_state_l,
                                const Eigen::Ref<EigenVec>& init_vel_r,
                                const Eigen::Ref<EigenVec>& init_vel_l,
                                const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>();
                gv_set_r_ = init_vel_r.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord[0] = obj_pose[7];

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state
                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
                updateObservation();
            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                            const Eigen::Ref<EigenVec>& obj_center,
                            const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                            const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                            const Eigen::Ref<EigenVec>& goal_pose_r,
                            const Eigen::Ref<EigenVec>& goal_pose_l,
                            const Eigen::Ref<EigenVec>& goal_qpos_r,
                            const Eigen::Ref<EigenVec>& goal_qpos_l,
                            const Eigen::Ref<EigenVec>& goal_contacts_r,
                            const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                            afford_center = target_center.cast<double>();
                            obj_center_o = obj_center.cast<double>();
                            Eigen::Vector3d hand_center_temp;
                            hand_center.setZero();
                            hand_center_temp.setZero();
                            raisim::Vec<3> sphere_pos;
                            for(int i = 0; i < 3 ; i++){
                                mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                                hand_center_temp[0] += sphere_pos.e()[0];
                                hand_center_temp[1] += sphere_pos.e()[1];
                                hand_center_temp[2] += sphere_pos.e()[2];
                            }
                            hand_center_temp /= 3;
                            mano_r_->getFramePosition(body_parts_r_[16], sphere_pos);
                            Eigen::Vector3d finger_center;
                            finger_center.setZero();
                            finger_center[0] = hand_center_temp[0];
                            finger_center[1] = hand_center_temp[1];
                            finger_center[2] = hand_center_temp[2];

                            hand_center_temp[0] += sphere_pos.e()[0];
                            hand_center_temp[1] += sphere_pos.e()[1];
                            hand_center_temp[2] += sphere_pos.e()[2];
                            hand_center_temp /= 2;

                            Eigen::Vector3d across_axis_w;
                            across_axis_w = finger_center - hand_center_temp;
                            across_axis_w = across_axis_w / across_axis_w.norm();

                            raisim::Vec<3> wrist_pos_w;
                            raisim::Mat<3,3> wrist_mat_r;
                            mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                            mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                            Eigen::Vector3d bias, wrist_target_temp;
                            auto affordance_id = arctic->getBodyIdx("top");
                            arctic->getPosition(affordance_id, Obj_Position_init);
                            arctic->getOrientation(affordance_id, Obj_orientation_init);
                            raisim::Mat<3,3> Obj_orientation_init_trans;
                            raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                            raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                            bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                            wrist_target_temp = wrist_pos_w.e() + bias;
                            wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                            hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                            hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                            hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                            hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                            Eigen::Vector3d grasp_axis_w;
                            grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                            grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                            across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                            across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                            }

            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }

                Eigen::Vector3d wrist_bias_w, wrist_bias;
                wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                wrist_bias = init_or_r_.e() * wrist_bias_w;
                actionMean_r_.head(3) += wrist_bias;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)


                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = obj_vel_in_wrist.squaredNorm();
                obj_qvel_reward_r = obj_qvel_in_wrist.squaredNorm();

                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;


                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }
                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");

                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);

                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame

                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) { continue; }
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) { continue; }
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) { continue; }
                    
                    if (contactMapping_r_.find(contact_af.getlocalBodyIndex()) == contactMapping_r_.end()) {
                        // Optionally, you might want to throw an exception or handle this more gracefully
                    } else {
                        contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                        impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                    }
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) { continue; }
                    
                    if (contactMapping_r_.find(contact_non_af.getlocalBodyIndex()) == contactMapping_r_.end()) {
                        // Optionally, you might want to throw an exception or handle this more gracefully
                    } else {
                        contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                        impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                    }
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);

                obDouble_r_ << target_center_dif,
                                gc_r_.segment(3, 3) - wrist_target_euler.e(),
                                gc_r_.tail(gcDim_ - 6),
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }

                Eigen::Vector3d current_across_axis_w;
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

                Eigen::Vector3d current_grasp_axis_w;
                current_grasp_axis_w = wrist_mat_r.e() * hand_center;
                current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

                Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
                current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
                current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

                float across_mul = 0;
                across_mul = current_across_axis_o.dot(across_axis_o);
                
                global_state_ << obj_pose_wrist.e(),
                                    frame_y_in_obj, // Note: This is now a placeholder, ensure it's not actually used for calculations if it's deprecated.
                                    joint_pos_in_obj,
                                    Obj_Position.e(),
                                    current_grasp_axis_o - grasp_axis_o,
                                    across_mul;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double close_finger_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 13;
            int num_bodyparts = 17;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;

            std::string body_parts_r_[17] =  {"z_rotation_joint",
                "1", "0", "3", "index_tip",  //02metadatapipmetadatacontact
                "5", "4",  "7","middle_tip",
                "9", "8", "11", "ring_tip",
                "12", "14","15", "thumb_tip"}; //12+13metadata

            std::string contact_bodies_r_[13] =  {"palm_lower",
                                                    "mcp_joint", "pip", "dip",
                                                    "mcp_joint_2", "pip_2", "dip_2",
                                                    "mcp_joint_3", "pip_3", "dip_3",
                                                    "thumb_temp_base", "thumb_pip", "thumb_dip"};

            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[17];
            raisim::Visuals *sphere[17];
            raisim::Visuals *joints_sphere[17];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class LEAP_FLOATING : public RaisimGymEnv {
        public:

            explicit LEAP_FLOATING(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
                pinch_grasp= false;
                pinch_grasp = cfg["pinch_grasp"].As<bool>();
                visualizable_ = cfg["visualize"].As<bool>();
                root_guided = cfg["root_guided"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/leap_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();


                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);


                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i=1; i < 5; i++){
                    finger_weights_contact(3*i) *= 3;
                }
                finger_weights_contact.segment(10,3) *= 2;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;


                for(int i=0; i< (num_contacts - 3); i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                for(int i=(num_contacts - 3); i< num_contacts; i++){
                    impulse_high[i] = 0.2;
                    impulse_low[i] = -0.0;
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                obDim_r_ = 108;
                obDim_l_ = 1;
                gsDim_ = 112;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                //                table_top->setPosition(1.25, 0, 0.475);
                //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg1->setPosition(0.2625,0.4675,0.2375);
                //                leg2->setPosition(2.2275,0.4875,0.2375);
                //                leg3->setPosition(0.2625,-0.4675,0.2375);
                //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();

                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);

                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);

                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>();
                gv_set_r_ = init_vel_r.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                obj_weight = arctic->getTotalMass();

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();

                        Eigen::Vector3d finger_center;
                        if(pinch_grasp){
                            raisim::Vec<3> sphere_pos;
                            // index and thumb tip
                            mano_r_->getFramePosition(body_parts_r_[4], sphere_pos);
                            hand_center_temp[0] += sphere_pos.e()[0];
                            hand_center_temp[1] += sphere_pos.e()[1];
                            hand_center_temp[2] += sphere_pos.e()[2];
                            
                            mano_r_->getFramePosition(body_parts_r_[16], sphere_pos);


                            finger_center.setZero();
                            finger_center[0] = hand_center_temp[0];
                            finger_center[1] = hand_center_temp[1];
                            finger_center[2] = hand_center_temp[2];

                            hand_center_temp[0] += sphere_pos.e()[0];
                            hand_center_temp[1] += sphere_pos.e()[1];
                            hand_center_temp[2] += sphere_pos.e()[2];
                            hand_center_temp /= 2;

                        }
                        else{
                            raisim::Vec<3> sphere_pos;
                            for(int i = 0; i < 3 ; i++){
                                mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                                hand_center_temp[0] += sphere_pos.e()[0];
                                hand_center_temp[1] += sphere_pos.e()[1];
                                hand_center_temp[2] += sphere_pos.e()[2];
                            }
                            hand_center_temp /= 3;
                            mano_r_->getFramePosition(body_parts_r_[16], sphere_pos);


                            finger_center.setZero();
                            finger_center[0] = hand_center_temp[0];
                            finger_center[1] = hand_center_temp[1];
                            finger_center[2] = hand_center_temp[2];

                            hand_center_temp[0] += sphere_pos.e()[0];
                            hand_center_temp[1] += sphere_pos.e()[1];
                            hand_center_temp[2] += sphere_pos.e()[2];
                            hand_center_temp /= 2;
                        }

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }            

                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();


                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);


                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame


                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
            Eigen::Vector3d current_across_axis_w;

                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);

                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
                raisim::Vec<3> obj_current_pos;
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(top_id, obj_current_pos); //Obj_Position: object base position at this time

                double displacement = (obj_current_pos.e() - obj_pos_init_.head(3)).squaredNorm();
                if (displacement > 0.1){
                    terminalReward = -10;
                    return true;
                }

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool pinch_grasp = true;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 13;
            int num_bodyparts = 17;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;


            std::string body_parts_r_[17] =  {"z_rotation_joint",
                "1", "0", "3", "index_tip",  //02metadatapipmetadatacontact
                "5", "4",  "7","middle_tip",
                "9", "8", "11", "ring_tip",
                "12", "14","15", "thumb_tip"}; //12+13metadata

            std::string contact_bodies_r_[13] =  {"palm_lower",
                            "mcp_joint", "pip", "dip",
                            "mcp_joint_2", "pip_2", "dip_2",
                            "mcp_joint_3", "pip_3", "dip_3",
                            "thumb_temp_base", "thumb_pip", "thumb_dip"};

            Eigen::VectorXd mean_pose;

            raisim::PolyLine *lines[17];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[17];
            raisim::Visuals *sphere[17];
            raisim::Visuals *joints_sphere[17];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class LEAP_TEST : public RaisimGymEnv {
        public:

            explicit LEAP_TEST(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                root_guided = cfg["root_guided"].As<bool>();
                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                lift = false;

                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/leap_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();

                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);


                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i=1; i < 5; i++){
                    finger_weights_contact(3*i) *= 3;
                }
                finger_weights_contact.segment(10,3) *= 2;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                for(int i=0; i< (num_contacts - 3); i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                for(int i=(num_contacts - 3); i< num_contacts; i++){
                    impulse_high[i] = 0.2;
                    impulse_low[i] = -0.0;
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                /// MUST BE DONE FOR ALL ENVIRONMENTS
                obDim_r_ = 108;
                obDim_l_ = 1;
                gsDim_ = 143;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
            //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
            //                table_top->setPosition(1.25, 0, 0.475);
            //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg1->setPosition(0.2625,0.4675,0.2375);
            //                leg2->setPosition(2.2275,0.4875,0.2375);
            //                leg3->setPosition(0.2625,-0.4675,0.2375);
            //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");
                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();

                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    lift = false;
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);

                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                lift = false;

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);

                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                obj_weight = arctic->getTotalMass();

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;
                        for(int i = 0; i < 3 ; i++){
                            mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                            hand_center_temp[0] += sphere_pos.e()[0];
                            hand_center_temp[1] += sphere_pos.e()[1];
                            hand_center_temp[2] += sphere_pos.e()[2];
                        }
                        hand_center_temp /= 3;
                        mano_r_->getFramePosition(body_parts_r_[16], sphere_pos);


                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
    
                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }
                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    if (lift){
                        wrist_bias_w[2] += 0.2;
                    }
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;

                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    if (lift){
                        wrist_bias_w[2] += 0.2;
                    }
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();


                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::max(0.0, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);


                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame

                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);

                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
            Eigen::Vector3d current_across_axis_w;

                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);


            raisim::Vec<3> trans_r;
            raisim::Mat<3,3> rot_r;
            raisim::Vec<4> quat_r;

            Eigen::VectorXd obj_state;
            obj_state.setZero(8);

            mano_r_->getFramePosition(body_parts_r_[0], trans_r); // Position: body position of a body in mano
            mano_r_->getFrameOrientation(body_parts_r_[0], rot_r); // Body_orientation:

            raisim::rotMatToQuat(rot_r, quat_r);
            obj_state = arctic->getGeneralizedCoordinate().e();

                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul,
                                trans_r.e(),
                                quat_r.e(),
                                gc_r_.tail(gcDim_ - 6),
                                obj_state;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {
                lift = true;
            }
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            bool lift = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 13;
            int num_bodyparts = 17;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;


            std::string body_parts_r_[17] =  {"z_rotation_joint",
                "1", "0", "3", "index_tip",  //02metadatapipmetadatacontact
                "5", "4",  "7","middle_tip",
                "9", "8", "11", "ring_tip",
                "12", "14","15", "thumb_tip"}; //12+13metadata

            std::string contact_bodies_r_[13] =  {"palm_lower",
                            "mcp_joint", "pip", "dip",
                            "mcp_joint_2", "pip_2", "dip_2",
                            "mcp_joint_3", "pip_3", "dip_3",
                            "thumb_temp_base", "thumb_pip", "thumb_dip"};


            Eigen::VectorXd mean_pose;

            raisim::PolyLine *lines[17];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[17];
            raisim::Visuals *sphere[17];
            raisim::Visuals *joints_sphere[17];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class SHADOW_TEST : public RaisimGymEnv {
        public:

            explicit SHADOW_TEST(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                root_guided = cfg["root_guided"].As<bool>();
                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                lift = false;

                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/shadow/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();

                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);


                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i = 0; i < num_contacts; i++){
                    const std::string& name = contact_bodies_r_[i];
                    if (name.find("distal") != std::string::npos) {
                        finger_weights_contact(i) *= 3.0;
                    }
                    if (name.rfind("th", 0) == 0) {  // thumb
                        finger_weights_contact(i) *= 2.0;
                    }
                }
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                finger_weights_aff.setOnes(num_bodyparts);
                const std::string tip_names[5] = {"FFtip_back", "MFtip_back", "RFtip_back", "LFtip_back", "THtip_back"};
                for(int t = 0; t < 5; t++){
                    for(int i = 0; i < num_bodyparts; i++){
                        if(body_parts_r_[i] == tip_names[t]){
                            finger_weights_aff[i] *= 4.0;
                            break;
                        }
                    }
                }
                finger_weights_aff /= finger_weights_aff.sum();
                finger_weights_aff *= num_bodyparts;

                for(int i = 0; i < num_contacts; i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                    if (contact_bodies_r_[i].rfind("th", 0) == 0) {
                        impulse_high[i] = 0.2;
                    }
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                /// MUST BE DONE FOR ALL ENVIRONMENTS
                /// Shadow test: obDim = 12 + 2*gcDim + 4*num_contacts; gsDim = 19 + 6*num_bodyparts + gcDim
                obDim_r_ = 136;
                obDim_l_ = 1;
                gsDim_ = 179; //19 + 6 * num_bodyparts + gcDim_; //173+6=179
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
            //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
            //                table_top->setPosition(1.25, 0, 0.475);
            //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg1->setPosition(0.2625,0.4675,0.2375);
            //                leg2->setPosition(2.2275,0.4875,0.2375);
            //                leg3->setPosition(0.2625,-0.4675,0.2375);
            //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");
                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();

                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    lift = false;
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);

                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                lift = false;

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);

                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                /// set initial hand pose (20 DoF) and velocity (20 DoF)
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                obj_weight = arctic->getTotalMass();

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;
                        // Shadow has an extra LFJ5 node; use name-based indices when available (fallback to legacy indexing otherwise)
                        int mid_ids[4] = {-1, -1, -1, -1};
                        int thumb_tip_id = -1;
                        for(int i = 0; i < num_bodyparts; i++){
                            if(body_parts_r_[i] == "FFJ2") mid_ids[0] = i;
                            else if(body_parts_r_[i] == "MFJ2") mid_ids[1] = i;
                            else if(body_parts_r_[i] == "RFJ2") mid_ids[2] = i;
                            else if(body_parts_r_[i] == "LFJ2") mid_ids[3] = i;
                            else if(body_parts_r_[i] == "THtip_back") thumb_tip_id = i;
                        }
                        if(mid_ids[0] >= 0 && mid_ids[1] >= 0 && mid_ids[2] >= 0 && mid_ids[3] >= 0){
                            for(int k = 0; k < 4; k++){
                                mano_r_->getFramePosition(body_parts_r_[mid_ids[k]], sphere_pos);
                                hand_center_temp[0] += sphere_pos.e()[0];
                                hand_center_temp[1] += sphere_pos.e()[1];
                                hand_center_temp[2] += sphere_pos.e()[2];
                            }
                        } else {
                            for(int i = 0; i < 4 ; i++){
                                mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                                hand_center_temp[0] += sphere_pos.e()[0];
                                hand_center_temp[1] += sphere_pos.e()[1];
                                hand_center_temp[2] += sphere_pos.e()[2];
                            }
                        }
                        hand_center_temp /= 4;
                        mano_r_->getFramePosition(body_parts_r_[(thumb_tip_id >= 0) ? thumb_tip_id : 20], sphere_pos);


                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }
                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    if (lift){
                        wrist_bias_w[2] += 0.2;
                    }
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;

                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    if (lift){
                        wrist_bias_w[2] += 0.2;
                    }
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();


                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::max(0.0, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);


                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame

                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);

                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),      // (mirror) 45, generalized coordinate
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
                Eigen::Vector3d current_across_axis_w;

                    current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

                Eigen::Vector3d current_grasp_axis_w;
                current_grasp_axis_w = wrist_mat_r.e() * hand_center;
                current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

                Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
                current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
                current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

                float across_mul = 0;
                across_mul = current_across_axis_o.dot(across_axis_o);

                raisim::Vec<3> trans_r;
                raisim::Mat<3,3> rot_r;
                raisim::Vec<4> quat_r;

                Eigen::VectorXd obj_state;
                obj_state.setZero(8);

                mano_r_->getFramePosition(body_parts_r_[0], trans_r); // Position: body position of a body in mano
                mano_r_->getFrameOrientation(body_parts_r_[0], rot_r); // Body_orientation:

                raisim::rotMatToQuat(rot_r, quat_r);
                    obj_state = arctic->getGeneralizedCoordinate().e();
                    global_state_ << obj_pose_wrist.e(),
                                    frame_y_in_obj,
                                    joint_pos_in_obj,
                                    Obj_Position.e(),
                                    current_grasp_axis_o - grasp_axis_o,
                                    across_mul,
                                    trans_r.e(),
                                    quat_r.e(),
                                    gc_r_.tail(gcDim_ - 6),
                                    obj_state;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {
                lift = true;
            }
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            bool lift = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 17;
            int num_bodyparts = 22;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;

            std::string body_parts_r_[22] =  {"z_rotation_joint",    // 0
                                            "FFJ3", "FFJ2", "FFJ1","FFtip_back", // 0.100568，0.045，0.025，0.026
                                            "MFJ3", "MFJ2", "MFJ1","MFtip_back", // 0.0996092，0.045，0.025，0.026
                                            "RFJ3", "RFJ2", "RFJ1","RFtip_back", // 0.0956347，0.045，0.025，0.026
                                            "LFJ5","LFJ3", "LFJ2", "LFJ1","LFtip_back", // 0.0925597，0.045，0.025，0.026
                                            "THJ4", "THJ2", "THJ1","THtip_back"  // 0.045489，0.038，0.032，0.0275
                                            
            };

            std::string contact_bodies_r_[17] =  {"palm",  // base_link
                                                "ffproximal", "ffmiddle", "ffdistal", // link_1.0 link_2.0   link_3.0_tip
                                                "mfproximal", "mfmiddle", "mfdistal", // link_5.0 link_6.0   link_7.0_tip
                                                "rfproximal", "rfmiddle", "rfdistal", // link_9.0 link_10.0   link_11.0_tip
                                                "lfmetacarpal","lfproximal", "lfmiddle", "lfdistal",
                                                "thproximal", "thmiddle", "thdistal"  // link_14.0 link_15.0   link_15.0_tip
            };


            Eigen::VectorXd mean_pose;

            raisim::PolyLine *lines[22];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[22];
            raisim::Visuals *sphere[22];
            raisim::Visuals *joints_sphere[22];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class ALLEGRO_TEST : public RaisimGymEnv {

        public:
    
            explicit ALLEGRO_TEST(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
    
                root_guided = cfg["root_guided"].As<bool>();
                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                lift = false;
    
                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);
    
                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);
    
                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/allegro/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();
    
                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");
    
                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    
                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;
    
                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);
    
                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);
    
                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);
    
    
                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);
    
                contact_body_idx_r_.setZero(num_contacts);
    
                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();
    
                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);
    
                pTarget_clipped_r.setZero(gcDim_);
    
                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);
    
                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i=1; i < 5; i++){
                    finger_weights_contact(3*i) *= 3;
                }
                finger_weights_contact.segment(10,3) *= 2;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;
    
                for(int i=0; i< (num_contacts - 3); i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                for(int i=(num_contacts - 3); i< num_contacts; i++){
                    impulse_high[i] = 0.2;
                    impulse_low[i] = -0.0;
                }
    
    
                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
    
                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));
    
                /// MUST BE DONE FOR ALL ENVIRONMENTS
                obDim_r_ = 108;
                obDim_l_ = 1;
                gsDim_ = 143;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);
    
                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();
    
                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();
    
                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }
    
                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);
    
                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);
    
                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }
    
                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);
    
                    /// Create table
                //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                //                table_top->setPosition(1.25, 0, 0.475);
                //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                //                leg1->setPosition(0.2625,0.4675,0.2375);
                //                leg2->setPosition(2.2275,0.4875,0.2375);
                //                leg3->setPosition(0.2625,-0.4675,0.2375);
                //                leg4->setPosition(2.2275,-0.4875,0.2375);
    
                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);
    
                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }
    
            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");
                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
    
                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }
    
            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}
    
            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    lift = false;
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);
    
                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object
    
                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);
    
                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
    
                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }
    
            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                             const Eigen::Ref<EigenVec>& init_state_l,
                             const Eigen::Ref<EigenVec>& init_vel_r,
                             const Eigen::Ref<EigenVec>& init_vel_l,
                             const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);
    
                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                lift = false;
    
                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);
    
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();
    
                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse
    
                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();
    
                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial object state
    
                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof
    
                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);
    
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                obj_weight = arctic->getTotalMass();
    
                updateObservation();
    
            }
    
            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                           const Eigen::Ref<EigenVec>& obj_center,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                           const Eigen::Ref<EigenVec>& goal_pose_r,
                           const Eigen::Ref<EigenVec>& goal_pose_l,
                           const Eigen::Ref<EigenVec>& goal_qpos_r,
                           const Eigen::Ref<EigenVec>& goal_qpos_l,
                           const Eigen::Ref<EigenVec>& goal_contacts_r,
                           const Eigen::Ref<EigenVec>& goal_contacts_l) final {
    
                           afford_center = target_center.cast<double>();
                           obj_center_o = obj_center.cast<double>();
                           Eigen::Vector3d hand_center_temp;
                           hand_center.setZero();
                           hand_center_temp.setZero();
                           raisim::Vec<3> sphere_pos;
                           for(int i = 0; i < 3 ; i++){
                               mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                               hand_center_temp[0] += sphere_pos.e()[0];
                               hand_center_temp[1] += sphere_pos.e()[1];
                               hand_center_temp[2] += sphere_pos.e()[2];
                           }
                           hand_center_temp /= 3;
                           mano_r_->getFramePosition(body_parts_r_[16], sphere_pos);
    
    
                           Eigen::Vector3d finger_center;
                           finger_center.setZero();
                           finger_center[0] = hand_center_temp[0];
                           finger_center[1] = hand_center_temp[1];
                           finger_center[2] = hand_center_temp[2];
    
                           hand_center_temp[0] += sphere_pos.e()[0];
                           hand_center_temp[1] += sphere_pos.e()[1];
                           hand_center_temp[2] += sphere_pos.e()[2];
                           hand_center_temp /= 2;
    
                           Eigen::Vector3d across_axis_w;
                           across_axis_w = finger_center - hand_center_temp;
                           across_axis_w = across_axis_w / across_axis_w.norm();
    
                           raisim::Vec<3> wrist_pos_w;
                           raisim::Mat<3,3> wrist_mat_r;
                           mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                           mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                           Eigen::Vector3d bias, wrist_target_temp;
                           auto affordance_id = arctic->getBodyIdx("top");
                           arctic->getPosition(affordance_id, Obj_Position_init);
                           arctic->getOrientation(affordance_id, Obj_orientation_init);
                           raisim::Mat<3,3> Obj_orientation_init_trans;
                           raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                           raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                           bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                           wrist_target_temp = wrist_pos_w.e() + bias;
                           wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());
    
                           hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                           hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                           hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                           hand_center = wrist_mat_r.e().transpose() * hand_center_temp;
    
                           Eigen::Vector3d grasp_axis_w;
                           grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                           grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                           across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                           across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                           }
    
            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
    
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);
    
                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
    
                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }
                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    if (lift){
                        wrist_bias_w[2] += 0.2;
                    }
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;

                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    if (lift){
                        wrist_bias_w[2] += 0.2;
                    }
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }
    
                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)
    
                /// Clip targets to limits
    
                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);
    
                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);
    
                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;
    
                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }
    
                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
    
                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }
    
                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();
    
    
                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::max(0.0, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));
    
                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;
    
                return rewards_sum_;
            }
    
            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }
    
                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();
    
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);
    
                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();
    
                mano_r_->getState(gc_r_, gv_r_);
    
                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);
    
    
                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame
    
                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();
    
                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }
    
                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }
    
                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }
    
                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }
    
                right_hand_torque = (pTarget_clipped_r - gc_r_);
    
                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;
    
                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
    
                obDouble_r_ << target_center_dif,           // 3, hand center diff
                               gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                               gc_r_.tail(gcDim_ - 6),
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;
    
                raisim::transpose(Obj_orientation_temp, Obj_orientation);
    
                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);
    
                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];
    
                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);
    
                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
               Eigen::Vector3d current_across_axis_w;
    
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;
    
               Eigen::Vector3d current_grasp_axis_w;
               current_grasp_axis_w = wrist_mat_r.e() * hand_center;
               current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();
    
               Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
               current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
               current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;
    
               float across_mul = 0;
               across_mul = current_across_axis_o.dot(across_axis_o);
    
    
              raisim::Vec<3> trans_r;
             raisim::Mat<3,3> rot_r;
             raisim::Vec<4> quat_r;
    
             Eigen::VectorXd obj_state;
             obj_state.setZero(8);
    
             mano_r_->getFramePosition(body_parts_r_[0], trans_r); // Position: body position of a body in mano
             mano_r_->getFrameOrientation(body_parts_r_[0], rot_r); // Body_orientation:
    
             raisim::rotMatToQuat(rot_r, quat_r);
              obj_state = arctic->getGeneralizedCoordinate().e();
    
                global_state_ << obj_pose_wrist.e(),
                                 frame_y_in_obj,
                                 joint_pos_in_obj,
                                 Obj_Position.e(),
                                 current_grasp_axis_o - grasp_axis_o,
                                 across_mul,
                                 trans_r.e(),
                                 quat_r.e(),
                                 gc_r_.tail(gcDim_ - 6),
                                 obj_state;
            }
    
            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }
    
            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }
    
    
            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {
                lift = true;
            }
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
    
                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }
    
                return false;
            }
    
        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            bool lift = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;
    
            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;
    
    
            int num_contacts = 13;
            int num_bodyparts = 17;
    
            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;
    
            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;
    
            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;
    
    
            std::string body_parts_r_[17] =  {"z_rotation_joint",
                                             "joint_1.0", "joint_2.0", "joint_3.0", "joint_3.0_tip",
                                             "joint_5.0", "joint_6.0", "joint_7.0", "joint_7.0_tip",
                                             "joint_9.0", "joint_10.0", "joint_11.0", "joint_11.0_tip",
                                             "joint_13.0", "joint_14.0", "joint_15.0", "joint_15.0_tip"};
    
            std::string contact_bodies_r_[13] =  {"base_link",
                                                  "link_1.0", "link_2.0", "link_3.0",    // right
                                                  "link_5.0", "link_6.0", "link_7.0",
                                                  "link_9.0", "link_10.0", "link_11.0",
                                                  "link_13.0", "link_14.0", "link_15.0"};
    
    
            Eigen::VectorXd mean_pose;
    
            raisim::PolyLine *lines[17];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[17];
            raisim::Visuals *sphere[17];
            raisim::Visuals *joints_sphere[17];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];
    
            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class MANO_TEST : public RaisimGymEnv {
        public:
    
            explicit MANO_TEST(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
    
                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                root_guided = cfg["root_guided"].As<bool>();
    
                lift = false;
    
                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);
    
                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "table", 0.0, 0.0, 0.0, 0.0, 0.0);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);
    
                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/mano_double/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();
    
                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "table", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");
    
                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    
                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;
    
                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);
    
                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);
    
                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);
    
                wrist_mat_r_in_obj_init.setZero();
            //            bodyLinearVel_r_.setZero(); bodyAngularVel_r_.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                //            gravity_comp.setZero(); gravity_mano.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);
    
                contact_body_idx_r_.setZero(num_contacts);
    
                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();
    
                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);
    
                pTarget_clipped_r.setZero(gcDim_);
    
                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);
    
                //            mano_weight = mano_r_->getTotalMass();
                //            gravity_mano = mano_weight * world_->getGravity();
    
                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                for(int i=1; i < 6; i++){
                    finger_weights_contact(3*i) *= 3;
                }
                finger_weights_contact.segment(13,3) *= 2;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;
    
                finger_weights_aff.setOnes(num_bodyparts);
                for(int i=0; i < 5; i++){
                    finger_weights_aff[4 * i+4] *= 4.0;
                }
                finger_weights_aff /= finger_weights_aff.sum();
                finger_weights_aff *= num_bodyparts;
    
                for(int i=0; i< (num_contacts - 3); i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                for(int i=13; i< 16; i++){
                    impulse_high[i] = 0.2;
                    impulse_low[i] = -0.0;
                }
    
    
                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
    
                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));
    
                /// MUST BE DONE FOR ALL ENVIRONMENTS
                obDim_r_ = 178;
                obDim_l_ = 1;
                gsDim_ = 136;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);
    
                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();
    
                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();
    
                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }
    
                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);
    
                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);
    
                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }
    
                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);
    
                    /// Create table
                    //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    //                table_top->setPosition(1.25, 0, 0.475);
                    //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg1->setPosition(0.2625,0.4675,0.2375);
                    //                leg2->setPosition(2.2275,0.4875,0.2375);
                    //                leg3->setPosition(0.2625,-0.4675,0.2375);
                    //                leg4->setPosition(2.2275,-0.4875,0.2375);
    
                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);
    
                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }
    
            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");
    
                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
    
                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }
    
            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
            }
    
            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    lift = false;
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);
    
                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object
    
                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);
    
                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
    
                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }
    
            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            // init_state_l in right-hand coord
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                             const Eigen::Ref<EigenVec>& init_state_l,
                             const Eigen::Ref<EigenVec>& init_vel_r,
                             const Eigen::Ref<EigenVec>& init_vel_l,
                             const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                lift = false;
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);
    
                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);
    
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                /// set initial hand pose (20 DoF) and velocity (20 DoF)
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();
    
                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse
    
                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();
    
                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial object state
    
                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof
    
                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);
    
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                obj_weight = arctic->getTotalMass();
    
                updateObservation();
    
            }
    
            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                           const Eigen::Ref<EigenVec>& obj_center,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                           const Eigen::Ref<EigenVec>& goal_pose_r,
                           const Eigen::Ref<EigenVec>& goal_pose_l,
                           const Eigen::Ref<EigenVec>& goal_qpos_r,
                           const Eigen::Ref<EigenVec>& goal_qpos_l,
                           const Eigen::Ref<EigenVec>& goal_contacts_r,
                           const Eigen::Ref<EigenVec>& goal_contacts_l) final {
    
                           afford_center = target_center.cast<double>();
                           obj_center_o = obj_center.cast<double>();
                           Eigen::Vector3d hand_center_temp;
                           hand_center.setZero();
                           hand_center_temp.setZero();
                           raisim::Vec<3> sphere_pos;
                           for(int i = 0; i < 4 ; i++){
                               mano_r_->getFramePosition(body_parts_r_[i*4+2], sphere_pos);
                               hand_center_temp[0] += sphere_pos.e()[0];
                               hand_center_temp[1] += sphere_pos.e()[1];
                               hand_center_temp[2] += sphere_pos.e()[2];
                           }
                           hand_center_temp /= 4;
                           mano_r_->getFramePosition(body_parts_r_[20], sphere_pos);
                    //                       if (visualizable_){
                    //                           aff_center_visual[4]->setPosition(sphere_pos.e());
                    //                       }
    
                           Eigen::Vector3d finger_center;
                           finger_center.setZero();
                           finger_center[0] = hand_center_temp[0];
                           finger_center[1] = hand_center_temp[1];
                           finger_center[2] = hand_center_temp[2];
    
                           hand_center_temp[0] += sphere_pos.e()[0];
                           hand_center_temp[1] += sphere_pos.e()[1];
                           hand_center_temp[2] += sphere_pos.e()[2];
                           hand_center_temp /= 2;
    
    
                           Eigen::Vector3d across_axis_w;
                           across_axis_w = finger_center - hand_center_temp;
                           across_axis_w = across_axis_w / across_axis_w.norm();
    
                           raisim::Vec<3> wrist_pos_w;
                           raisim::Mat<3,3> wrist_mat_r;
                           mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                           mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                           Eigen::Vector3d bias, wrist_target_temp;
                           auto affordance_id = arctic->getBodyIdx("top");
                           arctic->getPosition(affordance_id, Obj_Position_init);
                           arctic->getOrientation(affordance_id, Obj_orientation_init);
                           raisim::Mat<3,3> Obj_orientation_init_trans;
                           raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                           raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
    
                           bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                           wrist_target_temp = wrist_pos_w.e() + bias;
                           wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());
    
                           hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                           hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                           hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                           hand_center = wrist_mat_r.e().transpose() * hand_center_temp;
    
                           Eigen::Vector3d grasp_axis_w;
                           grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                           grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                           across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                           across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                           }
    
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
    
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                //            Eigen::Vector3d target_center;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);
                if (visualizable_){
                //                wrist_target[1]->setPosition(obj_center_w.e() + obj_pos_w.e());
                }
    
                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }
    
                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];
    
                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }
    
                Eigen::Vector3d wrist_bias_w, wrist_bias;
                if (root_guided){
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                }
                else{
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                }
                wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                if (lift){
                    wrist_bias_w[2] += 0.2;
                }
                wrist_bias = init_or_r_.e() * wrist_bias_w;
                actionMean_r_.head(3) += wrist_bias;

                if (root_guided){
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
    
    
                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)
    
                /// Clip targets to limits
    
                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);
    
                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);
    
                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;
    
                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }
    
                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
    
                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }
    
                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();
    
    
                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::max(0.0, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));
    
                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;
    
                return rewards_sum_;
            }
    
            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                //                if (visualizable_){
                //                    joints_sphere[i]->setPosition(Position_r.e());
                //                }
                }
    
                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();
    
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);
    
                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();
    
                mano_r_->getState(gc_r_, gv_r_);
    
                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);
    
                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame
    
                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();
    
                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }
    
                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }
    
                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }
    
                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }
    
                right_hand_torque = (pTarget_clipped_r - gc_r_);
    
                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;
    
                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
    
                obDouble_r_ << target_center_dif,           // 3, hand center diff
                               gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                               gc_r_.tail(gcDim_ - 6),      // (mirror) 45, generalized coordinate
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;
    
                raisim::transpose(Obj_orientation_temp, Obj_orientation);
    
                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);
    
                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];
    
                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);
    
                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
    
               Eigen::Vector3d current_across_axis_w;
               current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;
    
               Eigen::Vector3d current_grasp_axis_w;
               current_grasp_axis_w = wrist_mat_r.e() * hand_center;
               current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();
    
               Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
               current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
               current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;
    
               float across_mul = 0;
               across_mul = current_across_axis_o.dot(across_axis_o);
    
                global_state_ << obj_pose_wrist.e(),
                                 frame_y_in_obj,
                                 joint_pos_in_obj,
                                 Obj_Position.e(),
                                 current_grasp_axis_o - grasp_axis_o,
                                 across_mul;
            }
    
            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }
    
            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }
    
    
            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {
                lift = true;
            }
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
    
                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }
    
                return false;
            }
    
        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            bool lift = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;
    
            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;
    
    
            int num_contacts = 16;
            int num_bodyparts = 21;
    
            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;
    
            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;
    
            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;
    
            std::string body_parts_r_[21] = {"right_wrist_0rz",
                                             "right_index1_x", "right_index2_x", "right_index3_x", "right_index_tip",
                                             "right_middle1_x", "right_middle2_x", "right_middle3_x", "right_middle_tip",
                                             "right_pinky1_x", "right_pinky2_x", "right_pinky3_x", "right_pinky_tip",
                                             "right_ring1_x", "right_ring2_x", "right_ring3_x", "right_ring_tip",
                                             "right_thumb1_x", "right_thumb2_x", "right_thumb3_x", "right_thumb_tip",
            };
            std::string body_parts_l_[21] = {"left_wrist_0rz",
                                             "left_index1_x", "left_index2_x", "left_index3_x",
                                             "left_middle1_x", "left_middle2_x", "left_middle3_x",
                                             "left_pinky1_x", "left_pinky2_x", "left_pinky3_x",
                                             "left_ring1_x", "left_ring2_x", "left_ring3_x",
                                             "left_thumb1_x", "left_thumb2_x", "left_thumb3_x",
                                             "left_thumb_tip", "left_index_tip", "left_middle_tip", "left_ring_tip", "left_pinky_tip",
            };
    
            std::string contact_bodies_r_[16] = {"right_wrist_rz",
                                                 "right_index1_z",  "right_index2_z",  "right_index3_z",
                                                 "right_middle1_z", "right_middle2_z", "right_middle3_z",
                                                 "right_pinky1_z",  "right_pinky2_z",  "right_pinky3_z",
                                                 "right_ring1_z",   "right_ring2_z",   "right_ring3_z",
                                                 "right_thumb1_z",  "right_thumb2_z",  "right_thumb3_z"
            };
    
            std::string contact_bodies_l_[16] = {"left_wrist_rz",
                                                 "left_index1_z",  "left_index2_z",  "left_index3_z",
                                                 "left_middle1_z", "left_middle2_z", "left_middle3_z",
                                                 "left_pinky1_z",  "left_pinky2_z",  "left_pinky3_z",
                                                 "left_ring1_z",   "left_ring2_z",   "left_ring3_z",
                                                 "left_thumb1_z",  "left_thumb2_z",  "left_thumb3_z"
            };
    
            raisim::PolyLine *lines[21];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[21];
            raisim::Visuals *sphere[21];
            raisim::Visuals *joints_sphere[21];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];
    
            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class INSPIRE_FLOATING : public RaisimGymEnv {

        public:

            explicit INSPIRE_FLOATING(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                root_guided = cfg["root_guided"].As<bool>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/inspire_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();


                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);


                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                finger_weights_contact(3) *= 3;  // thumb_distal
                finger_weights_contact(5) *= 3;  // index_intermediate
                finger_weights_contact(7) *= 3;  // middle_intermediate
                finger_weights_contact(9) *= 3; // ring_intermediate
                finger_weights_contact(11) *= 3; // pinky_intermediate
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                for(int i=0; i<num_contacts; i++){ // The last 3 contacts (thumb links) likely have different impulse limits.
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                // Set higher impulse limits for fingertips
                impulse_high[3] = 0.2;  // thumb_distal
                impulse_high[5] = 0.2;  // index_intermediate
                impulse_high[7] = 0.2;  // middle_intermediate
                impulse_high[9] = 0.2; // ring_intermediate
                impulse_high[11] = 0.2; // pinky_intermediate

                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                obDim_r_ = 96;
                obDim_l_ = 1;
                gsDim_ = 112;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                    //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    //                table_top->setPosition(1.25, 0, 0.475);
                    //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg1->setPosition(0.2625,0.4675,0.2375);
                    //                leg2->setPosition(2.2275,0.4875,0.2375);
                    //                leg3->setPosition(0.2625,-0.4675,0.2375);
                    //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();

                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);

                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);

                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>();
                gv_set_r_ = init_vel_r.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                obj_weight = arctic->getTotalMass();

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        Eigen::Vector3d hand_center_temp_fingers_only; // metadata
                        hand_center_temp_fingers_only.setZero();
                        raisim::Vec<3> sphere_pos;

                        // Repository metadata.
                        mano_r_->getFramePosition(body_parts_r_[6], sphere_pos); // index_intermediate_joint
                        hand_center_temp_fingers_only += sphere_pos.e();
                        mano_r_->getFramePosition(body_parts_r_[9], sphere_pos); // middle_intermediate_joint
                        hand_center_temp_fingers_only += sphere_pos.e();
                        mano_r_->getFramePosition(body_parts_r_[12], sphere_pos); // ring_intermediate_joint
                        hand_center_temp_fingers_only += sphere_pos.e();
                        mano_r_->getFramePosition(body_parts_r_[15], sphere_pos); // pinky_intermediate_joint
                        hand_center_temp_fingers_only += sphere_pos.e();
                        hand_center_temp_fingers_only /= 4; // metadata

                        // Repository metadata.
                        mano_r_->getFramePosition(body_parts_r_[2], sphere_pos); // thumb_tip_joint
                        hand_center_temp = hand_center_temp_fingers_only; // metadata hand_center_temp
                        hand_center_temp += sphere_pos.e();
                        hand_center_temp /= 2; // metadata

                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];
                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }

                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }


                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();


                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);


                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame


                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
            Eigen::Vector3d current_across_axis_w;

                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);

                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
                raisim::Vec<3> obj_current_pos;
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(top_id, obj_current_pos); //Obj_Position: object base position at this time

                double displacement = (obj_current_pos.e() - obj_pos_init_.head(3)).squaredNorm();
                if (displacement > 0.1){
                    terminalReward = -10;
                    return true;
                }

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 12;
            int num_bodyparts = 17;

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;


            std::string body_parts_r_[17] =  {"z_rotation_joint",
            "thumb_proximal_pitch_joint", "thumb_intermediate_joint", "thumb_distal_joint", "thumb_tip_joint",
            "index_proximal_joint", "index_intermediate_joint", "index_tip_joint",
            "middle_proximal_joint", "middle_intermediate_joint", "middle_tip_joint",
            "ring_proximal_joint", "ring_intermediate_joint", "ring_tip_joint",
            "pinky_proximal_joint", "pinky_intermediate_joint", "pinky_tip_joint"};

            std::string contact_bodies_r_[12] =  {
                "hand_base_link",
                "thumb_proximal","thumb_intermediate", "thumb_distal",
                "index_proximal", "index_intermediate",
                "middle_proximal", "middle_intermediate",
                "ring_proximal", "ring_intermediate",
                "pinky_proximal", "pinky_intermediate",
            };

            Eigen::VectorXd mean_pose;

            raisim::PolyLine *lines[17];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[17];
            raisim::Visuals *sphere[17];
            raisim::Visuals *joints_sphere[17];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class INSPIRE_TEST : public RaisimGymEnv {

        public:
    
            explicit INSPIRE_TEST(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
    
                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                root_guided = cfg["root_guided"].As<bool>();
                lift = false;
    
                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);
    
                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);
    
                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/inspire_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();
    
                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");
    
                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    
                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;
    
                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);
    
                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);
    
                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);
    
    
                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);
    
                contact_body_idx_r_.setZero(num_contacts);
    
                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();
    
                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);
    
                pTarget_clipped_r.setZero(gcDim_);
    
                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);
    
                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                // Give higher weight to fingertips (thumb_distal, index_intermediate, middle_intermediate, ring_intermediate, pinky_intermediate)
                finger_weights_contact(3) *= 3;  // thumb_distal
                finger_weights_contact(5) *= 3;  // index_intermediate
                finger_weights_contact(7) *= 3;  // middle_intermediate
                finger_weights_contact(9) *= 3; // ring_intermediate
                finger_weights_contact(11) *= 3; // pinky_intermediate
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;
    
                for(int i=0; i<num_contacts; i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                // Set higher impulse limits for fingertips
                impulse_high[3] = 0.2;  // thumb_distal
                impulse_high[5] = 0.2;  // index_intermediate
                impulse_high[7] = 0.2;  // middle_intermediate
                impulse_high[9] = 0.2; // ring_intermediate
                impulse_high[11] = 0.2; // pinky_intermediate
    
                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
    
                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));
    
                /// MUST BE DONE FOR ALL ENVIRONMENTS
                obDim_r_ = 96;
                obDim_l_ = 1;
                gsDim_ = 139;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);
    
                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();
    
                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();
    
                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }
    
                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);
    
                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);
    
                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }
    
                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);
    
                    /// Create table
                    //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    //                table_top->setPosition(1.25, 0, 0.475);
                    //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg1->setPosition(0.2625,0.4675,0.2375);
                    //                leg2->setPosition(2.2275,0.4875,0.2375);
                    //                leg3->setPosition(0.2625,-0.4675,0.2375);
                    //                leg4->setPosition(2.2275,-0.4875,0.2375);
    
                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);
    
                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }
    
            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");
                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
    
                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }
    
            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}
    
            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    lift = false;
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);
    
                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object
    
                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);
    
                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
    
                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }
    
            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                             const Eigen::Ref<EigenVec>& init_state_l,
                             const Eigen::Ref<EigenVec>& init_vel_r,
                             const Eigen::Ref<EigenVec>& init_vel_l,
                             const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);
    
                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                lift = false;
    
                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);
    
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();
    
                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse
    
                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();
    
                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial object state
    
                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof
    
                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);
    
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                obj_weight = arctic->getTotalMass();
    
                updateObservation();
    
            }
    
            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                           const Eigen::Ref<EigenVec>& obj_center,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                           const Eigen::Ref<EigenVec>& goal_pose_r,
                           const Eigen::Ref<EigenVec>& goal_pose_l,
                           const Eigen::Ref<EigenVec>& goal_qpos_r,
                           const Eigen::Ref<EigenVec>& goal_qpos_l,
                           const Eigen::Ref<EigenVec>& goal_contacts_r,
                           const Eigen::Ref<EigenVec>& goal_contacts_l) final {
    
                           afford_center = target_center.cast<double>();
                           obj_center_o = obj_center.cast<double>();
                           Eigen::Vector3d hand_center_temp;
                           hand_center.setZero();
                           hand_center_temp.setZero();
                           Eigen::Vector3d hand_center_temp_fingers_only; // metadata
                           hand_center_temp_fingers_only.setZero();
                           raisim::Vec<3> sphere_pos;
    
                           // Repository metadata.
                           mano_r_->getFramePosition(body_parts_r_[6], sphere_pos); // index_intermediate_joint
                           hand_center_temp_fingers_only += sphere_pos.e();
                           mano_r_->getFramePosition(body_parts_r_[9], sphere_pos); // middle_intermediate_joint
                           hand_center_temp_fingers_only += sphere_pos.e();
                           mano_r_->getFramePosition(body_parts_r_[12], sphere_pos); // ring_intermediate_joint
                           hand_center_temp_fingers_only += sphere_pos.e();
                           mano_r_->getFramePosition(body_parts_r_[15], sphere_pos); // pinky_intermediate_joint
                           hand_center_temp_fingers_only += sphere_pos.e();
                           hand_center_temp_fingers_only /= 4; // metadata
    
                           // Repository metadata.
                           mano_r_->getFramePosition(body_parts_r_[2], sphere_pos); // thumb_tip_joint
                           hand_center_temp = hand_center_temp_fingers_only; // metadata hand_center_temp
                           hand_center_temp += sphere_pos.e();
                           hand_center_temp /= 2; // metadata
    
                           Eigen::Vector3d finger_center;
                           finger_center.setZero();
                           finger_center[0] = hand_center_temp[0];
                           finger_center[1] = hand_center_temp[1];
                           finger_center[2] = hand_center_temp[2];
    
                           hand_center_temp[0] += sphere_pos.e()[0];
                           hand_center_temp[1] += sphere_pos.e()[1];
                           hand_center_temp[2] += sphere_pos.e()[2];
                           hand_center_temp /= 2;
    
                           Eigen::Vector3d across_axis_w;
                           across_axis_w = finger_center - hand_center_temp;
                           across_axis_w = across_axis_w / across_axis_w.norm();
    
                           raisim::Vec<3> wrist_pos_w;
                           raisim::Mat<3,3> wrist_mat_r;
                           mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                           mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                           Eigen::Vector3d bias, wrist_target_temp;
                           auto affordance_id = arctic->getBodyIdx("top");
                           arctic->getPosition(affordance_id, Obj_Position_init);
                           arctic->getOrientation(affordance_id, Obj_orientation_init);
                           raisim::Mat<3,3> Obj_orientation_init_trans;
                           raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                           raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                           bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                           wrist_target_temp = wrist_pos_w.e() + bias;
                           wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());
    
                           hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                           hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                           hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                           hand_center = wrist_mat_r.e().transpose() * hand_center_temp;
    
                           Eigen::Vector3d grasp_axis_w;
                           grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                           grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                           across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                           across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                           }
    
            
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
    
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);
    
                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
    
                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];
    
                Eigen::Vector3d wrist_bias_w, wrist_bias;
                if (root_guided){
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                }
                else{
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                }
                wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                if (lift){
                    wrist_bias_w[2] += 0.2;
                }

                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }
                wrist_bias = init_or_r_.e() * wrist_bias_w;
                actionMean_r_.head(3) += wrist_bias;

                if (root_guided){
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
    
                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)
    
                /// Clip targets to limits
    
                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);
    
                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);
    
                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;
    
                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }
    
                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
    
                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }
    
                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();
    
    
                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::max(0.0, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));
    
                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;
    
                return rewards_sum_;
            }
    
            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }
    
                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();
    
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);
    
                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();
    
                mano_r_->getState(gc_r_, gv_r_);
    
                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);
    
    
                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame
    
                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();
    
                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }
    
                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }
    
                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }
    
                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }
    
                right_hand_torque = (pTarget_clipped_r - gc_r_);
    
                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;
    
                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
    
                obDouble_r_ << target_center_dif,           // 3, hand center diff
                               gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                               gc_r_.tail(gcDim_ - 6),
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;
    
                raisim::transpose(Obj_orientation_temp, Obj_orientation);
    
                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);
    
                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];
    
                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);
    
                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
               Eigen::Vector3d current_across_axis_w;
    
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;
    
               Eigen::Vector3d current_grasp_axis_w;
               current_grasp_axis_w = wrist_mat_r.e() * hand_center;
               current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();
    
               Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
               current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
               current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;
    
               float across_mul = 0;
               across_mul = current_across_axis_o.dot(across_axis_o);
    
    
              raisim::Vec<3> trans_r;
             raisim::Mat<3,3> rot_r;
             raisim::Vec<4> quat_r;
    
             Eigen::VectorXd obj_state;
             obj_state.setZero(8);
    
             mano_r_->getFramePosition(body_parts_r_[0], trans_r); // Position: body position of a body in mano
             mano_r_->getFrameOrientation(body_parts_r_[0], rot_r); // Body_orientation:
    
             raisim::rotMatToQuat(rot_r, quat_r);
              obj_state = arctic->getGeneralizedCoordinate().e();
    
                global_state_ << obj_pose_wrist.e(),//3
                                 frame_y_in_obj, // 17*3
                                 joint_pos_in_obj, //17*3
                                 Obj_Position.e(), //3
                                 current_grasp_axis_o - grasp_axis_o, //3
                                 across_mul, //1
                                 trans_r.e(), //3
                                 quat_r.e(), //4
                                 gc_r_.tail(gcDim_ - 6), //12
                                 obj_state; //8
            }
    
            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }
    
            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }
    
    
            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {
                lift = true;
            }
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
    
                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }
    
                return false;
            }
    
        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            bool lift = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;
    
            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;
    
    
            int num_contacts = 12;
            int num_bodyparts = 17;
    
            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;
    
            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;
    
            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;
    
    
            std::string body_parts_r_[17] =  {"z_rotation_joint",
                "thumb_proximal_pitch_joint", "thumb_intermediate_joint", "thumb_distal_joint", "thumb_tip_joint",
               "index_proximal_joint", "index_intermediate_joint", "index_tip_joint",
               "middle_proximal_joint", "middle_intermediate_joint", "middle_tip_joint",
               "ring_proximal_joint", "ring_intermediate_joint", "ring_tip_joint",
               "pinky_proximal_joint", "pinky_intermediate_joint", "pinky_tip_joint"};
    
            std::string contact_bodies_r_[12] =  {
                "hand_base_link",
                "thumb_proximal","thumb_intermediate", "thumb_distal",
                "index_proximal", "index_intermediate",
                "middle_proximal", "middle_intermediate",
                "ring_proximal", "ring_intermediate",
                "pinky_proximal", "pinky_intermediate",
            };
    
    
    
            Eigen::VectorXd mean_pose;
    
            raisim::PolyLine *lines[17];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[17];
            raisim::Visuals *sphere[17];
            raisim::Visuals *joints_sphere[17];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];
    
            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class ABILITY_FLOATING : public RaisimGymEnv {

        public:

            explicit ABILITY_FLOATING(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                root_guided = cfg["root_guided"].As<bool>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/ability_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();


                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);


                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts); // modified
                // Give higher weight to fingertips (thumb_L2, index_L2, middle_L2, ring_L2, pinky_L2)
                finger_weights_contact(2) *= 3;  // thumb_L2
                finger_weights_contact(4) *= 3;  // index_L2
                finger_weights_contact(6) *= 3;  // middle_L2
                finger_weights_contact(8) *= 3; // ring_L2
                finger_weights_contact(10) *= 3; // pinky_L2
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                for(int i=0; i<num_contacts; i++){ // modified
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                // Set higher impulse limits for fingertips
                impulse_high[2] = 0.2;  // thumb_L2
                impulse_high[4] = 0.2;  // index_L2
                impulse_high[6] = 0.2;  // middle_L2
                impulse_high[8] = 0.2; // ring_L2
                impulse_high[10] = 0.2; // pinky_L2


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                obDim_r_ = 88; // corrected to 88 based on re-calculation and gcDim_ = 16
                obDim_l_ = 1;
                gsDim_ = 106; // modified: Adjusted based on global_state_ composition (3+48+48+3+3+1 = 106)
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                    //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    //                table_top->setPosition(1.25, 0, 0.475);
                    //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg1->setPosition(0.2625,0.4675,0.2375);
                    //                leg2->setPosition(2.2275,0.4875,0.2375);
                    //                leg3->setPosition(0.2625,-0.4675,0.2375);
                    //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();

                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);

                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);

                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>();
                gv_set_r_ = init_vel_r.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                obj_weight = arctic->getTotalMass();

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;
                        // modified: calculate initial hand_center_temp by averaging the second (middle) segments of index, middle, ring, and pinky fingers
                        for(int i = 0; i < 4 ; i++){ // modified: loop for 4 fingers (index, middle, ring, pinky)
                            mano_r_->getFramePosition(body_parts_r_[4 + i*3 + 1], sphere_pos); // modified: Indices 5, 8, 11, 14 for _q2 joints
                            hand_center_temp[0] += sphere_pos.e()[0];
                            hand_center_temp[1] += sphere_pos.e()[1];
                            hand_center_temp[2] += sphere_pos.e()[2];
                        }
                        hand_center_temp /= 4; // modified: Averaged over 4 points
                        // modified: further refine with thumb tip position
                        mano_r_->getFramePosition(body_parts_r_[3], sphere_pos); // modified: Index 3 for "thumb_tip_joint"
                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }
                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }


                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();


                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);


                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame


                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
            Eigen::Vector3d current_across_axis_w;

                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);

                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
                raisim::Vec<3> obj_current_pos;
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(top_id, obj_current_pos); //Obj_Position: object base position at this time

                double displacement = (obj_current_pos.e() - obj_pos_init_.head(3)).squaredNorm();
                if (displacement > 0.1){
                    terminalReward = -10;
                    return true;
                }

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 11; // modified
            int num_bodyparts = 16; // modified

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;


            std::string body_parts_r_[16] =  { // modified
                                    "z_rotation_joint",    // metadata
                                    "thumb_q1", "thumb_q2", "thumb_tip_joint", // metadata (metadata，metadata)
                                    "index_q1", "index_q2", "index_tip_joint", // metadata
                                    "middle_q1", "middle_q2", "middle_tip_joint", // metadata
                                    "ring_q1", "ring_q2", "ring_tip_joint",  // metadata
                                    "pinky_q1", "pinky_q2", "pinky_tip_joint"}; // metadata

            std::string contact_bodies_r_[11] =  { // modified
                                "base", // modified: Palm link name
                                "thumb_L1", "thumb_L2", // modified
                                "index_L1", "index_L2", // modified
                                "middle_L1", "middle_L2", // modified
                                "ring_L1", "ring_L2", // modified
                                "pinky_L1", "pinky_L2" // modified
            };

            raisim::PolyLine *lines[17];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[16]; // modified
            raisim::Visuals *sphere[16]; // modified
            raisim::Visuals *joints_sphere[16]; // modified
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class ABILITY_TEST : public RaisimGymEnv {

        public:
    
            explicit ABILITY_TEST(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
    
                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                root_guided = cfg["root_guided"].As<bool>();
                lift = false;
    
                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);
    
                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);
    
                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/ability_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();
    
                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");
    
                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    
                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;
    
                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);
    
                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);
    
                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);
    
    
                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);
    
                contact_body_idx_r_.setZero(num_contacts);
    
                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();
    
                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);
    
                pTarget_clipped_r.setZero(gcDim_);
    
                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);
    
                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts); // modified
                // Give higher weight to fingertips (thumb_L2, index_L2, middle_L2, ring_L2, pinky_L2)
                finger_weights_contact(2) *= 3;  // thumb_L2
                finger_weights_contact(4) *= 3;  // index_L2
                finger_weights_contact(6) *= 3;  // middle_L2
                finger_weights_contact(8) *= 3; // ring_L2
                finger_weights_contact(10) *= 3; // pinky_L2
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;
    
                for(int i=0; i<num_contacts; i++){ // modified
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                // Set higher impulse limits for fingertips
                impulse_high[2] = 0.2;  // thumb_L2
                impulse_high[4] = 0.2;  // index_L2
                impulse_high[6] = 0.2;  // middle_L2
                impulse_high[8] = 0.2; // ring_L2
                impulse_high[10] = 0.2; // pinky_L2
    
    
                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
    
                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));
    
                /// MUST BE DONE FOR ALL ENVIRONMENTS
                obDim_r_ = 88; // corrected to 88 based on re-calculation and gcDim_ = 16
                obDim_l_ = 1;
                gsDim_ = 131; // modified: Adjusted based on global_state_ composition (3+48+48+3+3+1+3+4+(16-6)+8 = 131)
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);
    
                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();
    
                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();
    
                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }
    
                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);
    
                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);
    
                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }
    
                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);
    
                    /// Create table
                        //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                        //                table_top->setPosition(1.25, 0, 0.475);
                        //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                        //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                        //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                        //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                        //                leg1->setPosition(0.2625,0.4675,0.2375);
                        //                leg2->setPosition(2.2275,0.4875,0.2375);
                        //                leg3->setPosition(0.2625,-0.4675,0.2375);
                        //                leg4->setPosition(2.2275,-0.4875,0.2375);
    
                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);
    
                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }
    
            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");
                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
    
                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }
    
            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}
    
            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    lift = false;
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);
    
                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object
    
                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);
    
                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
    
                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }
    
            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                             const Eigen::Ref<EigenVec>& init_state_l,
                             const Eigen::Ref<EigenVec>& init_vel_r,
                             const Eigen::Ref<EigenVec>& init_vel_l,
                             const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);
    
                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                lift = false;
    
                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);
    
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();
    
                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse
    
                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();
    
                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial object state
    
                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof
    
                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);
    
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                obj_weight = arctic->getTotalMass();
    
                updateObservation();
    
            }
    
            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                           const Eigen::Ref<EigenVec>& obj_center,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                           const Eigen::Ref<EigenVec>& goal_pose_r,
                           const Eigen::Ref<EigenVec>& goal_pose_l,
                           const Eigen::Ref<EigenVec>& goal_qpos_r,
                           const Eigen::Ref<EigenVec>& goal_qpos_l,
                           const Eigen::Ref<EigenVec>& goal_contacts_r,
                           const Eigen::Ref<EigenVec>& goal_contacts_l) final {
    
                           afford_center = target_center.cast<double>();
                           obj_center_o = obj_center.cast<double>();
                           Eigen::Vector3d hand_center_temp;
                           hand_center.setZero();
                           hand_center_temp.setZero();
                           raisim::Vec<3> sphere_pos;
                           // modified: calculate initial hand_center_temp by averaging the second (middle) segments of index, middle, ring, and pinky fingers
                           for(int i = 0; i < 4 ; i++){ // modified: loop for 4 fingers (index, middle, ring, pinky)
                               mano_r_->getFramePosition(body_parts_r_[4 + i*3 + 1], sphere_pos); // modified: Indices 5, 8, 11, 14 for _q2 joints
                               hand_center_temp[0] += sphere_pos.e()[0];
                               hand_center_temp[1] += sphere_pos.e()[1];
                               hand_center_temp[2] += sphere_pos.e()[2];
                           }
                           hand_center_temp /= 4; // modified: Averaged over 4 points
                           // modified: further refine with thumb tip position
                           mano_r_->getFramePosition(body_parts_r_[3], sphere_pos); // modified: Index 3 for "thumb_tip_joint"
                           Eigen::Vector3d finger_center;
                           finger_center.setZero();
                           finger_center[0] = hand_center_temp[0];
                           finger_center[1] = hand_center_temp[1];
                           finger_center[2] = hand_center_temp[2];
    
                           hand_center_temp[0] += sphere_pos.e()[0];
                           hand_center_temp[1] += sphere_pos.e()[1];
                           hand_center_temp[2] += sphere_pos.e()[2];
                           hand_center_temp /= 2;
    
                           Eigen::Vector3d across_axis_w;
                           across_axis_w = finger_center - hand_center_temp;
                           across_axis_w = across_axis_w / across_axis_w.norm();
    
                           raisim::Vec<3> wrist_pos_w;
                           raisim::Mat<3,3> wrist_mat_r;
                           mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                           mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                           Eigen::Vector3d bias, wrist_target_temp;
                           auto affordance_id = arctic->getBodyIdx("top");
                           arctic->getPosition(affordance_id, Obj_Position_init);
                           arctic->getOrientation(affordance_id, Obj_orientation_init);
                           raisim::Mat<3,3> Obj_orientation_init_trans;
                           raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                           raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                           bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                           wrist_target_temp = wrist_pos_w.e() + bias;
                           wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());
    
                           hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                           hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                           hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                           hand_center = wrist_mat_r.e().transpose() * hand_center_temp;
    
                           Eigen::Vector3d grasp_axis_w;
                           grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                           grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                           across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                           across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                           }
    
            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
    
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);
    
                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
    
                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];
    
                Eigen::Vector3d wrist_bias_w, wrist_bias;
                if (root_guided){
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                }
                else{
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                }
                wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                if (lift){
                    wrist_bias_w[2] += 0.2;
                }
                wrist_bias = init_or_r_.e() * wrist_bias_w;
                actionMean_r_.head(3) += wrist_bias;
                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }
    
                if (root_guided){
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
    
                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)
    
                /// Clip targets to limits
    
                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);
    
                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);
    
                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;
    
                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }
    
                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
    
                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }
    
                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();
    
    
                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::max(0.0, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));
    
                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;
    
                return rewards_sum_;
            }
    
            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }
    
                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();
    
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);
    
                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();
    
                mano_r_->getState(gc_r_, gv_r_);
    
                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);
    
    
                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame
    
                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();
    
                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }
    
                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }
    
                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }
    
                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }
    
                right_hand_torque = (pTarget_clipped_r - gc_r_);
    
                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;
    
                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                obDouble_r_ << target_center_dif,  //3
                               gc_r_.segment(3, 3) - wrist_target_euler.e(), //3
                               gc_r_.tail(gcDim_ - 6), //10 (gcDim_ = 16)
                                right_hand_torque, //16 (gcDim_ = 16)
                                wrist_vel_in_wrist, //3
                                wrist_qvel_in_wrist, //3
                                contacts_r_af, //11
                                impulses_r_af, //11
                                obj_vel_in_wrist, //3
                                obj_qvel_in_wrist, //3
                                contacts_r_non_af, //11
                                impulses_r_non_af; //11
    
                raisim::transpose(Obj_orientation_temp, Obj_orientation);
    
                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);
    
                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];
    
                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);
    
                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
               Eigen::Vector3d current_across_axis_w;
    
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;
    
               Eigen::Vector3d current_grasp_axis_w;
               current_grasp_axis_w = wrist_mat_r.e() * hand_center;
               current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();
    
               Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
               current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
               current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;
    
               float across_mul = 0;
               across_mul = current_across_axis_o.dot(across_axis_o);
    
    
              raisim::Vec<3> trans_r;
             raisim::Mat<3,3> rot_r;
             raisim::Vec<4> quat_r;
    
             Eigen::VectorXd obj_state;
             obj_state.setZero(8);
    
             mano_r_->getFramePosition(body_parts_r_[0], trans_r); // Position: body position of a body in mano
             mano_r_->getFrameOrientation(body_parts_r_[0], rot_r); // Body_orientation:
    
             raisim::rotMatToQuat(rot_r, quat_r);
              obj_state = arctic->getGeneralizedCoordinate().e();
    
                global_state_ << obj_pose_wrist.e(),
                                 frame_y_in_obj,
                                 joint_pos_in_obj,
                                 Obj_Position.e(),
                                 current_grasp_axis_o - grasp_axis_o,
                                 across_mul,
                                 trans_r.e(),
                                 quat_r.e(),
                                 gc_r_.tail(gcDim_ - 6),
                                 obj_state;
            }
    
            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }
    
            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }
    
    
            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {
                lift = true;
            }
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
    
                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }
    
                return false;
            }
    
        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            bool lift = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;
    
            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;
    
    
            int num_contacts = 11; // modified
            int num_bodyparts = 16; // modified
    
            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;
    
            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;
    
            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;
    
    
            std::string body_parts_r_[16] =  { // modified
                                     "z_rotation_joint",    // metadata
                                     "thumb_q1", "thumb_q2", "thumb_tip_joint", // metadata (metadata，metadata)
                                     "index_q1", "index_q2", "index_tip_joint", // metadata
                                     "middle_q1", "middle_q2", "middle_tip_joint", // metadata
                                     "ring_q1", "ring_q2", "ring_tip_joint",  // metadata
                                     "pinky_q1", "pinky_q2", "pinky_tip_joint"}; // metadata
    
            std::string contact_bodies_r_[11] =  { // modified
                                "base", // modified: Palm link name
                                "thumb_L1", "thumb_L2", // modified
                                "index_L1", "index_L2", // modified
                                "middle_L1", "middle_L2",
                                "ring_L1", "ring_L2",
                                "pinky_L1", "pinky_L2"};
    
    
            Eigen::VectorXd mean_pose;
    
            raisim::PolyLine *lines[17];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[16]; // modified
            raisim::Visuals *sphere[16]; // modified
            raisim::Visuals *joints_sphere[16]; // modified
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];
    
            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class SCHUNK_FLOATING : public RaisimGymEnv {

        public:
    
            explicit SCHUNK_FLOATING(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
    
                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                root_guided = cfg["root_guided"].As<bool>();

    
                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);
    
                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);
    
                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
    
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/schunk_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63)); // modified for schunk hand
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();
    
    
                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");
    
                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    
                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;
    
                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);
    
                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);
    
                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);
    
    
                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);
    
                contact_body_idx_r_.setZero(num_contacts);
    
                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();
    
                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);
    
                pTarget_clipped_r.setZero(gcDim_);
    
                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);
    
                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                // For Schunk hand, increase weight for fingertip links
                finger_weights_contact(3) *= 3; // Thumb tip link (right_hand_c)
                finger_weights_contact(7) *= 3; // Ring tip link (right_hand_r)
                finger_weights_contact(10) *= 3; // Pinky tip link (right_hand_q)
                finger_weights_contact(13) *= 3; // Index tip link (right_hand_t)
                finger_weights_contact(16) *= 3; // Middle tip link (right_hand_s)
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;
    
    
                for(int i=0; i< num_contacts; i++){
                    if (i >= 1 && i <= 3) { // Thumb links (index 1 to 3) have higher impulse limits
                        impulse_high[i] = 0.2;
                    } else {
                        impulse_high[i] = 0.1;
                    }
                    impulse_low[i] = -0.0;
                }
    
    
                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
    
                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));
    
                obDim_r_ = 132;
                obDim_l_ = 1;
                gsDim_ = 142;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);
    
                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();
    
                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();
    
                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }
    
                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);
    
                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);
    
                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }
    
                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081); // modified: fixed has 8081, floating had no port, so kept 8081
    
                    /// Create table
            //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
            //                table_top->setPosition(1.25, 0, 0.475);
            //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg1->setPosition(0.2625,0.4675,0.2375);
            //                leg2->setPosition(2.2275,0.4875,0.2375);
            //                leg3->setPosition(0.2625,-0.4675,0.2375);
            //                leg4->setPosition(2.2275,-0.4875,0.2375);
    
                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);
    
                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }
    
            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
    
                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }
    
            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}
    
            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);
    
                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object
    
                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);
    
                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
    
                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }
    
            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                             const Eigen::Ref<EigenVec>& init_state_l,
                             const Eigen::Ref<EigenVec>& init_vel_r,
                             const Eigen::Ref<EigenVec>& init_vel_l,
                             const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);
    
                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);
    
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>();
                gv_set_r_ = init_vel_r.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();
    
                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse
    
                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();
    
                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial object state
    
                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof
    
                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);
    
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                obj_weight = arctic->getTotalMass();
    
                updateObservation();
    
            }
    
            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                           const Eigen::Ref<EigenVec>& obj_center,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                           const Eigen::Ref<EigenVec>& goal_pose_r,
                           const Eigen::Ref<EigenVec>& goal_pose_l,
                           const Eigen::Ref<EigenVec>& goal_qpos_r,
                           const Eigen::Ref<EigenVec>& goal_qpos_l,
                           const Eigen::Ref<EigenVec>& goal_contacts_r,
                           const Eigen::Ref<EigenVec>& goal_contacts_l) final {
    
                           afford_center = target_center.cast<double>();
                           obj_center_o = obj_center.cast<double>();
                           Eigen::Vector3d hand_center_temp;
                           hand_center.setZero();
                           hand_center_temp.setZero();
                           raisim::Vec<3> sphere_pos;
                           Eigen::Vector3d hand_center_temp_fingers; // modified for schunk hand
                           hand_center_temp_fingers.setZero(); // modified for schunk hand
                           int finger_count = 0; // modified for schunk hand
    
                          // Repository metadata.
                          // Index finger middle joint: body_parts_r_[15]
                          mano_r_->getFramePosition(body_parts_r_[15], sphere_pos);
                          hand_center_temp_fingers += sphere_pos.e();
                          finger_count++;
    
                          // Middle finger middle joint: body_parts_r_[19]
                          mano_r_->getFramePosition(body_parts_r_[19], sphere_pos);
                          hand_center_temp_fingers += sphere_pos.e();
                          finger_count++;
    
                          // Ring finger middle joint: body_parts_r_[7]
                          mano_r_->getFramePosition(body_parts_r_[7], sphere_pos);
                          hand_center_temp_fingers += sphere_pos.e();
                          finger_count++;
    
                          // Pinky finger middle joint: body_parts_r_[11]
                          mano_r_->getFramePosition(body_parts_r_[11], sphere_pos);
                          hand_center_temp_fingers += sphere_pos.e();
                          finger_count++;
    
                          hand_center_temp_fingers /= finger_count; // metadata
    
                          // Repository metadata.
                          raisim::Vec<3> thumb_tip_pos;
                          mano_r_->getFramePosition(body_parts_r_[4], thumb_tip_pos); // Thumb tip: body_parts_r_[4]
    
                          Eigen::Vector3d finger_center;
                          finger_center = hand_center_temp_fingers; // metadata finger_center
    
                          hand_center_temp = (hand_center_temp_fingers + thumb_tip_pos.e()) / 2.0; // metadata
    
                           Eigen::Vector3d across_axis_w;
                           across_axis_w = finger_center - hand_center_temp;
                           across_axis_w = across_axis_w / across_axis_w.norm();
    
                           raisim::Vec<3> wrist_pos_w;
                           raisim::Mat<3,3> wrist_mat_r;
                           mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                           mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                           Eigen::Vector3d bias, wrist_target_temp;
                           auto affordance_id = arctic->getBodyIdx("top");
                           arctic->getPosition(affordance_id, Obj_Position_init);
                           arctic->getOrientation(affordance_id, Obj_orientation_init);
                           raisim::Mat<3,3> Obj_orientation_init_trans;
                           raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                           raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                           bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                           wrist_target_temp = wrist_pos_w.e() + bias;
                           wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());
    
                           hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                           hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                           hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                           hand_center = wrist_mat_r.e().transpose() * hand_center_temp;
    
                           Eigen::Vector3d grasp_axis_w;
                           grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                           grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                           across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                           across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                           }
    
            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
    
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);
    
                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }
    
                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];
    
                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }
    
                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }
                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)
    
                /// Clip targets to limits
    
                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);
    
                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);
    
                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;
    
                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }
    
                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
    
                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }
    
                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();
    
    
                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));
    
                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;
    
                return rewards_sum_;
            }
    
            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }
    
                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();
    
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);
    
                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();
    
                mano_r_->getState(gc_r_, gv_r_);
    
                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);
    
    
                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame
    
    
                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();
    
                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }
    
                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }
    
                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }
    
                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }
    
                right_hand_torque = (pTarget_clipped_r - gc_r_);
    
                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;
    
                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                obDouble_r_ << target_center_dif,           // 3, hand center diff
                               gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                               gc_r_.tail(gcDim_ - 6),
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;
    
                raisim::transpose(Obj_orientation_temp, Obj_orientation);
    
                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);
    
                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];
    
                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);
    
                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
               Eigen::Vector3d current_across_axis_w;
    
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;
    
               Eigen::Vector3d current_grasp_axis_w;
               current_grasp_axis_w = wrist_mat_r.e() * hand_center;
               current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();
    
               Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
               current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
               current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;
    
               float across_mul = 0;
               across_mul = current_across_axis_o.dot(across_axis_o);
    
                global_state_ << obj_pose_wrist.e(),
                                 frame_y_in_obj,
                                 joint_pos_in_obj,
                                 Obj_Position.e(),
                                 current_grasp_axis_o - grasp_axis_o,
                                 across_mul;
            }
    
            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }
    
            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }
    
    
            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
                raisim::Vec<3> obj_current_pos;
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(top_id, obj_current_pos); //Obj_Position: object base position at this time
    
                double displacement = (obj_current_pos.e() - obj_pos_init_.head(3)).squaredNorm();
                if (displacement > 0.1){
                    terminalReward = -10;
                    return true;
                }
    
                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }
    
                return false;
            }
    
        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = false;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;
    
            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;
    
    
            int num_contacts = 17; // modified for schunk hand
            int num_bodyparts = 22; // modified for schunk hand
    
            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;
    
            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;
    
            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;
            // Repository metadata.
            std::string body_parts_r_[22] =  { // modified for schunk hand
                                     "z_rotation_joint",  //palm
                                     "right_hand_Thumb_Flexion", "right_hand_j3", "right_hand_j4", "thtip_joint",  //thumb
                                      "right_hand_j5",         //additional point
                                     "right_hand_Ring_Finger", "right_hand_j12", "right_hand_j16", "rftip_joint",  //Ring
                                     "right_hand_Pinky", "right_hand_j13", "right_hand_j17", "lftip_joint",     //pinky
                                     "right_hand_Index_Finger_Proximal","right_hand_Index_Finger_Distal", "right_hand_j14", "fftip_joint", // Index
                                     "right_hand_Middle_Finger_Proximal","right_hand_Middle_Finger_Distal", "right_hand_j15", "mftip_joint", // middle
                                     };
    
            std::string contact_bodies_r_[17] =  { // modified for schunk hand
                                                  "right_hand_base_link",       //palm
                                                  "right_hand_a", "right_hand_b", "right_hand_c", //thumb
                                                  "right_hand_e2",                //additional point
                                                  "right_hand_j", "right_hand_n", "right_hand_r",  // Ring
                                                  "right_hand_i", "right_hand_m", "right_hand_q", //pinky
                                                  "right_hand_l", "right_hand_p", "right_hand_t", // index
                                                  "right_hand_k", "right_hand_o", "right_hand_s", // middle
                                                  };
    
            Eigen::VectorXd mean_pose;
    
            raisim::PolyLine *lines[17];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[22]; // modified for schunk hand
            raisim::Visuals *sphere[22]; // modified for schunk hand
            raisim::Visuals *joints_sphere[22]; // modified for schunk hand
            raisim::Visuals *aff_center_visual[7]; // floating specific (was 7), user wanted 5 for fixed. I'll keep 7 for floating and adjust based on further instructions
            raisim::Visuals *wrist_target[2];
    
            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class SCHUNK_TEST : public RaisimGymEnv {
        public:

            explicit SCHUNK_TEST(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                lift = false;
                root_guided = cfg["root_guided"].As<bool>();

                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/schunk_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();

                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);


                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts);
                // For Schunk hand, increase weight for fingertip links
                finger_weights_contact(3) *= 3; // Thumb tip link (right_hand_c)
                finger_weights_contact(7) *= 3; // Ring tip link (right_hand_r)
                finger_weights_contact(10) *= 3; // Pinky tip link (right_hand_q)
                finger_weights_contact(13) *= 3; // Index tip link (right_hand_t)
                finger_weights_contact(16) *= 3; // Middle tip link (right_hand_s)
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                for(int i=0; i< num_contacts; i++){
                    if (i >= 1 && i <= 3) { // Thumb links (index 1 to 3) have higher impulse limits
                        impulse_high[i] = 0.2;
                    } else {
                        impulse_high[i] = 0.1;
                    }
                    impulse_low[i] = -0.0;
                }


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                /// MUST BE DONE FOR ALL ENVIRONMENTS
                obDim_r_ = 132;
                obDim_l_ = 1;
                gsDim_ = 177;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
            //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
            //                table_top->setPosition(1.25, 0, 0.475);
            //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
            //                leg1->setPosition(0.2625,0.4675,0.2375);
            //                leg2->setPosition(2.2275,0.4875,0.2375);
            //                leg3->setPosition(0.2625,-0.4675,0.2375);
            //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");
                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();

                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    lift = false;
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);

                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                lift = false;

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);

                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                obj_weight = arctic->getTotalMass();

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;
                        Eigen::Vector3d hand_center_temp_fingers;
                        hand_center_temp_fingers.setZero();
                        int finger_count = 0;

                        // Repository metadata.
                        // Index finger middle joint: body_parts_r_[15]
                        mano_r_->getFramePosition(body_parts_r_[15], sphere_pos);
                        hand_center_temp_fingers += sphere_pos.e();
                        finger_count++;

                        // Middle finger middle joint: body_parts_r_[19]
                        mano_r_->getFramePosition(body_parts_r_[19], sphere_pos);
                        hand_center_temp_fingers += sphere_pos.e();
                        finger_count++;

                        // Ring finger middle joint: body_parts_r_[7]
                        mano_r_->getFramePosition(body_parts_r_[7], sphere_pos);
                        hand_center_temp_fingers += sphere_pos.e();
                        finger_count++;

                        // Pinky finger middle joint: body_parts_r_[11]
                        mano_r_->getFramePosition(body_parts_r_[11], sphere_pos);
                        hand_center_temp_fingers += sphere_pos.e();
                        finger_count++;

                        hand_center_temp_fingers /= finger_count; // metadata

                        // Repository metadata.
                        raisim::Vec<3> thumb_tip_pos;
                        mano_r_->getFramePosition(body_parts_r_[4], thumb_tip_pos); // Thumb tip: body_parts_r_[4]

                        Eigen::Vector3d finger_center;
                        finger_center = hand_center_temp_fingers; // metadata finger_center

                        hand_center_temp = (hand_center_temp_fingers + thumb_tip_pos.e()) / 2.0; // metadata

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    if (lift){
                        wrist_bias_w[2] += 0.2;
                    }
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    if (lift){
                        wrist_bias_w[2] += 0.2;
                    }
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }

                if (root_guided){
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();


                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::max(0.0, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);


                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame


                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);

                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
            Eigen::Vector3d current_across_axis_w;

                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);

            raisim::Vec<3> trans_r;
            raisim::Mat<3,3> rot_r;
            raisim::Vec<4> quat_r;

            Eigen::VectorXd obj_state;
            obj_state.setZero(8);

            mano_r_->getFramePosition(body_parts_r_[0], trans_r);
            mano_r_->getFrameOrientation(body_parts_r_[0], rot_r);

            raisim::rotMatToQuat(rot_r, quat_r);
            obj_state = arctic->getGeneralizedCoordinate().e();

                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul,
                                trans_r.e(),
                                quat_r.e(),
                                gc_r_.tail(gcDim_ - 6),
                                obj_state;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {
                lift = true;
            }
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
        int gcDim_, gvDim_, nJoints_;
        int gcDim_obj, gvDim_obj;
        bool visualizable_ = false;
        bool unseen = false;
        bool new_category = false;
        bool lift = false;
        bool root_guided = false;
        raisim::ArticulatedSystem* mano_;
        Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
        Eigen::VectorXd obj_pos_init_;
        Eigen::VectorXd joint_pos_in_world;
        std::string load_set;

        double affordance_contact_reward_r= 0.0;
        double not_affordance_contact_reward_r = 0.0;
        double affordance_impulse_reward_r= 0.0;
        double not_affordance_impulse_reward_r = 0.0;
        double wrist_vel_reward_r = 0.0;
        double wrist_qvel_reward_r = 0.0;
        double obj_vel_reward_r = 0.0;
        double obj_qvel_reward_r = 0.0;
        double obj_weight = 0.0;
        double mano_weight = 0.0;
        double wrist_Pgain = 100.0;
        double wrist_Dgain = 0.1;
        double rot_Pgain = 100.0;
        double rot_Dgain = 0.2;


        int num_contacts = 17;
        int num_bodyparts = 22;

        raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
        raisim::Vec<3> init_root_r_, init_obj_;
        Eigen::Vector3d init_center;
        Eigen::VectorXd joint_limit_high, joint_limit_low;
        Eigen::VectorXd impulse_high, impulse_low;
        Eigen::VectorXd actionMean_r_, actionStd_r_;
        Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
        Eigen::VectorXd finger_weights_contact, finger_weights_aff;
        Eigen::VectorXd contacts_r_af, impulses_r_af;
        Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
        Eigen::VectorXd contact_body_idx_r_;
        Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
        raisim::Vec<3> Position;
        raisim::Vec<3> wrist_vel, wrist_qvel;
        Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
        Eigen::VectorXd right_hand_torque, right_wrist_torque;
        Eigen::VectorXd pTarget_clipped_r;
        Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
        Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

        Eigen::VectorXd last_closest_points;
        Eigen::VectorXd joints_pos;

        raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
        raisim::Box *box;
        raisim::ArticulatedSystem *arctic, *mano_r_;
        raisim::ArticulatedSystemVisual *arcticVisual;
        raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
        raisim::Vec<4> obj_quat;
        raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
        Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
        bool first_reset_=true;
        float rewards_sum_[2];
        bool has_non_aff = false;
        double hand_mass = 0.0;

        std::string body_parts_r_[22] =  {
                                 "z_rotation_joint",  //palm
                                 "right_hand_Thumb_Flexion", "right_hand_j3", "right_hand_j4", "thtip_joint",  //thumb
                                  "right_hand_j5",         //additional point
                                 "right_hand_Ring_Finger", "right_hand_j12", "right_hand_j16", "rftip_joint",  //Ring
                                 "right_hand_Pinky", "right_hand_j13", "right_hand_j17", "lftip_joint",     //pinky
                                 "right_hand_Index_Finger_Proximal","right_hand_Index_Finger_Distal", "right_hand_j14", "fftip_joint", // Index
                                 "right_hand_Middle_Finger_Proximal","right_hand_Middle_Finger_Distal", "right_hand_j15", "mftip_joint", // middle
                                 };


        std::string contact_bodies_r_[17] =  {
                                              "right_hand_base_link",       //palm
                                              "right_hand_a", "right_hand_b", "right_hand_c", //thumb
                                              "right_hand_e2",                //additional point
                                              "right_hand_j", "right_hand_n", "right_hand_r",  // Ring
                                              "right_hand_i", "right_hand_m", "right_hand_q", //pinky
                                              "right_hand_l", "right_hand_p", "right_hand_t", // index
                                              "right_hand_k", "right_hand_o", "right_hand_s", // middle
                                              };


        Eigen::VectorXd mean_pose;

        raisim::PolyLine *lines[17];
        raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
        raisim::Visuals *Cylinder[22];
        raisim::Visuals *sphere[22];
        raisim::Visuals *joints_sphere[22];
        raisim::Visuals *aff_center_visual[7]; // Test env has only 7 visuals for aff_center. Keep this as 7.
        raisim::Visuals *wrist_target[2];

        std::map<int,int> contactMapping_r_;
        std::map<int,int> contactMapping_l_;
        std::string resourceDir_;
        std::vector<raisim::Vec<2>> joint_limits_;
        raisim::PolyLine *line;
    };
    class BRAINCO_FLOATING : public RaisimGymEnv {

        public:

            explicit BRAINCO_FLOATING(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                root_guided = cfg["root_guided"].As<bool>();


                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);

                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);

                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();

                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/brainco/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();


                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");

                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;

                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);

                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);

                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);


                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);

                contact_body_idx_r_.setZero(num_contacts);

                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();

                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);

                pTarget_clipped_r.setZero(gcDim_);

                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);

                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts); // modified
                // Give higher weight to fingertips (thumb_L2, index_L2, middle_L2, ring_L2, pinky_L2)
                finger_weights_contact(2) *= 3; 
                finger_weights_contact(4) *= 3;  
                finger_weights_contact(6) *= 3;  
                finger_weights_contact(8) *= 3; 
                finger_weights_contact(10) *= 3;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                for(int i=0; i<num_contacts; i++){ // modified
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                // Set higher impulse limits for fingertips
                impulse_high[2] = 0.2;  
                impulse_high[4] = 0.2;      
                impulse_high[6] = 0.2;  
                impulse_high[8] = 0.2;  
                impulse_high[10] = 0.2; 


                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);

                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));

                obDim_r_ = 90; // corrected to 90 based on re-calculation and gcDim_ = 17
                obDim_l_ = 1;
                gsDim_ = 106; // modified: Adjusted based on global_state_ composition (3+48+48+3+3+1 = 106)
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);

                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();

                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();

                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }

                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);

                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);

                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }

                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);

                    /// Create table
                    //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    //                table_top->setPosition(1.25, 0, 0.475);
                    //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    //                leg1->setPosition(0.2625,0.4675,0.2375);
                    //                leg2->setPosition(2.2275,0.4875,0.2375);
                    //                leg3->setPosition(0.2625,-0.4675,0.2375);
                    //                leg4->setPosition(2.2275,-0.4875,0.2375);

                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);

                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }

            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");

                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();

                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }

            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}

            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);

                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object

                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);

                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);

                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }

            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                            const Eigen::Ref<EigenVec>& init_state_l,
                            const Eigen::Ref<EigenVec>& init_vel_r,
                            const Eigen::Ref<EigenVec>& init_vel_l,
                            const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);

                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);

                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);

                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));

                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>();
                gv_set_r_ = init_vel_r.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();

                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse

                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();

                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);

                /// set initial object state

                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof

                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);

                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);

                obj_weight = arctic->getTotalMass();

                updateObservation();

            }

            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                        const Eigen::Ref<EigenVec>& obj_center,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                        const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                        const Eigen::Ref<EigenVec>& goal_pose_r,
                        const Eigen::Ref<EigenVec>& goal_pose_l,
                        const Eigen::Ref<EigenVec>& goal_qpos_r,
                        const Eigen::Ref<EigenVec>& goal_qpos_l,
                        const Eigen::Ref<EigenVec>& goal_contacts_r,
                        const Eigen::Ref<EigenVec>& goal_contacts_l) final {

                        afford_center = target_center.cast<double>();
                        obj_center_o = obj_center.cast<double>();
                        Eigen::Vector3d hand_center_temp;
                        hand_center.setZero();
                        hand_center_temp.setZero();
                        raisim::Vec<3> sphere_pos;
                        // modified: calculate initial hand_center_temp by averaging the second (middle) segments of index, middle, ring, and pinky fingers
                        for(int i = 0; i < 4 ; i++){ // modified: loop for 4 fingers (index, middle, ring, pinky)
                            mano_r_->getFramePosition(body_parts_r_[5+ i*3 + 1], sphere_pos); 
                            hand_center_temp[0] += sphere_pos.e()[0];
                            hand_center_temp[1] += sphere_pos.e()[1];
                            hand_center_temp[2] += sphere_pos.e()[2];
                        }
                        hand_center_temp /= 4; // modified: Averaged over 4 points
                        // modified: further refine with thumb tip position
                        mano_r_->getFramePosition(body_parts_r_[3], sphere_pos); // modified: Index 3 for "thumb_tip_joint"
                        Eigen::Vector3d finger_center;
                        finger_center.setZero();
                        finger_center[0] = hand_center_temp[0];
                        finger_center[1] = hand_center_temp[1];
                        finger_center[2] = hand_center_temp[2];

                        hand_center_temp[0] += sphere_pos.e()[0];
                        hand_center_temp[1] += sphere_pos.e()[1];
                        hand_center_temp[2] += sphere_pos.e()[2];
                        hand_center_temp /= 2;

                        Eigen::Vector3d across_axis_w;
                        across_axis_w = finger_center - hand_center_temp;
                        across_axis_w = across_axis_w / across_axis_w.norm();

                        raisim::Vec<3> wrist_pos_w;
                        raisim::Mat<3,3> wrist_mat_r;
                        mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                        mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                        Eigen::Vector3d bias, wrist_target_temp;
                        auto affordance_id = arctic->getBodyIdx("top");
                        arctic->getPosition(affordance_id, Obj_Position_init);
                        arctic->getOrientation(affordance_id, Obj_orientation_init);
                        raisim::Mat<3,3> Obj_orientation_init_trans;
                        raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                        raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                        bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                        wrist_target_temp = wrist_pos_w.e() + bias;
                        wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());

                        hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                        hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                        hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                        hand_center = wrist_mat_r.e().transpose() * hand_center_temp;

                        Eigen::Vector3d grasp_axis_w;
                        grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                        grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                        across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                        across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                        }

            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {

                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);

                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);

                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
                // if (visualizable_){
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }

                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                // }
                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }


                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)

                /// Clip targets to limits

                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);

                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);

                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;

                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }

                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);

                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }

                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();


                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::min(obj_weight * 5, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));

                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;

                return rewards_sum_;
            }

            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();

                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);

                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();

                mano_r_->getState(gc_r_, gv_r_);

                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);


                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame


                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();

                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }

                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }

                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }

                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }

                right_hand_torque = (pTarget_clipped_r - gc_r_);

                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;

                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                obDouble_r_ << target_center_dif,           // 3, hand center diff
                            gc_r_.segment(3, 3) - wrist_target_euler.e(),         // 3, wrist orientation diff
                            gc_r_.tail(gcDim_ - 6),
                                right_hand_torque,
                                wrist_vel_in_wrist,
                                wrist_qvel_in_wrist,
                                contacts_r_af,
                                impulses_r_af,
                                obj_vel_in_wrist,
                                obj_qvel_in_wrist,
                                contacts_r_non_af,
                                impulses_r_non_af;

                raisim::transpose(Obj_orientation_temp, Obj_orientation);

                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);

                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];

                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);

                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
            Eigen::Vector3d current_across_axis_w;

                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;

            Eigen::Vector3d current_grasp_axis_w;
            current_grasp_axis_w = wrist_mat_r.e() * hand_center;
            current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();

            Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
            current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
            current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;

            float across_mul = 0;
            across_mul = current_across_axis_o.dot(across_axis_o);

                global_state_ << obj_pose_wrist.e(),
                                frame_y_in_obj,
                                joint_pos_in_obj,
                                Obj_Position.e(),
                                current_grasp_axis_o - grasp_axis_o,
                                across_mul;
            }

            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }

            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }


            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {}
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
                raisim::Vec<3> obj_current_pos;
                auto top_id = arctic->getBodyIdx("top");
                arctic->getPosition(top_id, obj_current_pos); //Obj_Position: object base position at this time

                double displacement = (obj_current_pos.e() - obj_pos_init_.head(3)).squaredNorm();
                if (displacement > 0.1){
                    terminalReward = -10;
                    return true;
                }

                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }

                return false;
            }

        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;

            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;


            int num_contacts = 11; // modified
            int num_bodyparts = 16; // modified

            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;

            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;

            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;


            std::string body_parts_r_[16] =  { // modified
                                    "z_rotation_joint",    // metadata
                                    "right_thumb_metacarpal_joint", "right_thumb_distal_joint", "right_thumb_tip_joint",
                                    "right_index_proximal_joint", "right_index_distal_joint", "right_index_tip_joint",
                                    "right_middle_proximal_joint", "right_middle_distal_joint", "right_middle_tip_joint",
                                    "right_ring_proximal_joint", "right_ring_distal_joint", "right_ring_tip_joint",
                                    "right_pinky_proximal_joint", "right_pinky_distal_joint", "right_pinky_tip_joint"
                                }; // index for other fingers using distal as intermediate slotoint", "right_pinky_tip_joint"}; // index for other fingers using distal as intermediate slot

            std::string contact_bodies_r_[11] =  { 
                                "right_base_link", // modified: Palm link name
                                "right_thumb_proximal_link", "right_thumb_distal_link", // modified
                                "right_index_proximal_link", "right_index_distal_link", // modified
                                "right_middle_proximal_link", "right_middle_distal_link", // modified
                                "right_ring_proximal_link", "right_ring_distal_link", // modified
                                "right_pinky_proximal_link", "right_pinky_distal_link" // modified
            };

            raisim::PolyLine *lines[16];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[16]; // modified
            raisim::Visuals *sphere[16]; // modified
            raisim::Visuals *joints_sphere[16]; // modified
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];

            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class BRAINCO_TEST : public RaisimGymEnv {

        public:
    
            explicit BRAINCO_TEST(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
    
                root_guided = cfg["root_guided"].As<bool>();
                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                lift = false;
    
                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);
    
                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);
    
                /// add mano
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/brainco/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));
                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();
    
                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");
    
                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    
                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;
    
                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);
    
                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);
    
                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);
    
    
                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);
    
                contact_body_idx_r_.setZero(num_contacts);
    
                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();
    
                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);
    
                pTarget_clipped_r.setZero(gcDim_);
    
                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);
    
                /// initialize 3D positions weights for fingertips higher than for other fingerparts
                finger_weights_contact.setOnes(num_contacts); // modified
                // Give higher weight to fingertips (thumb_L2, index_L2, middle_L2, ring_L2, pinky_L2)
                finger_weights_contact(2) *= 3; 
                finger_weights_contact(4) *= 3;  
                finger_weights_contact(6) *= 3;  
                finger_weights_contact(8) *= 3; 
                finger_weights_contact(10) *= 3;
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;
    
                for(int i=0; i<num_contacts; i++){ // modified
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                }
                // Set higher impulse limits for fingertips
                impulse_high[2] = 0.2;  // thumb_L2
                impulse_high[4] = 0.2;  // index_L2
                impulse_high[6] = 0.2;  // middle_L2
                impulse_high[8] = 0.2; // ring_L2
                impulse_high[10] = 0.2; // pinky_L2
    
    
                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
    
                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));
    
                /// MUST BE DONE FOR ALL ENVIRONMENTS
                obDim_r_ = 90; // corrected to  based on re-calculation and gcDim_ = 16
                obDim_l_ = 1;
                gsDim_ = 132; // modified: Adjusted based on global_state_ composition (3+48+48+3+3+1+3+4+(16-6)+8 = 132)
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);
    
                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();
    
                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();
    
                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }
    
                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);
    
                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);
    
                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }
    
                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);
    
                    /// Create table
                        //                table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                        //                table_top->setPosition(1.25, 0, 0.475);
                        //                leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                        //                leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                        //                leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                        //                leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                        //                leg1->setPosition(0.2625,0.4675,0.2375);
                        //                leg2->setPosition(2.2275,0.4875,0.2375);
                        //                leg3->setPosition(0.2625,-0.4675,0.2375);
                        //                leg4->setPosition(2.2275,-0.4875,0.2375);
    
                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);
    
                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }
    
            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");
                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
    
                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }
    
            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}
    
            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    lift = false;
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);
    
                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object
    
                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);
    
                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
    
                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }
    
            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                             const Eigen::Ref<EigenVec>& init_state_l,
                             const Eigen::Ref<EigenVec>& init_vel_r,
                             const Eigen::Ref<EigenVec>& init_vel_l,
                             const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);
    
                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                lift = false;
    
                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);
    
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();
    
                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse
    
                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();
    
                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial object state
    
                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof
    
                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);
    
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                obj_weight = arctic->getTotalMass();

                updateObservation();
            }
    
            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                           const Eigen::Ref<EigenVec>& obj_center,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                           const Eigen::Ref<EigenVec>& goal_pose_r,
                           const Eigen::Ref<EigenVec>& goal_pose_l,
                           const Eigen::Ref<EigenVec>& goal_qpos_r,
                           const Eigen::Ref<EigenVec>& goal_qpos_l,
                           const Eigen::Ref<EigenVec>& goal_contacts_r,
                           const Eigen::Ref<EigenVec>& goal_contacts_l) final {
    
                           afford_center = target_center.cast<double>();
                           obj_center_o = obj_center.cast<double>();
                           Eigen::Vector3d hand_center_temp;
                           hand_center.setZero();
                           hand_center_temp.setZero();
                           raisim::Vec<3> sphere_pos;
                           // modified: calculate initial hand_center_temp by averaging the second (middle) segments of index, middle, ring, and pinky fingers
                           for(int i = 0; i < 4 ; i++){ // modified: loop for 4 fingers (index, middle, ring, pinky)
                               mano_r_->getFramePosition(body_parts_r_[5 + i*3], sphere_pos); 
                               hand_center_temp[0] += sphere_pos.e()[0];
                               hand_center_temp[1] += sphere_pos.e()[1];
                               hand_center_temp[2] += sphere_pos.e()[2];
                           }
                           hand_center_temp /= 4; // modified: Averaged over 4 points
                           // modified: further refine with thumb tip position
                           mano_r_->getFramePosition(body_parts_r_[3], sphere_pos); // modified: Index 3 for "thumb_tip_joint"
                           Eigen::Vector3d finger_center;
                           finger_center.setZero();
                           finger_center[0] = hand_center_temp[0];
                           finger_center[1] = hand_center_temp[1];
                           finger_center[2] = hand_center_temp[2];
    
                           hand_center_temp[0] += sphere_pos.e()[0];
                           hand_center_temp[1] += sphere_pos.e()[1];
                           hand_center_temp[2] += sphere_pos.e()[2];
                           hand_center_temp /= 2;
    
                           Eigen::Vector3d across_axis_w;
                           across_axis_w = finger_center - hand_center_temp;
                           across_axis_w = across_axis_w / across_axis_w.norm();
    
                           raisim::Vec<3> wrist_pos_w;
                           raisim::Mat<3,3> wrist_mat_r;
                           mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                           mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                           Eigen::Vector3d bias, wrist_target_temp;
                           auto affordance_id = arctic->getBodyIdx("top");
                           arctic->getPosition(affordance_id, Obj_Position_init);
                           arctic->getOrientation(affordance_id, Obj_orientation_init);
                           raisim::Mat<3,3> Obj_orientation_init_trans;
                           raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                           raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                           bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                           wrist_target_temp = wrist_pos_w.e() + bias;
                           wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());
    
                           hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                           hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                           hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                           hand_center = wrist_mat_r.e().transpose() * hand_center_temp;
    
                           Eigen::Vector3d grasp_axis_w;
                           grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                           grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                           across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                           across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                           }
    
            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
    
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);
    
                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
    
                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];
    
                Eigen::Vector3d wrist_bias_w, wrist_bias;
                if (root_guided){
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                }
                else{
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                }
                wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                if (lift){
                    wrist_bias_w[2] += 0.2;
                }
                wrist_bias = init_or_r_.e() * wrist_bias_w;
                actionMean_r_.head(3) += wrist_bias;
                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }
    
                if (root_guided){
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
    
                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)
    
                /// Clip targets to limits
    
                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);
    
                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);
    
                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;
    
                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }
    
                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
    
                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }
    
                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();
    
    
                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::max(0.0, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));
    
                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;
    
                return rewards_sum_;
            }
    
            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }

                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();
    
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);
    
                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();
    
                mano_r_->getState(gc_r_, gv_r_);
    
                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);
    
    
                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame
    
                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();
    
                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }
    
                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }
    
                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }
    
                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }
    
                right_hand_torque = (pTarget_clipped_r - gc_r_);
    
                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;
    
                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                obDouble_r_ << target_center_dif,  //3
                               gc_r_.segment(3, 3) - wrist_target_euler.e(), //3
                               gc_r_.tail(gcDim_ - 6), //11 (gcDim_ = 17)
                                right_hand_torque, //17 (gcDim_ = 17)
                                wrist_vel_in_wrist, //3
                                wrist_qvel_in_wrist, //3
                                contacts_r_af, //11
                                impulses_r_af, //11
                                obj_vel_in_wrist, //3
                                obj_qvel_in_wrist, //3
                                contacts_r_non_af, //11
                                impulses_r_non_af; //11
                                //total:3+3+11+17+3+3+11+11+3+3+11+11= 90

                raisim::transpose(Obj_orientation_temp, Obj_orientation);
    
                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);
    
                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    if(i > 16){
                        frame_y_frame.setZero();
                        frame_y_frame[0] = -0.717;
                        frame_y_frame[2] = -0.717;
                    }
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];
    
                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);
    
                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
               Eigen::Vector3d current_across_axis_w;
    
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;
    
               Eigen::Vector3d current_grasp_axis_w;
               current_grasp_axis_w = wrist_mat_r.e() * hand_center;
               current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();
    
               Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
               current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
               current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;
    
               float across_mul = 0;
               across_mul = current_across_axis_o.dot(across_axis_o);
    
    
              raisim::Vec<3> trans_r;
             raisim::Mat<3,3> rot_r;
             raisim::Vec<4> quat_r;
    
             Eigen::VectorXd obj_state;
             obj_state.setZero(8);
    
             mano_r_->getFramePosition(body_parts_r_[0], trans_r); // Position: body position of a body in mano
             mano_r_->getFrameOrientation(body_parts_r_[0], rot_r); // Body_orientation:
    
             raisim::rotMatToQuat(rot_r, quat_r);
              obj_state = arctic->getGeneralizedCoordinate().e();
    
                global_state_ << obj_pose_wrist.e(),//3
                                 frame_y_in_obj, //3*16=48
                                 joint_pos_in_obj,//48
                                 Obj_Position.e(), //3
                                 current_grasp_axis_o - grasp_axis_o, //3
                                 across_mul, //1
                                 trans_r.e(), //3 
                                 quat_r.e(), //4 
                                 gc_r_.tail(gcDim_ - 6), //11
                                 obj_state; //8
                                 //total:3+48+48+3+3+1+3+4+11+8=132
                
            }
    
            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }
    
            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }
    
    
            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {
                lift = true;
            }
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {
    
                if(obDouble_r_.hasNaN() || global_state_.hasNaN())
                {
                    return true;
                }
    
                return false;
            }
    
        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool root_guided = true;
            bool unseen = false;
            bool new_category = false;
            bool lift = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;
    
            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;
    
    
            int num_contacts = 11; // modified
            int num_bodyparts = 16; // modified
    
            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;
    
            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;
    
            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;
    
    
            std::string body_parts_r_[16] =  { "z_rotation_joint",
                            "right_thumb_metacarpal_joint","right_thumb_distal_joint","right_thumb_tip_joint",
                           "right_index_proximal_joint","right_index_distal_joint","right_index_tip_joint",
                           "right_middle_proximal_joint","right_middle_distal_joint","right_middle_tip_joint",
                           "right_ring_proximal_joint","right_ring_distal_joint","right_ring_tip_joint",
                           "right_pinky_proximal_joint","right_pinky_distal_joint","right_pinky_tip_joint",
            };
    
            std::string contact_bodies_r_[11] =  { 
                                "right_base_link",
                                "right_thumb_proximal_link", "right_thumb_distal_link", // modified
                                "right_index_proximal_link", "right_index_distal_link", // modified
                                "right_middle_proximal_link", "right_middle_distal_link",
                                "right_ring_proximal_link", "right_ring_distal_link",
                                "right_pinky_proximal_link", "right_pinky_distal_link"};
    
    
            Eigen::VectorXd mean_pose;
    
            raisim::PolyLine *lines[16];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[16]; // modified
            raisim::Visuals *sphere[16]; // modified
            raisim::Visuals *joints_sphere[16]; // modified
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];
    
            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
    class BARRETT_TEST : public RaisimGymEnv {
        public:
            explicit BARRETT_TEST(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                    RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {
    
                visualizable_ = cfg["visualize"].As<bool>();
                load_set = cfg["load_set"].As<std::string>();
                lift = false;
                root_guided = cfg["root_guided"].As<bool>();

                /// create world
                world_ = std::make_unique<raisim::World>();
                world_->addGround();
                world_->setERP(0.0);
    
                world_->setMaterialPairProp("object", "object", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("object", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setMaterialPairProp("finger", "finger", 3.0, 0.0, 0.0, 3.0, 0.1);
                world_->setDefaultMaterial(3.0, 0, 0, 3.0, 0.1);
    
                /// add hand
                std::string hand_model_r =  cfg["hand_model_r"].As<std::string>();
                resourceDir_ = resourceDir;
                mano_r_ = world_->addArticulatedSystem(resourceDir+"/barrett_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63));
                // mano_r_ = world_->addArticulatedSystem(resourceDir+"/barrett_hand/"+hand_model_r,"",{},raisim::COLLISION(0),raisim::COLLISION(0)|raisim::COLLISION(2)|raisim::COLLISION(63));

                mano_r_->setName("mano_r");
                hand_mass = mano_r_->getTotalMass();
    
                /// add table
                box = static_cast<raisim::Box*>(world_->addBox(2, 1, 0.5, 100, "", raisim::COLLISION(1)));
                box->setPosition(1.25, 0, 0.25);
                box->setAppearance("0.0 0.0 0.0 0.0");
    
                /// set PD control mode
                mano_r_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    
                /// get actuation dimensions
                gcDim_ = mano_r_->getGeneralizedCoordinateDim();
                gvDim_ = mano_r_->getDOF();
                nJoints_ = gcDim_-3;
    
                gc_r_.setZero(gcDim_);
                gv_r_.setZero(gvDim_);
                gc_set_r_.setZero(gcDim_); gv_set_r_.setZero(gvDim_);
    
                /// initialize all variables
                pTarget_r_.setZero(gcDim_); vTarget_r_.setZero(gvDim_); pTarget6_r_.setZero(6);
                actionDim_ = gcDim_;
                actionMean_r_.setZero(actionDim_);
                actionStd_r_.setOnes(actionDim_);
                joint_limit_high.setZero(actionDim_); joint_limit_low.setZero(actionDim_);
    
                right_hand_torque.setZero(gcDim_);
                right_wrist_torque.setZero(6);
    
    
                wrist_mat_r_in_obj_init.setZero();
                init_or_r_.setZero();  init_rot_r_.setZero(); init_root_r_.setZero();
                init_obj_rot_.setZero(); init_obj_or_.setZero(); init_obj_.setZero();
                wrist_mat_r_in_obj_init.setZero();
                wrist_vel.setZero(); wrist_qvel.setZero(); wrist_vel_in_wrist.setZero(); wrist_qvel_in_wrist.setZero();
                hand_center.setZero();
                afford_center.setZero();
                wrist_target_o.setZero();
                init_center.setZero();
                frame_y_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_obj.setZero(num_bodyparts*3);
                joint_pos_in_world.setZero(num_bodyparts*3);
    
                contact_body_idx_r_.setZero(num_contacts);
    
                obj_pos_init_.setZero(8);
                Position.setZero();
                Obj_Position.setZero(); Obj_Position_init.setZero(); Obj_orientation.setZero(); Obj_orientation_temp.setZero(); Obj_orientation_init.setZero();
                obj_quat.setZero();
                Obj_qvel.setZero(); Obj_linvel.setZero();
                obj_vel_in_wrist.setZero(); obj_qvel_in_wrist.setZero();
    
                contacts_r_af.setZero(num_contacts); contacts_r_non_af.setZero(num_contacts);
                impulses_r_af.setZero(num_contacts); impulses_r_non_af.setZero(num_contacts);
                impulse_high.setZero(num_contacts); impulse_low.setZero(num_contacts);
    
                pTarget_clipped_r.setZero(gcDim_);
    
                last_closest_points.setZero(num_bodyparts*3);
                joints_pos.setZero(num_bodyparts*3);
    
                /// Barrett: no explicit fingertip frames. Weight distal links slightly higher.
                finger_weights_contact.setOnes(num_contacts);
                for(int i = 0; i < num_contacts; i++){
                    const std::string& name = contact_bodies_r_[i];
                    if (name.find("dist") != std::string::npos) {
                        finger_weights_contact(i) *= 2.0;
                    }
                }
                finger_weights_contact /= finger_weights_contact.sum();
                finger_weights_contact *= num_contacts;

                finger_weights_aff.setOnes(num_bodyparts);
                finger_weights_aff /= finger_weights_aff.sum();
                finger_weights_aff *= num_bodyparts;

                for(int i=0; i<num_contacts; i++){
                    impulse_high[i] = 0.1;
                    impulse_low[i] = -0.0;
                    if (contact_bodies_r_[i].find("dist") != std::string::npos) {
                        impulse_high[i] = 0.2;
                    }
                }
    
    
                /// set PD gains
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
    
                mano_r_->setPdGains(jointPgain, jointDgain);
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));
                mano_r_->setGeneralizedCoordinate(Eigen::VectorXd::Zero(gcDim_));
    
                /// MUST BE DONE FOR ALL ENVIRONMENTS
                /// obDim = 12 + 2*gcDim + 4*num_contacts
                obDim_r_ = 76; // 12 + 2*14 + 4*9
                obDim_l_ = 1;
                /// Floating gsDim = 10 + 6*num_bodyparts; Test gsDim = 19 + 6*num_bodyparts + gcDim
                gsDim_ = 87;
                obDouble_r_.setZero(obDim_r_);
                obDouble_l_.setZero(obDim_l_);
                global_state_.setZero(gsDim_);
    
                float finger_action_std = cfg["finger_action_std"].As<float>();
                float rot_action_std = cfg["rot_action_std"].As<float>();
    
                /// retrieve joint limits from model
                joint_limits_ = mano_r_->getJointLimits();
    
                for(int i=0; i < int(gcDim_); i++){
                    actionMean_r_[i] = (joint_limits_[i][1]+joint_limits_[i][0])/2.0;
                    joint_limit_low[i] = joint_limits_[i][0];
                    joint_limit_high[i] = joint_limits_[i][1];
                }
    
                /// set actuation parameters
                actionStd_r_.setConstant(finger_action_std);
                actionStd_r_.head(3).setConstant(0.005);
                actionStd_r_.segment(3,3).setConstant(0.005);
    
                /// Initialize reward
                rewards_r_.initializeFromConfigurationFile(cfg["reward"]);
    
                for(int i = 0; i < num_contacts ;i++){
                    contact_body_idx_r_[i] =  mano_r_->getBodyIdx(contact_bodies_r_[i]);
                    contactMapping_r_.insert(std::pair<int,int>(int(mano_r_->getBodyIdx(contact_bodies_r_[i])),i));
                }
    
                /// start visualization server
                if (visualizable_) {
                    if(server_) server_->lockVisualizationServerMutex();
                    server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                    server_->launchServer(8081);
    
                    /// Create table
                    table_top = server_->addVisualBox("tabletop", 2.0, 1.0, 0.05, 0.44921875, 0.30859375, 0.1953125, 1, "");
                    table_top->setPosition(1.25, 0, 0.475);
                    leg1 = server_->addVisualCylinder("leg1", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    leg2 = server_->addVisualCylinder("leg2", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    leg3 = server_->addVisualCylinder("leg3", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    leg4 = server_->addVisualCylinder("leg4", 0.025, 0.475, 0.0, 0.0, 0.0, 1, "");
                    leg1->setPosition(0.2625,0.4675,0.2375);
                    leg2->setPosition(2.2275,0.4875,0.2375);
                    leg3->setPosition(0.2625,-0.4675,0.2375);
                    leg4->setPosition(2.2275,-0.4875,0.2375);
    
                    /// initialize Cylinders for sensor
                    for(int i = 0; i < num_bodyparts; i++){
                        Cylinder[i] = server_->addVisualCylinder(body_parts_r_[i]+"_cylinder", 0.005, 0.1, 1, 0, 1);
                        sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_sphere", 0.005, 0, 1, 0, 1);
                        joints_sphere[i] = server_->addVisualSphere(body_parts_r_[i]+"_joints_sphere", 0.01, 0, 0, 1, 1);
                    }
                    for(int i = 0; i < 5; i++){
                        aff_center_visual[i] = server_->addVisualSphere(body_parts_r_[i]+"_aff_center", 0.01, 0, 0, 1, 1);
                    }
                    aff_center_visual[5] = server_->addVisualSphere(body_parts_r_[5]+"_aff_center", 0.02, 1, 1, 0, 1);
                    aff_center_visual[6] = server_->addVisualSphere(body_parts_r_[6]+"_aff_center", 0.02, 1, 1, 0, 1);
                    wrist_target[0] = server_->addVisualSphere("wrist_target", 0.03, 1, 0, 1, 1);
                    wrist_target[1] = server_->addVisualSphere("wrist_start", 0.03, 1, 0, 1, 1);
    
                    if(server_) server_->unlockVisualizationServerMutex();
                }
            }
    
            void init() final { }
            void load_object(const Eigen::Ref<EigenVecInt>& obj_idx, const Eigen::Ref<EigenVec>& obj_weight, const Eigen::Ref<EigenVec>& obj_dim, const Eigen::Ref<EigenVecInt>& obj_type) final {}
            /// This function loads the object into the environment
            void load_articulated(const std::string& obj_model){
                arctic = static_cast<raisim::ArticulatedSystem*>(world_->addArticulatedSystem(resourceDir_+"/"+load_set+"/"+obj_model, "", {}, raisim::COLLISION(2), raisim::COLLISION(0)|raisim::COLLISION(1)|raisim::COLLISION(2)|raisim::COLLISION(63)));
                arctic->setName("arctic");
                gcDim_obj = arctic->getGeneralizedCoordinateDim();
                gvDim_obj = arctic->getDOF();
    
                Eigen::VectorXd gen_coord = Eigen::VectorXd::Zero(gcDim_obj);
                gen_coord[4] = 1;
                arctic->setGeneralizedCoordinate(gen_coord);
                arctic->setGeneralizedVelocity(Eigen::VectorXd::Zero(gvDim_obj));
                obj_weight = arctic->getTotalMass();
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
                arctic->setPdGains(objPgain, objDgain);
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                double non_aff_mass = arctic->getMass(non_affordance_id);
                if (non_aff_mass > 0.001){
                    has_non_aff = true;
                }
                else{
                    has_non_aff = false;
                }
            }
    
            void set_joint_sensor_visual(const Eigen::Ref<EigenVec>& joint_sensor_visual) final {
                raisim::Vec<3> sphere_pos_temp, sphere_pos_wrist, sphere_pos, joint_pos;
                raisim::Mat<3,3> frame_mat;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    joint_pos = joint_pos_in_world.segment(i*3,3).cast<double>();
                    sphere_pos_wrist = joint_sensor_visual.segment(i*3,3).cast<double>();
                    raisim::matvecmul(frame_mat, sphere_pos_wrist, sphere_pos_temp);
                    last_closest_points.segment(i*3,3) = sphere_pos_temp.e();
                    vecadd(joint_pos, sphere_pos_temp);
            }}
    
            /// Resets the object and hand to its initial pose
            void reset() final {
                if (first_reset_)
                {
                    first_reset_=false;
                }
                else{
                    lift = false;
                    /// all settings to initial state configuration
                    actionMean_r_.setZero();
                    mano_r_->setBasePos(init_root_r_);
                    mano_r_->setBaseOrientation(init_rot_r_);
                    mano_r_->setState(gc_set_r_, gv_set_r_);
    
                    gvDim_obj = arctic->getDOF();
                    arctic->setBasePos(init_obj_);
                    arctic->setBaseOrientation(init_obj_rot_);
                    arctic->setState(obj_pos_init_, Eigen::VectorXd::Zero(gvDim_obj));
                    //obj_pos_init: reset pose of object
    
                    box->clearExternalForcesAndTorques();
                    box->setPosition(1.25, 0, 0.25);
                    box->setOrientation(1,0,0,0);
                    box->setVelocity(0,0,0,0,0,0);
    
                    auto affordance_id = arctic->getBodyIdx("top");
                    arctic->getAngularVelocity(affordance_id, Obj_qvel);
                    updateObservation();
                    Eigen::VectorXd gen_force;
                    gen_force.setZero(gcDim_);
                    mano_r_->setGeneralizedForce(gen_force);
    
                    gen_force.setZero(gcDim_obj);
                    arctic->setGeneralizedForce(gen_force);
                }
            }
    
            /// Resets the state to a user defined input
            // obj_pose: 8 DOF [trans(3), ori(4, quat), joint angle(1)]
            void reset_state(const Eigen::Ref<EigenVec>& init_state_r,
                             const Eigen::Ref<EigenVec>& init_state_l,
                             const Eigen::Ref<EigenVec>& init_vel_r,
                             const Eigen::Ref<EigenVec>& init_vel_l,
                             const Eigen::Ref<EigenVec>& obj_pose) final {
                /// reset gains (only required in case for inference)
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                jointPgain.head(3).setConstant(wrist_Pgain);
                jointDgain.head(3).setConstant(wrist_Dgain);
                jointPgain.tail(gcDim_-3).setConstant(rot_Pgain);
                jointDgain.tail(gcDim_-3).setConstant(rot_Dgain);
                mano_r_->setPdGains(jointPgain, jointDgain);
    
                Eigen::VectorXd gen_force;
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                lift = false;
    
                /// reset table position (only required in case for inference)
                box->setPosition(1.25, 0, 0.25);
                box->setOrientation(1,0,0,0);
                box->setVelocity(0,0,0,0,0,0);
    
                Eigen::VectorXd objPgain(gvDim_obj), objDgain(gvDim_obj);
                objPgain.setZero();
                objDgain.setZero();
                arctic->setPdGains(objPgain, objDgain);
    
                mano_r_->setGeneralizedForce(Eigen::VectorXd::Zero(gcDim_));
                arctic->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_obj));
    
                gc_set_r_.head(6).setZero();
                gc_set_r_.tail(gcDim_-6) = init_state_r.tail(gcDim_-6).cast<double>(); //.cast<double>();
                gv_set_r_ = init_vel_r.cast<double>(); //.cast<double>();
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial root position in global frame as origin in new coordinate frame
                init_root_r_  = init_state_r.head(3);
                init_obj_ = obj_pose.head(3).cast<double>();
    
                /// set initial root orientation in global frame as origin in new coordinate frame
                raisim::Vec<4> quat;
                raisim::eulerToQuat(init_state_r.segment(3,3),quat); // initial base ori, in quat
                raisim::quatToRotMat(quat, init_rot_r_); // ..., in matrix
                raisim::transpose(init_rot_r_, init_or_r_); // ..., inverse
    
                int arcticCoordDim = arctic->getGeneralizedCoordinateDim();
                int arcticVelDim = arctic->getDOF();
                Eigen::VectorXd arcticCoord, arcticVel;
                arcticCoord.setZero(arcticCoordDim);
                arcticVel.setZero(arcticVelDim);
                arcticCoord = obj_pose.cast<double>();
    
                raisim::quatToRotMat(obj_pose.segment(3,4), init_obj_rot_);
                raisim::transpose(init_obj_rot_, init_obj_or_);
                arctic->setBasePos(init_obj_);
                arctic->setBaseOrientation(init_obj_rot_);
                arctic->setState(arcticCoord, arcticVel);
                mano_r_->setBasePos(init_root_r_);
                mano_r_->setBaseOrientation(init_rot_r_);
                mano_r_->setState(gc_set_r_, gv_set_r_);
    
                /// set initial object state
    
                obj_pos_init_  = obj_pose.cast<double>(); // 8 dof
    
                /// Set action mean to initial pose (first 6DoF since start at 0)
                actionMean_r_.setZero();
                actionMean_r_.tail(gcDim_-6) = gc_set_r_.tail(gcDim_-6);
    
                gen_force.setZero(gcDim_);
                mano_r_->setGeneralizedForce(gen_force);
    
                obj_weight = arctic->getTotalMass();
    
                updateObservation();
    
            }
    
            void set_goals(const Eigen::Ref<EigenVec>& target_center,
                           const Eigen::Ref<EigenVec>& obj_center,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_r,
                           const Eigen::Ref<EigenVec>& ee_goal_pos_l,
                           const Eigen::Ref<EigenVec>& goal_pose_r,
                           const Eigen::Ref<EigenVec>& goal_pose_l,
                           const Eigen::Ref<EigenVec>& goal_qpos_r,
                           const Eigen::Ref<EigenVec>& goal_qpos_l,
                           const Eigen::Ref<EigenVec>& goal_contacts_r,
                           const Eigen::Ref<EigenVec>& goal_contacts_l) final {
    
                           afford_center = target_center.cast<double>();
                           obj_center_o = obj_center.cast<double>();
                           Eigen::Vector3d hand_center_temp;
                           hand_center.setZero();
                           hand_center_temp.setZero();
                           raisim::Vec<3> sphere_pos;
                           // Barrett: average non-thumb middle segments, then refine with thumb distal.
                           // non-thumb: finger_2_med_joint (idx=5), finger_3_med_joint (idx=7)
                           mano_r_->getFramePosition(body_parts_r_[5], sphere_pos);
                           hand_center_temp[0] += sphere_pos.e()[0];
                           hand_center_temp[1] += sphere_pos.e()[1];
                           hand_center_temp[2] += sphere_pos.e()[2];
                           mano_r_->getFramePosition(body_parts_r_[7], sphere_pos);
                           hand_center_temp[0] += sphere_pos.e()[0];
                           hand_center_temp[1] += sphere_pos.e()[1];
                           hand_center_temp[2] += sphere_pos.e()[2];
                           hand_center_temp /= 2;

                           // thumb (no fingertip): use finger_1_dist_joint (idx=3)
                           mano_r_->getFramePosition(body_parts_r_[3], sphere_pos);
                           Eigen::Vector3d finger_center;
                           finger_center.setZero();
                           finger_center[0] = hand_center_temp[0];
                           finger_center[1] = hand_center_temp[1];
                           finger_center[2] = hand_center_temp[2];
    
                           hand_center_temp[0] += sphere_pos.e()[0];
                           hand_center_temp[1] += sphere_pos.e()[1];
                           hand_center_temp[2] += sphere_pos.e()[2];
                           hand_center_temp /= 2;
    
                           Eigen::Vector3d across_axis_w;
                           across_axis_w = finger_center - hand_center_temp;
                           across_axis_w = across_axis_w / across_axis_w.norm();
    
                           raisim::Vec<3> wrist_pos_w;
                           raisim::Mat<3,3> wrist_mat_r;
                           mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                           mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                           Eigen::Vector3d bias, wrist_target_temp;
                           auto affordance_id = arctic->getBodyIdx("top");
                           arctic->getPosition(affordance_id, Obj_Position_init);
                           arctic->getOrientation(affordance_id, Obj_orientation_init);
                           raisim::Mat<3,3> Obj_orientation_init_trans;
                           raisim::transpose(Obj_orientation_init, Obj_orientation_init_trans);
                           raisim::matmul(Obj_orientation_init_trans, wrist_mat_r, wrist_mat_r_in_obj_init);
                           bias = Obj_orientation_init.e() * afford_center + Obj_Position_init.e() - hand_center_temp;
                           wrist_target_temp = wrist_pos_w.e() + bias;
                           wrist_target_o = Obj_orientation_init.e().transpose() * (wrist_target_temp - Obj_Position_init.e());
    
                           hand_center_temp[0] = hand_center_temp[0] - wrist_pos_w[0];
                           hand_center_temp[1] = hand_center_temp[1] - wrist_pos_w[1];
                           hand_center_temp[2] = hand_center_temp[2] - wrist_pos_w[2];
                           hand_center = wrist_mat_r.e().transpose() * hand_center_temp;
    
                           Eigen::Vector3d grasp_axis_w;
                           grasp_axis_w = hand_center_temp / hand_center_temp.norm();
                           grasp_axis_o = Obj_orientation_init.e().transpose() * grasp_axis_w;
                           across_axis_o = Obj_orientation_init.e().transpose() * across_axis_w;
                           across_axis_wrist = wrist_mat_r.e().transpose() * across_axis_w;
                           }
    
            /// This function takes an environment step given an action (26DoF) input
            float* step(const Eigen::Ref<EigenVec>& action_r, const Eigen::Ref<EigenVec>& action_l) final {
    
                auto affordance_id = arctic->getBodyIdx("top");
                raisim::Vec<3> obj_pos_w, target_center, afford_center_w, obj_center_w;
                raisim::Mat<3,3> obj_rot_w;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                raisim::matvecmul(obj_rot_w, obj_center_o, obj_center_w);
    
                raisim::Vec<3> wrist_pos_w;
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
    
                target_center[0] = afford_center_w[0] + obj_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2];
    
                Eigen::Vector3d hand_center_w;
                hand_center_w = wrist_mat_r.e() * hand_center;
                hand_center_w[0] += wrist_pos_w[0];
                hand_center_w[1] += wrist_pos_w[1];
                hand_center_w[2] += wrist_pos_w[2];
    
                if (root_guided){
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = target_center[0] - hand_center_w[0];
                    wrist_bias_w[1] = target_center[1] - hand_center_w[1];
                    wrist_bias_w[2] = target_center[2] - hand_center_w[2];
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    if (lift){
                        wrist_bias_w[2] += 0.5;
                    }   
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                    raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                    raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                    raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                    raisim::matmul(wrist_mat_r_trans, target_wrist_in_world, target_wrist_in_wrist);
                    raisim::Vec<3> wrist_target_euler;
                    raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                    actionMean_r_.segment(3,3) = wrist_target_euler.e();  // use relative orientation
                }
                else{
                    Eigen::Vector3d wrist_bias_w, wrist_bias;
                    wrist_bias_w[0] = 0;
                    wrist_bias_w[1] = 0;
                    wrist_bias_w[2] = 0;
                    wrist_bias_w[2] += hand_mass * 9.81 * 0.01;
                    if (lift){
                        wrist_bias_w[2] += 0.2;
                    }   
                    wrist_bias = init_or_r_.e() * wrist_bias_w;
                    actionMean_r_.head(3) += wrist_bias;
                }

                // if (visualizable_){
                //     aff_center_visual[5]->setPosition(hand_center_w);
                //     aff_center_visual[6]->setPosition(target_center.e());
                // }
    

                /// Compute position target for actuators
                pTarget_r_ = action_r.cast<double>();
                pTarget_r_ = pTarget_r_.cwiseProduct(actionStd_r_); //residual action * scaling
                pTarget_r_ += actionMean_r_; //add wrist bias (first 3DOF) and last pose (23DoF)
    
                /// Clip targets to limits
    
                pTarget_clipped_r = pTarget_r_.cwiseMax(joint_limit_low).cwiseMin(joint_limit_high);
    
                /// Set PD targets (velocity zero)
                mano_r_->setPdTarget(pTarget_clipped_r, vTarget_r_);
    
                /// Apply N control steps
                for (int i = 0; i < int(control_dt_ / simulation_dt_ + 1e-10); i++){
                    if(server_) server_->lockVisualizationServerMutex();
                    world_->integrate();
                    if(server_) server_->unlockVisualizationServerMutex();
                }
                /// update observation and set new mean to the latest pose
                updateObservation();
                actionMean_r_ = gc_r_;
    
                affordance_contact_reward_r = contacts_r_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                if (has_non_aff){
                    not_affordance_contact_reward_r = contacts_r_non_af.cwiseProduct(finger_weights_contact).sum() / num_contacts;
                }
                else{
                    not_affordance_contact_reward_r = 0;
                }
    
                Eigen::VectorXd impulses_r_af_clipped, impulses_r_non_af_clipped;
                impulses_r_af_clipped.setZero(num_contacts);
                impulses_r_non_af_clipped.setZero(num_contacts);
                impulses_r_af_clipped = impulses_r_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
                impulses_r_non_af_clipped = impulses_r_non_af.cwiseMax(impulse_low).cwiseMin(impulse_high);
    
                affordance_impulse_reward_r = impulses_r_af_clipped.cwiseProduct(finger_weights_contact).sum();
                if (has_non_aff){
                    not_affordance_impulse_reward_r = impulses_r_non_af_clipped.cwiseProduct(finger_weights_contact).sum();
                }
                else{
                    not_affordance_impulse_reward_r = 0;
                }
    
                wrist_vel_reward_r = wrist_vel_in_wrist.squaredNorm();
                wrist_qvel_reward_r = wrist_qvel_in_wrist.squaredNorm();
                obj_vel_reward_r = Obj_linvel.e().squaredNorm();
                obj_qvel_reward_r = Obj_qvel.e().squaredNorm();
    
    
                rewards_r_.record("affordance_contact_reward", std::max(0.0, affordance_contact_reward_r));
                rewards_r_.record("not_affordance_contact_reward", std::max(0.0, not_affordance_contact_reward_r));
                rewards_r_.record("affordance_impulse_reward", std::max(0.0, affordance_impulse_reward_r));
                rewards_r_.record("not_affordance_impulse_reward", std::max(0.0, not_affordance_impulse_reward_r));
                rewards_r_.record("wrist_vel_reward_", std::max(0.0, wrist_vel_reward_r));
                rewards_r_.record("wrist_qvel_reward_", std::max(0.0, wrist_qvel_reward_r));
                rewards_r_.record("obj_vel_reward_", std::max(0.0, obj_vel_reward_r));
                rewards_r_.record("obj_qvel_reward_", std::max(0.0, obj_qvel_reward_r));
                rewards_r_.record("torque", std::max(0.0, (right_hand_torque.squaredNorm() + 4 * right_wrist_torque.squaredNorm())));
    
                rewards_sum_[0] = rewards_r_.sum();
                rewards_sum_[1] = 0;
    
                return rewards_sum_;
            }
    
            /// This function computes and updates the observation/state space
            void updateObservation() {
                raisim::Vec<3> Position_r;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFramePosition(body_parts_r_[i], Position_r);
                    joints_pos.segment(i*3,3) = Position_r.e();
                }
    
                // update observation
                impulses_r_af.setZero();
                contacts_r_af.setZero();
                impulses_r_non_af.setZero();
                contacts_r_non_af.setZero();
    
                raisim::Mat<3,3> wrist_mat_r, wrist_mat_r_trans;
                mano_r_->getFrameOrientation(body_parts_r_[0], wrist_mat_r);
                raisim::transpose(wrist_mat_r, wrist_mat_r_trans);
                mano_r_->getFrameVelocity(body_parts_r_[0], wrist_vel);
                mano_r_->getFrameAngularVelocity(body_parts_r_[0], wrist_qvel);
    
                wrist_vel_in_wrist = wrist_mat_r.e().transpose() * wrist_vel.e();
                wrist_qvel_in_wrist = wrist_mat_r.e().transpose() * wrist_qvel.e();
    
                mano_r_->getState(gc_r_, gv_r_);
    
                /// Get updated object pose
                auto affordance_id = arctic->getBodyIdx("top");
                auto non_affordance_id = arctic->getBodyIdx("bottom");
                arctic->getPosition(affordance_id, Obj_Position);
                arctic->getOrientation(affordance_id, Obj_orientation_temp);
                raisim::rotMatToQuat(Obj_orientation_temp, obj_quat);
                arctic->getAngularVelocity(affordance_id, Obj_qvel);
                arctic->getVelocity(affordance_id, Obj_linvel);
    
    
                obj_vel_in_wrist = wrist_mat_r.e().transpose() * Obj_linvel.e() - wrist_vel_in_wrist; // object velocity in wrist frame
                obj_qvel_in_wrist = wrist_mat_r.e().transpose() * Obj_qvel.e() - wrist_qvel_in_wrist; // object angular velocity in wrist frame
    
                /// compute current contacts of hand parts and the contact force
                auto& contact_list_obj = arctic->getContacts();
    
                for(auto& contact_af: mano_r_->getContacts()) {
                    if (contact_af.skip() || contact_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != affordance_id) continue;
                    contacts_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = 1;
                    impulses_r_af[contactMapping_r_[contact_af.getlocalBodyIndex()]] = contact_af.getImpulse().norm();
                }
    
                for(auto& contact_non_af: mano_r_->getContacts()) {
                    if (contact_non_af.skip() || contact_non_af.getPairObjectIndex() != arctic->getIndexInWorld()) continue;
                    if (contact_non_af.getPairObjectBodyType() != raisim::BodyType::DYNAMIC) continue;
                    if (contact_list_obj[contact_non_af.getPairContactIndexInPairObject()].getlocalBodyIndex() != non_affordance_id) continue;
                    contacts_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = 1;
                    impulses_r_non_af[contactMapping_r_[contact_non_af.getlocalBodyIndex()]] = contact_non_af.getImpulse().norm();
                }
    
                for(int i=0; i<num_contacts; i++){
                    if (contacts_r_non_af[i] == 1){
                        contacts_r_non_af[i] = contacts_r_non_af[i] - contacts_r_af[i];
                        impulses_r_non_af[i] = impulses_r_non_af[i] - impulses_r_af[i];
                    }
                }
    
                if (has_non_aff){
                }
                else{
                    contacts_r_non_af.setZero();
                    impulses_r_non_af.setZero();
                }
    
                right_hand_torque = (pTarget_clipped_r - gc_r_);
    
                raisim::Vec<3> obj_pos_w, afford_center_w, wrist_pos_w;
                raisim::Mat<3,3> obj_rot_w;
                Eigen::Vector3d target_center, target_center_wrist, target_center_dif;
                arctic->getPosition(affordance_id, obj_pos_w);
                arctic->getOrientation(affordance_id, obj_rot_w);
                raisim::matvecmul(obj_rot_w, afford_center, afford_center_w);
                mano_r_->getFramePosition(body_parts_r_[0], wrist_pos_w);
                target_center[0] = afford_center_w[0] + obj_pos_w[0] - wrist_pos_w[0];
                target_center[1] = afford_center_w[1] + obj_pos_w[1] - wrist_pos_w[1];
                target_center[2] = afford_center_w[2] + obj_pos_w[2] - wrist_pos_w[2];
                target_center_wrist = wrist_mat_r.e().transpose() * target_center;
                target_center_dif = target_center_wrist - hand_center;
    
                raisim::Mat<3,3> target_wrist_in_obj, target_wrist_in_world, target_wrist_in_wrist;
                raisim::matmul(obj_rot_w, wrist_mat_r_in_obj_init, target_wrist_in_world);
                raisim::matmul(init_or_r_, target_wrist_in_world, target_wrist_in_wrist);
                raisim::Vec<3> wrist_target_euler;
                raisim::RotmatToEuler(target_wrist_in_wrist, wrist_target_euler);
                obDouble_r_ << target_center_dif,  //3
                               gc_r_.segment(3, 3) - wrist_target_euler.e(), //3
                               gc_r_.tail(gcDim_ - 6), //8
                               right_hand_torque, //14
                               wrist_vel_in_wrist, //3
                               wrist_qvel_in_wrist, //3
                               contacts_r_af, //9
                               impulses_r_af, //9
                               obj_vel_in_wrist, //3
                               obj_qvel_in_wrist, //3
                               contacts_r_non_af, //9
                               impulses_r_non_af; //9
    
                raisim::transpose(Obj_orientation_temp, Obj_orientation);
    
                raisim::Mat<3,3> obj_pose_wrist_mat;
                raisim::Vec<3> obj_pose_wrist;
                raisim::matmul(wrist_mat_r_trans, Obj_orientation_temp, obj_pose_wrist_mat);
                raisim::RotmatToEuler(obj_pose_wrist_mat, obj_pose_wrist);
    
                raisim::Mat<3,3> frame_mat;
                raisim::Vec<3> frame_y_frame, frame_y_w, frame_y_o, joint_pos_w, joint_pos_o;
                frame_y_frame.setZero();
                frame_y_frame[1] = -1;
                for(int i = 0; i < num_bodyparts ; i++){
                    mano_r_->getFrameOrientation(body_parts_r_[i], frame_mat);
                    mano_r_->getFramePosition(body_parts_r_[i], joint_pos_w);
                    raisim::matvecmul(frame_mat, frame_y_frame, frame_y_w);
                    raisim::matvecmul(Obj_orientation, frame_y_w, frame_y_o);
                    joint_pos_in_world[i * 3] = joint_pos_w[0];
                    joint_pos_in_world[i * 3 + 1] = joint_pos_w[1];
                    joint_pos_in_world[i * 3 + 2] = joint_pos_w[2];
    
                    raisim::Vec<3>  joint_pos_o_temp;
                    joint_pos_o_temp[0] = joint_pos_w[0] - Obj_Position[0];
                    joint_pos_o_temp[1] = joint_pos_w[1] - Obj_Position[1];
                    joint_pos_o_temp[2] = joint_pos_w[2] - Obj_Position[2];
                    raisim::matvecmul(Obj_orientation, joint_pos_o_temp, joint_pos_o);
    
                    frame_y_in_obj[i * 3] = frame_y_o[0];
                    frame_y_in_obj[i * 3 + 1] = frame_y_o[1];
                    frame_y_in_obj[i * 3 + 2] = frame_y_o[2];
                    joint_pos_in_obj[i * 3] = joint_pos_o[0];
                    joint_pos_in_obj[i * 3 + 1] = joint_pos_o[1];
                    joint_pos_in_obj[i * 3 + 2] = joint_pos_o[2];
                }
               Eigen::Vector3d current_across_axis_w;
    
                current_across_axis_w = wrist_mat_r.e() * across_axis_wrist;
    
               Eigen::Vector3d current_grasp_axis_w;
               current_grasp_axis_w = wrist_mat_r.e() * hand_center;
               current_grasp_axis_w = current_grasp_axis_w / current_grasp_axis_w.norm();
    
               Eigen::Vector3d current_grasp_axis_o, current_across_axis_o;
               current_grasp_axis_o = obj_rot_w.e().transpose() * current_grasp_axis_w;
               current_across_axis_o = obj_rot_w.e().transpose() * current_across_axis_w;
    
               float across_mul = 0;
               across_mul = current_across_axis_o.dot(across_axis_o);
    
    
              raisim::Vec<3> trans_r;
             raisim::Mat<3,3> rot_r;
             raisim::Vec<4> quat_r;
    
             Eigen::VectorXd obj_state;
             obj_state.setZero(8);
    
             mano_r_->getFramePosition(body_parts_r_[0], trans_r); // Position: body position of a body in mano
             mano_r_->getFrameOrientation(body_parts_r_[0], rot_r); // Body_orientation:
    
             raisim::rotMatToQuat(rot_r, quat_r);
              obj_state = arctic->getGeneralizedCoordinate().e();
    

                    global_state_ << obj_pose_wrist.e(),
                                     frame_y_in_obj,
                                     joint_pos_in_obj,
                                     Obj_Position.e(),
                                     current_grasp_axis_o - grasp_axis_o,
                                     across_mul,
                                     trans_r.e(),
                                     quat_r.e(),
                                     gc_r_.tail(gcDim_ - 6),
                                     obj_state;
   
            }
    
            /// Set observation in wrapper to current observation
            void observe(Eigen::Ref<EigenVec> ob_r, Eigen::Ref<EigenVec> ob_l) final {
                ob_r = obDouble_r_.cast<float>();
                ob_l = obDouble_l_.cast<float>();
            }
    
            void get_global_state(Eigen::Ref<EigenVec> gs) {
                gs = global_state_.cast<float>();
            }
    
    
            void set_rootguidance() final {}
            void switch_root_guidance(bool is_on) {
                lift = true;
            }
            /// Since the episode lengths are fixed, this function is used to catch instabilities in simulation and reset the env in such cases
            bool isTerminalState(float& terminalReward) final {

                    // raisim::Vec<3> obj_current_pos;
                    // auto top_id = arctic->getBodyIdx("top");
                    // arctic->getPosition(top_id, obj_current_pos);
                    // double displacement = (obj_current_pos.e() - obj_pos_init_.head(3)).squaredNorm();
                    // if (displacement > 0.1){
                    //     terminalReward = -10;
                    //     return true;
                    
                // }

                if(obDouble_r_.hasNaN() || global_state_.hasNaN()) {
                    return true;
                }

                return false;
            }
    
        private:
            int gcDim_, gvDim_, nJoints_;
            int gcDim_obj, gvDim_obj;
            bool visualizable_ = false;
            bool unseen = false;
            bool root_guided = true;
            bool new_category = false;
            bool lift = false;
            raisim::ArticulatedSystem* mano_;
            Eigen::VectorXd gc_r_, gv_r_, pTarget_r_, pTarget6_r_, vTarget_r_, gc_set_r_, gv_set_r_;
            Eigen::VectorXd obj_pos_init_;
            Eigen::VectorXd joint_pos_in_world;
            std::string load_set;
    
            double affordance_contact_reward_r= 0.0;
            double not_affordance_contact_reward_r = 0.0;
            double affordance_impulse_reward_r= 0.0;
            double not_affordance_impulse_reward_r = 0.0;
            double wrist_vel_reward_r = 0.0;
            double wrist_qvel_reward_r = 0.0;
            double obj_vel_reward_r = 0.0;
            double obj_qvel_reward_r = 0.0;
            double obj_weight = 0.0;
            double mano_weight = 0.0;
            double wrist_Pgain = 100.0;
            double wrist_Dgain = 0.1;
            double rot_Pgain = 100.0;
            double rot_Dgain = 0.2;
    
    
            int num_contacts = 9;
            int num_bodyparts = 9;
    
            raisim::Mat<3,3> init_rot_r_, init_or_r_, init_obj_rot_, init_obj_or_, wrist_mat_r_in_obj_init;
            raisim::Vec<3> init_root_r_, init_obj_;
            Eigen::Vector3d init_center;
            Eigen::VectorXd joint_limit_high, joint_limit_low;
            Eigen::VectorXd impulse_high, impulse_low;
            Eigen::VectorXd actionMean_r_, actionStd_r_;
            Eigen::VectorXd obDouble_r_, obDouble_l_, global_state_;
            Eigen::VectorXd finger_weights_contact, finger_weights_aff;
            Eigen::VectorXd contacts_r_af, impulses_r_af;
            Eigen::VectorXd contacts_r_non_af, impulses_r_non_af;
            Eigen::VectorXd contact_body_idx_r_;
            Eigen::VectorXd frame_y_in_obj, joint_pos_in_obj;
            raisim::Vec<3> Position;
            raisim::Vec<3> wrist_vel, wrist_qvel;
            Eigen::Vector3d wrist_vel_in_wrist, wrist_qvel_in_wrist;
            Eigen::VectorXd right_hand_torque, right_wrist_torque;
            Eigen::VectorXd pTarget_clipped_r;
            Eigen::Vector3d hand_center, afford_center, wrist_target_o, obj_center_o;
            Eigen::Vector3d grasp_axis_o, across_axis_o, across_axis_wrist;
    
            Eigen::VectorXd last_closest_points;
            Eigen::VectorXd joints_pos;
    
            raisim::Mesh *obj_mesh_1, *obj_mesh_2, *obj_mesh_3, *obj_mesh_4;
            raisim::Box *box;
            raisim::ArticulatedSystem *arctic, *mano_r_;
            raisim::ArticulatedSystemVisual *arcticVisual;
            raisim::Mat<3,3> Obj_orientation, Obj_orientation_temp, Obj_orientation_init;
            raisim::Vec<4> obj_quat;
            raisim::Vec<3> Obj_Position, Obj_Position_init, Obj_qvel, Obj_linvel;
            Eigen::Vector3d obj_vel_in_wrist, obj_qvel_in_wrist;
            bool first_reset_=true;
            float rewards_sum_[2];
            bool has_non_aff = false;
            double hand_mass = 0.0;
    

                // NOTE: body_parts_r_ uses joint names (NOT link names)
                std::string body_parts_r_[9] = {
                    "z_rotation_joint",
                    "finger_1_prox_joint", "finger_1_med_joint", "finger_1_dist_joint",
                    "finger_2_prox_joint", "finger_2_med_joint", "finger_2_dist_joint",
                    "finger_3_med_joint", "finger_3_dist_joint",
                };

                // contact_bodies_r_ uses link(body) names
                std::string contact_bodies_r_[9] = {
                    "base_link",
                    "finger_1_prox_link", "finger_1_med_liink", "finger_1_dist_link",
                    "finger_2_prox_link", "finger_2_med_link", "finger_2_dist_link",
                    "finger_3_med_link", "finger_3_dist_link",
                };
    
    
            Eigen::VectorXd mean_pose;
    
            raisim::PolyLine *lines[9];
            raisim::Visuals *table_top, *leg1,*leg2,*leg3,*leg4, *plane;
            raisim::Visuals *Cylinder[9];
            raisim::Visuals *sphere[9];
            raisim::Visuals *joints_sphere[9];
            raisim::Visuals *aff_center_visual[7];
            raisim::Visuals *wrist_target[2];
    
            std::map<int,int> contactMapping_r_;
            std::map<int,int> contactMapping_l_;
            std::string resourceDir_;
            std::vector<raisim::Vec<2>> joint_limits_;
            raisim::PolyLine *line;
    };
}
