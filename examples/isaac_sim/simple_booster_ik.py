try:
    import isaacsim
except ImportError:
    pass

import torch
import numpy as np
import sys
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False, "width": "1920", "height": "1080"})

import carb
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from helper import add_extensions, add_robot_to_scene

def main():
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    
    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "booster_t1_left_arm.yml"))["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world, position=np.array([0.0, 0.0, 0.7]))
    
    dummy_cuboid = Cuboid(name="dummy", dims=[0.01, 0.01, 0.01], pose=[10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0])
    world_cfg = WorldConfig(cuboid=[dummy_cuboid])
    
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        num_ik_seeds=400,
        position_threshold=0.02,
        rotation_threshold=0.50,
        self_collision_check=True,
    )
    motion_gen = MotionGen(motion_gen_config)
    
    add_extensions(simulation_app, None)
    
    reach_vec = tensor_args.to_device([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    pose_metric = PoseCostMetric(reach_partial_pose=True, reach_vec_weight=reach_vec)
    
    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        max_attempts=4,
        partial_ik_opt=True,
        timeout=10.0,
    )
    
    my_world.scene.add_default_ground_plane()
    
    retract_joint_state = JointState.from_position(
        tensor_args.to_device(default_config).unsqueeze(0),
        joint_names=j_names
    )
    retract_joint_state = retract_joint_state.get_ordered_joint_state(motion_gen.kinematics.joint_names)
    kin_state = motion_gen.compute_kinematics(retract_joint_state)
    retract_ee_pos = kin_state.ee_pos_seq[0].cpu().numpy()
    print(f"Retract end-effector position: {retract_ee_pos}")
    
    target_positions = [
        retract_ee_pos + np.array([0.0, 0.15, 0.1]),   # Forward and up
        retract_ee_pos + np.array([0.0, 0.25, 0.2]),   # More forward, higher
        retract_ee_pos + np.array([0.0, 0.2, 0.0]),    # Forward, same height as retract
    ]
    
    current_target_idx = 0
    cmd_plan = None
    cmd_idx = 0
    idx_list = None
    
    i = 0
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("Click Play to start")
            i += 1
            continue
        
        step_index = my_world.current_time_step_index
        
        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot.set_joint_velocities(np.zeros(len(idx_list)), idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for _ in range(len(idx_list))]), joint_indices=idx_list
            )
            continue
        
        if step_index < 50:
            idx_list = [robot.get_dof_index(x) for x in j_names]
            art_action = ArticulationAction(
                np.array(default_config),
                np.zeros(len(default_config)),
                joint_indices=idx_list,
            )
            robot.get_articulation_controller().apply_action(art_action)
            continue
        
        if cmd_plan is not None and cmd_idx < len(cmd_plan.position):
            cmd_state = cmd_plan[cmd_idx]
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            robot.get_articulation_controller().apply_action(art_action)
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            continue
        
        if current_target_idx < len(target_positions) + 1:
            sim_js = robot.get_joints_state()
            cu_js = JointState(
                position=tensor_args.to_device(sim_js.positions),
                velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=robot.dof_names,
            )
            cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
            
            if current_target_idx < len(target_positions):
                target_pos = target_positions[current_target_idx]
                print(f"Planning to target {current_target_idx + 1}/4: {target_pos}")
                
                ik_goal = Pose(
                    position=tensor_args.to_device(target_pos),
                    quaternion=tensor_args.to_device([1.0, 0.0, 0.0, 0.0]),
                )
                
                plan_config.pose_cost_metric = pose_metric
                result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
                
                if result.success.item():
                    print(f"SUCCESS: Plan found for target {current_target_idx + 1}")
                    cmd_plan = result.get_interpolated_plan()
                    cmd_plan = motion_gen.get_full_js(cmd_plan)
                    idx_list = []
                    common_js_names = []
                    for x in robot.dof_names:
                        if x in cmd_plan.joint_names:
                            idx_list.append(robot.get_dof_index(x))
                            common_js_names.append(x)
                    cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)
                    cmd_idx = 0
                    current_target_idx += 1
                else:
                    print(f"FAILED: {result.status} for target {current_target_idx + 1}")
                    print(f"Target: {target_pos}")
                    if result.position_error is not None:
                        print(f"Position error: {result.position_error.item():.4f}m")
                    current_target_idx += 1
            else:
                print(f"Moving to retract config (target 4/4)")
                retract_plan = JointState(
                    position=retract_joint_state.position,
                    velocity=tensor_args.to_device([0.0] * len(default_config)).unsqueeze(0),
                    acceleration=tensor_args.to_device([0.0] * len(default_config)).unsqueeze(0),
                    jerk=tensor_args.to_device([0.0] * len(default_config)).unsqueeze(0),
                    joint_names=motion_gen.kinematics.joint_names,
                )
                retract_plan = motion_gen.get_full_js(retract_plan)
                idx_list = []
                common_js_names = []
                for x in robot.dof_names:
                    if x in retract_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                retract_plan = retract_plan.get_ordered_joint_state(common_js_names)
                
                cmd_plan = retract_plan
                cmd_idx = 0
                current_target_idx += 1
    
    simulation_app.close()

if __name__ == "__main__":
    main()

