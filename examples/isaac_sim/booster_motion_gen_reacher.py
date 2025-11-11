#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Modified for Booster T1 Humanoid Robot - spawns robot at correct height
try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")  # this is necessary to allow isaac sim to use this torch instance

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="booster_t1_left_arm.yml", help="robot configuration to load")
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

parser.add_argument(
    "--reach_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Reach partial pose",
    type=float,
    default=None,
)
parser.add_argument(
    "--hold_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Hold partial pose while moving to goal",
    type=float,
    default=None,
)

parser.add_argument(
    "--robot_height",
    type=float,
    default=0.7,
    help="Height at which to spawn robot trunk (humanoid standing height, feet on ground at z=-0.02)",
)


args = parser.parse_args()

############################################################


# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

def main():
    num_targets = 0
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    # Based on FK: retract pose is at [0.062, 0.156, -0.206] relative to base
    # With base at 0.7m, hand is at ~0.5m world height
    # Target should be in front, to the left, at similar height or slightly above
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.3, 0.3, 0.85]),  # 30cm forward, 30cm left, 0.85m world height (0.15m above base)
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    # MODIFIED: Spawn robot at specified height (default 0.7 for Booster T1)
    print(f"[DEBUG] Spawning robot at height {args.robot_height}...")
    robot, robot_prim_path = add_robot_to_scene(
        robot_cfg, my_world, position=np.array([0.0, 0.0, args.robot_height])
    )
    print(f"[DEBUG] Robot spawned at {robot_prim_path}", file=sys.stdout, flush=True)

    articulation_controller = None

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    enable_finetune_trajopt = True
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
        enable_finetune_trajopt = False
    
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        # num_ik_seeds=400,  # Increased from 200 for better IK success rate
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
        position_threshold=0.02,
        rotation_threshold=0.50,  # Large threshold to allow relaxed orientation matching
    )
    motion_gen = MotionGen(motion_gen_config)
    
    retract_state = JointState.from_position(
        tensor_args.to_device(default_config).unsqueeze(0),
        joint_names=j_names
    )
    retract_state = retract_state.get_ordered_joint_state(motion_gen.kinematics.joint_names)
    kin_state = motion_gen.compute_kinematics(retract_state)
    retract_ee_pos = kin_state.ee_pos_seq.cpu().numpy()[0]
    retract_ee_quat = kin_state.ee_quat_seq.cpu().numpy()[0]
    
    print(f"[DEBUG] Retract end-effector position (base frame): {retract_ee_pos}", file=sys.stderr, flush=True)
    print(f"[DEBUG] Retract end-effector orientation: {retract_ee_quat}", file=sys.stderr, flush=True)
    
    # Default end-effector orientation: Use retract orientation as starting point
    # With partial_ik_opt=True, the solver can relax orientation constraints
    # This is better than a fixed orientation that might not match the robot's reachable space
    default_ee_orientation = retract_ee_quat.copy()
    
    if not args.reactive:
        motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False)

    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,  # Enable graph planning after 3 failed attempts
        max_attempts=max_attempts, # 4
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5 if not args.reactive else 1.0,
        # partial_ik_opt=True,  # Allow partial IK - helps when robot is in difficult configurations
        # num_ik_seeds=None,  # Use default from MotionGenConfig (400 seeds)
        timeout=15.0,  # Give more time for difficult IK problems
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    cmd_plan = None
    cmd_idx = 0
    idx_list = None  # Initialize idx_list to None, will be set when plan is created
    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    past_cmd = None
    target_orientation = None
    past_orientation = None
    # reach_vec: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
    # Set rotation weights to 0.0 for position-only IK - orientation will be ignored
    # This allows IK to succeed even when exact orientation match is impossible
    reach_vec = tensor_args.to_device([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # Position-only IK
    pose_metric = PoseCostMetric(reach_partial_pose=True, reach_vec_weight=reach_vec)
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****", file=sys.stderr, flush=True)
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        if step_index < 10:
            robot._articulation_view.initialize()
            # Set controlled left arm joints
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            print(f"[INIT] Setting initial joint positions:", file=sys.stderr, flush=True)
            print(f"  Controlled joints: {j_names}", file=sys.stderr, flush=True)
            print(f"  Controlled config: {default_config}", file=sys.stderr, flush=True)

            # Set high effort limits for controlled joints
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue
        
        # Hold position for a few more steps to ensure robot is fully settled
        if step_index < 50:
            # Hold controlled joints
            idx_list = [robot.get_dof_index(x) for x in j_names]
            art_action = ArticulationAction(
                np.array(default_config),
                np.zeros(len(default_config)),
                joint_indices=idx_list,
            )
            articulation_controller.apply_action(art_action)
            
            if step_index == 49:
                print(f"[INIT] Initialization complete, robot should be static now", file=sys.stderr, flush=True)
            continue

        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path, file=sys.stderr, flush=True)
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(len(obstacles.objects), file=sys.stderr, flush=True)

            motion_gen.update_world(obstacles)
            print("Updated World", file=sys.stderr, flush=True)
            carb.log_info("Synced CuRobo world from stage.")

        cube_position, cube_orientation = target.get_world_pose()
        
        robot_base_position = np.array([0.0, 0.0, args.robot_height])
        cube_position_base_frame = cube_position - robot_base_position

        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None", file=sys.stderr, flush=True)
            continue
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )

        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0

        if args.reactive and past_cmd is not None:
            # past_cmd only contains controlled joints, so we need to update only those joints in cu_js
            # Create a mapping from past_cmd joint names to cu_js indices
            for i, jname in enumerate(past_cmd.joint_names):
                if jname in cu_js.joint_names:
                    idx = cu_js.joint_names.index(jname)
                    cu_js.position[idx] = past_cmd.position[i]
                    cu_js.velocity[idx] = past_cmd.velocity[i]
                    cu_js.acceleration[idx] = past_cmd.acceleration[i]
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
            
            # Get robot base position in world frame to offset spheres
            robot_position, _ = robot.get_world_pose()

            if spheres is None:
                spheres = []
                # create spheres:

                for si, s in enumerate(sph_list[0]):
                    sphere_world_pos = np.ravel(s.position) + robot_position
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=sphere_world_pos,
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        sphere_world_pos = np.ravel(s.position) + robot_position
                        spheres[si].set_world_pose(position=sphere_world_pos)
                        spheres[si].set_radius(float(s.radius))

        robot_static = False
        controlled_joint_indices = [robot.dof_names.index(jname) for jname in j_names if jname in robot.dof_names]
        controlled_joint_velocities = [sim_js.velocities[i] for i in controlled_joint_indices]
        if (np.max(np.abs(controlled_joint_velocities)) < 0.5) or args.reactive:
            robot_static = True
        
        if (
            (
                np.linalg.norm(cube_position - target_pose) > 1e-3
                or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
            )
            and np.linalg.norm(past_pose - cube_position) == 0.0
            and np.linalg.norm(past_orientation - cube_orientation) == 0.0
            and robot_static
        ):
            ee_translation_goal = cube_position_base_frame
            
            # Check if target is too far from retract position
            dist_from_retract = np.linalg.norm(ee_translation_goal - retract_ee_pos)
            print(f"[DEBUG] Target distance from retract: {dist_from_retract:.3f}m", file=sys.stderr, flush=True)
            
            # Check start state validity BEFORE planning to get detailed collision info
            valid_start, start_status = motion_gen.check_start_state(cu_js.unsqueeze(0))
            if not valid_start:
                print(f"[COLLISION_DEBUG] Start state invalid BEFORE planning: {start_status}", file=sys.stderr, flush=True)
                print(f"[COLLISION_DEBUG] Start joint state: {cu_js.position.cpu().numpy()}", file=sys.stderr, flush=True)
                # Try to continue anyway - sometimes collision detection is conservative
                # But log the issue for debugging
            
            # Use retract orientation as starting point - partial_ik_opt=True and 
            # reach_partial_pose=True with relaxed rotation weights allow the solver
            # to find a reachable orientation even if exact match isn't possible
            ee_orientation_teleop_goal = default_ee_orientation

            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            plan_config.pose_cost_metric = pose_metric
            import time as time_module
            plan_start_time = time_module.time()
            print(f"[PLAN_DEBUG] Starting planning to target: {ee_translation_goal}", file=sys.stderr, flush=True)
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            plan_time = time_module.time() - plan_start_time
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()
            print(f"[PLAN_DEBUG] Planning completed in {plan_time:.2f}s: success={succ}, status={result.status}, valid_query={result.valid_query}", file=sys.stderr, flush=True)
            if result.position_error is not None:
                print(f"[PLAN_DEBUG] Position error: {result.position_error.item():.4f}m", file=sys.stderr, flush=True)
            if result.rotation_error is not None:
                print(f"[PLAN_DEBUG] Rotation error: {result.rotation_error.item():.4f}rad", file=sys.stderr, flush=True)
            if not succ:
                # Debug: print IK failure details
                pos_err = result.position_error.item() if result.position_error is not None else None
                rot_err = result.rotation_error.item() if result.rotation_error is not None else None
                print(f"[IK_DEBUG] Plan failed. Status: {result.status}", file=sys.stderr, flush=True)
                print(f"[IK_DEBUG] Valid query: {result.valid_query}", file=sys.stderr, flush=True)
                if pos_err is not None:
                    print(f"[IK_DEBUG] Position error: {pos_err:.4f}m", file=sys.stderr, flush=True)
                if rot_err is not None:
                    print(f"[IK_DEBUG] Rotation error: {rot_err:.4f}rad", file=sys.stderr, flush=True)
                print(f"[IK_DEBUG] Target position (base frame): {ee_translation_goal}", file=sys.stderr, flush=True)
                print(f"[IK_DEBUG] Current joint state: {cu_js.position.cpu().numpy()}", file=sys.stderr, flush=True)
                
                # Check if start state is valid
                valid_start, start_status = motion_gen.check_start_state(cu_js.unsqueeze(0))
                if not valid_start:
                    print(f"[IK_DEBUG] Start state invalid: {start_status}", file=sys.stderr, flush=True)
            if num_targets == 1:
                if args.constrain_grasp_approach:
                    pose_metric = PoseCostMetric.create_grasp_approach_metric()
                if args.reach_partial_pose is not None:
                    reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
                    pose_metric = PoseCostMetric(
                        reach_partial_pose=True, reach_vec_weight=reach_vec
                    )
                if args.hold_partial_pose is not None:
                    hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
                    pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
            if succ:
                print(f"[PLAN_DEBUG] Plan SUCCESS! Getting interpolated plan...", file=sys.stderr, flush=True)
                print(f"[PLAN_DEBUG] Target pose: position={ik_goal.position.cpu().numpy()}, quaternion={ik_goal.quaternion.cpu().numpy()}", file=sys.stderr, flush=True)
                print(f"[PLAN_DEBUG] Position error from result: {result.position_error.item() if result.position_error is not None else 'None'}m", file=sys.stderr, flush=True)
                
                # Check optimized plan first
                if result.optimized_plan is not None:
                    opt_plan = result.optimized_plan
                    if len(opt_plan.position) > 0:
                        opt_first = opt_plan.position[0].cpu().numpy()
                        opt_last = opt_plan.position[-1].cpu().numpy()
                        print(f"[PLAN_DEBUG] Optimized plan: {len(opt_plan.position)} steps", file=sys.stderr, flush=True)
                        print(f"[PLAN_DEBUG] Optimized first joints: {opt_first}", file=sys.stderr, flush=True)
                        print(f"[PLAN_DEBUG] Optimized last joints: {opt_last}", file=sys.stderr, flush=True)
                
                num_targets += 1
                cmd_plan_full = result.get_interpolated_plan()
                cmd_plan_full = motion_gen.get_full_js(cmd_plan_full)
                
                # Verify end-effector actually moves - use full plan with all joints BEFORE filtering
                if len(cmd_plan_full.position) > 0:
                    first_js_full_plan = cmd_plan_full[0]
                    last_js_full_plan = cmd_plan_full[-1]
                    first_kin_full = motion_gen.compute_kinematics(first_js_full_plan.unsqueeze(0))
                    last_kin_full = motion_gen.compute_kinematics(last_js_full_plan.unsqueeze(0))
                    first_ee_pos_full = first_kin_full.ee_pos_seq[0].cpu().numpy()
                    last_ee_pos_full = last_kin_full.ee_pos_seq[0].cpu().numpy()
                    ee_movement_full = np.linalg.norm(last_ee_pos_full - first_ee_pos_full)
                    target_ee_pos = ik_goal.position.cpu().numpy()
                    target_dist_first = np.linalg.norm(first_ee_pos_full - target_ee_pos)
                    target_dist_last = np.linalg.norm(last_ee_pos_full - target_ee_pos)
                    print(f"[PLAN_DEBUG] EE position first (full plan): {first_ee_pos_full}", file=sys.stderr, flush=True)
                    print(f"[PLAN_DEBUG] EE position last (full plan): {last_ee_pos_full}", file=sys.stderr, flush=True)
                    print(f"[PLAN_DEBUG] Target EE position: {target_ee_pos}", file=sys.stderr, flush=True)
                    print(f"[PLAN_DEBUG] EE movement distance: {ee_movement_full:.4f}m", file=sys.stderr, flush=True)
                    print(f"[PLAN_DEBUG] Distance to target (first): {target_dist_first:.4f}m", file=sys.stderr, flush=True)
                    print(f"[PLAN_DEBUG] Distance to target (last): {target_dist_last:.4f}m", file=sys.stderr, flush=True)
                
                # Only use controlled joints (left arm joints) for execution
                idx_list = []
                common_js_names = []
                for x in j_names:  # Use j_names (controlled joints) instead of sim_js_names (all joints)
                    if x in cmd_plan_full.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)

                cmd_plan = cmd_plan_full.get_ordered_joint_state(common_js_names)
                cmd_idx = 0
                print(f"[PLAN_DEBUG] Plan ready: {len(cmd_plan.position)} steps, controlled joints: {common_js_names}", file=sys.stderr, flush=True)
                print(f"[PLAN_DEBUG] Joint indices: {idx_list}", file=sys.stderr, flush=True)
                if len(cmd_plan.position) > 0:
                    first_joints = cmd_plan.position[0].cpu().numpy()
                    last_joints = cmd_plan.position[-1].cpu().numpy()
                    joint_diff = last_joints - first_joints
                    print(f"[PLAN_DEBUG] First step joint values (controlled only): {first_joints}", file=sys.stderr, flush=True)
                    print(f"[PLAN_DEBUG] Last step joint values (controlled only): {last_joints}", file=sys.stderr, flush=True)
                    print(f"[PLAN_DEBUG] Joint change from first to last: {joint_diff}", file=sys.stderr, flush=True)
                    print(f"[PLAN_DEBUG] Max joint change: {np.max(np.abs(joint_diff)):.6f} rad", file=sys.stderr, flush=True)

            else:
                carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            target_pose = cube_position
            target_orientation = cube_orientation
        
        past_pose = cube_position
        past_orientation = cube_orientation
        if cmd_plan is not None and idx_list is not None and len(idx_list) > 0:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            # get full dof state
            joint_positions = cmd_state.position.cpu().numpy()
            joint_velocities = cmd_state.velocity.cpu().numpy()
            
            # Debug: compare current state with command
            if cmd_idx == 0 or cmd_idx == len(cmd_plan.position) - 1 or step_index % 100 == 0:
                current_joint_positions = [sim_js.positions[robot.get_dof_index(j)] for j in j_names if j in robot.dof_names]
                print(f"[EXEC_DEBUG] Step {cmd_idx}/{len(cmd_plan.position)}", file=sys.stderr, flush=True)
                print(f"[EXEC_DEBUG] Current joints: {current_joint_positions}", file=sys.stderr, flush=True)
                print(f"[EXEC_DEBUG] Command joints: {joint_positions}", file=sys.stderr, flush=True)
                print(f"[EXEC_DEBUG] Joint diff: {np.array(joint_positions) - np.array(current_joint_positions)}", file=sys.stderr, flush=True)
            
            art_action = ArticulationAction(
                joint_positions,
                joint_velocities,
                joint_indices=idx_list,
            )
            # set desired joint angles obtained from IK:
            articulation_controller.apply_action(art_action)
            cmd_idx += 1
            # Step multiple times per command to allow robot to follow
            for _ in range(5):  # Increased from 2 to 5 to give more time
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                print(f"[EXEC_DEBUG] Plan execution complete!", file=sys.stderr, flush=True)
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
                idx_list = None
        elif step_index % 500 == 0:
            print(f"[EXEC_DEBUG] No plan to execute (cmd_plan is None)", file=sys.stderr, flush=True)
    simulation_app.close()


if __name__ == "__main__":
    print("[TRACE 5] About to call main()...", file=sys.stderr, flush=True)
    main()
    print("[TRACE 6] main() completed", file=sys.stderr, flush=True)

