python scripts/run_trained_agent.py --agent 'checkpoints/baseline_bctrans_model_epoch_950.pth' --bddl_file 'scripts/bddl_files/KITCHEN_SCENE9_SUBGOAL_eval-task-1_turn_on_stove.bddl' --video_prompt 'datasets/eval-task-1_turn_on_stove_put_pan_on_stove_put_bowl_on_shelf/image_demo.hdf5' --video_path 'task-1_eval_bctrans_subgoal.mp4'
python scripts/run_trained_agent.py --agent 'checkpoints/baseline_bctrans_model_epoch_950.pth' --bddl_file 'scripts/bddl_files/KITCHEN_SCENE9_SUBGOAL_eval-task-2_put_bowl_on_stove.bddl' --video_prompt 'datasets/eval-task-2_put_bowl_on_stove_turn_on_stove_put_pan_in_shelf/image_demo.hdf5' --video_path 'task-2_eval_bctrans_subgoal.mp4'
python scripts/run_trained_agent.py --agent 'checkpoints/baseline_bctrans_model_epoch_950.pth' --bddl_file 'scripts/bddl_files/KITCHEN_SCENE9_SUBGOAL_eval-task-3_put_bowl_on_shelf.bddl' --video_prompt 'datasets/eval-task-3_put_bowl_on_shelf_put_pan_in_shelf/image_demo.hdf5' --video_path 'task-3_eval_bctrans_subgoal.mp4'
python scripts/run_trained_agent.py --agent 'checkpoints/baseline_bctrans_model_epoch_950.pth' --bddl_file 'scripts/bddl_files/KITCHEN_SCENE9_SUBGOAL_eval-task-4_put_bowl_on_stove.bddl' --video_prompt 'datasets/eval-task-4_put_bowl_on_stove_put_pan_in_shelf/image_demo.hdf5' --video_path 'task-4_eval_bctrans_subgoal.mp4'
python scripts/run_trained_agent.py --agent 'checkpoints/baseline_bctrans_model_epoch_950.pth' --bddl_file 'scripts/bddl_files/KITCHEN_SCENE9_SUBGOAL_eval-task-5_put_pan_on_stove.bddl' --video_prompt 'datasets/eval-task-5_put_pan_on_stove_put_bowl_on_shelf/image_demo.hdf5' --video_path 'task-5_eval_bctrans_subgoal.mp4'