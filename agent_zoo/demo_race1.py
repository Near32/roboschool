from OpenGL import GL
import os, sys, subprocess
import numpy as np
import gym
import roboschool

def play(env, pi, video):
    episode_n = 0
    while 1:
        episode_n += 1
        obs = env.reset()
        if video: 
            video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env=env, base_path=("/tmp/demo_race1_episode%i" % episode_n), enabled=True)
        while 1:
            a = pi.act(obs,None)
            obs, rew, done, info = env.step(a)
            if video: 
                video_recorder.capture_frame()
            if done: 
                break
        if video: 
            video_recorder.close()
            print("Video recorded :: episode {}".format(episode_n))
        break

if len(sys.argv)==1:
    import roboschool.multiplayer
    stadium = roboschool.scene_stadium.MultiplayerStadiumScene(gravity=9.8, timestep=0.0165/4, frame_skip=4)
    gameserver = roboschool.multiplayer.SharedMemoryServer(stadium, "race", want_test_window=True)
    # We start subprocesses between constructor and serve_forever(), because constructor creates necessary pipes to connect to
    for n in range(stadium.players_count):
        subprocess.Popen([sys.executable, sys.argv[0], "race", "%i"%n])
    gameserver.serve_forever()

else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1,
        device_count = { "GPU": 0 } )
    sess = tf.InteractiveSession(config=config)
    # If this gives you an error, try CUDA_VISIBLE_DEVICES=  (nothing visible)

    from RoboschoolWalker2d_v1_2017jul        import ZooPolicyTensorflow as PolWalker
    from RoboschoolHopper_v1_2017jul          import ZooPolicyTensorflow as PolHopper
    from RoboschoolHalfCheetah_v1_2017jul     import ZooPolicyTensorflow as PolHalfCheetah
    from RoboschoolHumanoid_v1_2017jul        import ZooPolicyTensorflow as PolHumanoid1
    from RoboschoolHumanoidFlagrun_v1_2017jul import ZooPolicyTensorflow as PolHumanoid2
    # Flagrun and Harder is compatible with normal Humanoid in observations and actions.

    possible_participants = [
        ("RoboschoolWalker2d-v1", PolWalker),
        ("RoboschoolHopper-v1",   PolHopper),
        ("RoboschoolHalfCheetah-v1", PolHalfCheetah),
        ("RoboschoolHumanoid-v1", PolHumanoid1),
        ("RoboschoolHumanoid-v1", PolHumanoid2),
        ]
    env_id, PolicyClass = possible_participants[ np.random.randint(len(possible_participants)) ]
    env = gym.make(env_id)
    env.unwrapped.multiplayer(env, game_server_guid=sys.argv[1], player_n=int(sys.argv[2]))

    pi = PolicyClass("mymodel", env.observation_space, env.action_space)

    if int(sys.argv[2]) == 1:
        play(env, pi, video=True)
    else :
        while 1:
            obs = env.reset()
            while 1:
                a = pi.act(obs, None)
                obs, rew, done, info = env.step(a)
                if int(sys.argv[2])==1:
                    still_open = env.render("human")
                if done: break
