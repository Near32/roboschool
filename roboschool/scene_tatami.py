import os
from roboschool.scene_abstract import Scene, cpp_household

class TatamiScene(Scene):
    stadium_halflen   = 105*0.25    # FOOBALL_FIELD_HALFLEN
    stadium_halfwidth = 50*0.25     # FOOBALL_FIELD_HALFWID

    def episode_restart(self):
        Scene.episode_restart(self)   # contains cpp_world.clean_everything()
        tatami_pose = cpp_household.Pose()
        '''
        TODO: make a tatami scene, or just leave it with the stadium as is...
        '''
        self.tatami = self.cpp_world.load_thingy(
            os.path.join(os.path.dirname(__file__), "models_outdoor/stadium/stadium1.obj"),
            tatami_pose, 1.0, 0, 0xFFFFFF, True)
        self.ground_plane_mjcf = self.cpp_world.load_mjcf(os.path.join(os.path.dirname(__file__), "mujoco_assets/ground_plane.xml"))

class SinglePlayerTatamiScene(TatamiScene):
    "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
    multiplayer = False

class MultiplayerTatamiScene(TatamiScene):
    multiplayer = True
    players_count = 2
    def actor_introduce(self, robot):
        TatamiScene.actor_introduce(self, robot)
        i = robot.player_n - 1  # 0 1 2 => -1 0 +1
        robot.move_robot(0, i, 0)
