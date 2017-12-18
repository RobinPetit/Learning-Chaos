
from games import Games

class Parameters:

    # environment settings
    games = Games()
    game = games.SPACE_INVADERS
    action_space = games.get_action_space(game)
    display = True
    FPS = 30

    """
    [Article] Following previous approaches to playing Atari 2600 games, we also use a simple
    frame-skipping technique. More precisely, the agent sees and selects actions on
    every kth frame instead of every frame, and its last action is repeated on skipped
    frames. Because running the emulator forward for one step requires much less
    computation than having the agent select an action, this technique allows the agent
    to play roughly k times more games without significantly increasing the runtime.
    We use k = 4 for all games
    """
    frame_skipping = 4


    # preprocessing
    image_width = 84
    image_height = 84


    # Q-learning settings
    """
    [Article] As the scale of scores varies greatly from game to game, we clipped all posi-
    tive rewards at 1 and all negative rewards at 21, leaving 0 rewards unchanged.
    Clipping the rewards in this manner limits the scale of the error derivatives and
    makes it easier to use the same learning rate across multiple games. At the same time,
    it could affect the performance of our agent since it cannot differentiate between
    rewards of different magnitude.
    """
    positive_reward = 1
    negative_reward = -1
    no_reward = 0

    """
    [Article] The function w from algorithm ϕ described below applies this preprocess-
    ing to the m most recent frames and stacks them to produce the input to the
    Q-function, in which m = 4, although the algorithm is robust to different values of
    m (for example, 3 or 5).
    """
    m_recent_frames = 4

    MAX_STEPS = 5000

