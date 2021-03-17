from CNN_connector import *

import mcts_rave_CNN as mcts
from mcts_rave_CNN import *
import matplotlib.pyplot as plt

mcts_iter = 600
n_game = 1

# ----- MAIN -----

@timer
def play_game(model):

    mcts.init_mcts()

    # Optimization for turn 1
    sign = 'X'
    last_move = (4, 4)
    apply_move(mcts.game, mcts.mini_game, last_move, sign)
    print(f"'{sign}' apply {last_move}")

    sign = 'O'
    i = 0
    for k in range(mcts_iter):
        reset_state()
        MCTS(last_move, sign=sign, model=model)
        i += 1
    print(f"MCTS iter {i}", file=sys.stderr, flush=True)

    print_board(mcts.game, mcts.mini_game)

    while True:

        i = 0
        # while time.time() - begin_time < turn_time:
        for k in range(mcts_iter):
            reset_state()

            # print(f"MONTE CARLO BEGIN {i}", file=sys.stderr, flush=True)
            MCTS(last_move, sign=sign, model=model)
            # print(f"MONTE CARLO END {i}", file=sys.stderr, flush=True)
            i += 1

        print(f"MCTS\tlast_move {last_move}\tnbr iter {i}", file=sys.stderr, flush=True)

        last_move = print_best_move(last_move, sign)

        print_board(mcts.game, mcts.mini_game) # For console tests

        winner = mcts.is_win(mcts.mini_game)
        if winner or all([all([mcts.mini_game[tmpy][tmpx] != ' ' for tmpx in range(3)]) for tmpy in range(3)]):
            if winner:
                print(f"WINNER IS {winner}")
            else:
                print(f"- DRAW -")
            # print(f"Percentage escape get_moves_id() -> {100 * stat_1 / (stat_1 + stat_2)} %")
            # print(f"Percentage save get_moves_id() -> {100 * stat_3 / (stat_2)} %")
            # print(f"Values -> {stat_1}\t{stat_2}\t{stat_3}")
            return (1 if winner == 'X' else -1) if winner else 0

        sign = 'X' if sign == 'O' else 'O'
        # turn_time = 0.095


model = RLModel(name='model_night_0')

# for k in range(1, n_game + 1):
while True:

    # print(f"Play game {k}/{n_game}")
    cross_win = play_game(model)
    # cross_win = play_game(None)

    # First player = X but first states/qualities save is O
    win = [-cross_win if i % 2 == 0 else cross_win for i in range(len(mcts.states))]

    # print(f"END GAME {k}/{n_game}")
    print(np.array(mcts.states).shape)
    print(np.array(mcts.qualities).shape)
    print(mcts.signs)
    print(np.array(win).shape)
    print(f"cross_win at 0: {win}")
    print(f"Feature -1: {mcts.states[-1]}")

    # win = np.array(win)

    if cross_win:
        history = model.fit(np.array(mcts.states), [np.array(mcts.qualities), np.array(win)])

        with open("dataset_night_0.mcts_rave_CNN", 'a') as f:
            for state, quality, winning in zip(mcts.states, mcts.qualities, win):
                # [[f.write(f"{sign},") for sign in row] for row in state]
                # [[f.write(f"{sign},") for sign in row] for row in quality]
                # print(f"state {type(state)}: {state}")
                # print(f"quality {type(quality)}: {quality}")
                [f.write(f"{sign},") for sign in state.flatten().tolist()]
                [f.write(f"{sign},") for sign in quality.flatten().tolist()]
                f.write(f'{winning}\n')
            f.close()

        model.model.save(model.name)

# print([k for k, _ in history.history.items()])
# loss_curve = history.history["loss"]
# pacc_curve = history.history["p_accuracy"]
# vacc_curve = history.history["v_accuracy"]
# plt.plot(loss_curve, label="Train")
# plt.legend(loc='upper left')
# plt.title("Loss")
# plt.show()

# plt.plot(pacc_curve)
# plt.plot(vacc_curve)
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['policy', 'value'], loc='upper left')

# plt.show()
# loss, acc = model.evaluate(self.features, self.targets)
# print("Test Loss", loss)
# print("Test Accuracy", acc)
