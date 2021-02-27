import mcts_rave_CNN as mcts
from mcts_rave_CNN import *
from CNN_connector import *

mcts_iter = 100

# ----- MAIN -----

@timer
def play_game():

    # Optimization for turn 1
    sign = 'X'
    last_move = (4, 4)
    apply_move(mcts.game, mcts.mini_game, last_move, 'X')
    print(f"'{sign}' apply {last_move}")

    sign = 'O'
    i = 0
    for k in range(mcts_iter):
        reset_state()
        MCTS(last_move, sign=sign)
        i += 1
    print(f"MCTS iter {i}", file=sys.stderr, flush=True)

    print_board(mcts.game, mcts.mini_game)

    # turn_time = 0.990
    turn_time = 0.090
    while True:

        i = 0
        # while time.time() - begin_time < turn_time:
        for k in range(mcts_iter):
            reset_state()

            # print(f"MONTE CARLO BEGIN {i}", file=sys.stderr, flush=True)
            MCTS(last_move, sign=sign)
            # print(f"MONTE CARLO END {i}", file=sys.stderr, flush=True)
            i += 1

        print(f"MCTS\tlast_move {last_move}\tnbr iter {i}", file=sys.stderr, flush=True)

        last_move = print_best_move(last_move, sign)

        print_board(mcts.game, mcts.mini_game) # For console tests
        
        winner = is_win()
        if winner or all([all([mini_game[tmpy][tmpx] != ' ' for tmpx in range(3)]) for tmpy in range(3)]):
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



for k in range(2):

    init_mcts()
    cross_win = play_game()

    # First player = X but first states/qualities save is O
    win = [-cross_win if i % 2 == 0 else cross_win for i, state in enumerate(mcts.states)]

    print(mcts.states)
    print(mcts.qualities)
    print(mcts.signs)
    print(win)
    print(f"cross_win: {cross_win}")

    with open(f"datasets/data_{k}", 'a') as f:
        f.write(str(mcts.states))
        f.write('\n')
        f.write(str(mcts.qualities))
        f.write('\n')
        f.write(str(win))
        f.write('\n')

    fit(mcts.states, [mcts.qualities, win])
