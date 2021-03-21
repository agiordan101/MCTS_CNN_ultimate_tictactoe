from mcts_rave import * 
import mcts_rave as mcts

# ----- MAIN -----
print_board(game, mini_game) # For console tests
# last_move = parsing()

# last_move = (-1, -1)
# sign = 'X' if last_move[0] == -1 else 'O'

n_iter_mcts = 1000
n_game = 1

@timer
def play_game():

    init_mcts()

    # sign = 'X'
    # last_move = (4, 4)
    # apply_move(game, mini_game, last_move, sign)

    # sign = 'O'
    # i = 0
    # for k in range(n_iter_mcts):
    #     reset_state()
    #     MCTS(last_move, sign=sign)
    #     i += 1
    # print(f"MCTS iter {i}", file=sys.stderr, flush=True)

    # print("Apply my move -> 4 4")

    sign = 'X'
    last_move = (-1, -1)
    print_board(mcts.game, mcts.mini_game)

    while True:

        i = 0
        for k in range(n_iter_mcts):
            reset_state()

            # print(f"MONTE CARLO BEGIN {i}", file=sys.stderr, flush=True)
            MCTS(last_move, sign=sign)
            # print(f"MONTE CARLO END {i}", file=sys.stderr, flush=True)
            i += 1

        print(f"MCTS\tlast_move {last_move}\tnbr iter {i}", file=sys.stderr, flush=True)

        reset_state()
        last_move = print_best_move(last_move, sign)

        print_board(mcts.game, mcts.mini_game) # For console tests

        winner = is_win(mcts.mini_game)
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


while True:
# for k in range(n_game):

    # print(f"Play game {k}/{n_game}")
    cross_win = play_game()
    # cross_win = play_game(None)

    # First player = X but first states/qualities save is O
    win = [-cross_win if i % 2 == 0 else cross_win for i in range(len(mcts.states))]

    # print(f"END GAME {k}/{n_game}")
    print(np.array(mcts.states).shape)
    print(np.array(mcts.qualities).shape)
    print(mcts.signs)
    print(np.array(win).shape)
    print(f"cross_win at 0: {win}")

    for state, mask, quality, winning in zip(mcts.states, mcts.masks, mcts.qualities, win):

        state = state.flatten()

        player = state
        opponent = state.flatten()
        player = np.where(player == 1, 1, 0).tolist()
        opponent = np.where(opponent == -1, 1, 0).tolist()

        state = state.tolist()
        mask = mask.flatten().tolist()
        quality = quality.flatten().tolist()

        print_board(mcts.game, mcts.mini_game)
        print(f"state {len(state)} / {state}\n")
        print(f"player {len(player)} / {player}\n")
        print(f"opponnent {len(opponent)} / {opponent}\n")
        print(f"mask {len(mask)} / {mask}\n")
        print(f"quality {len(quality)} / {quality}\n")
        print(f"win {winning}")

        if cross_win != 0:
            with open("dataset_mixed.csv", 'a+') as f:

                [f.write(f"{sign},") for sign in state]
                [f.write(f'{sign},') for sign in mask]
                [f.write(f"{sign},") for sign in quality]
                f.write(f'{winning}\n')
                f.close()

            with open("dataset_onehot.csv", 'a+') as f:
                [f.write(f'{sign},') for sign in player]
                [f.write(f'{sign},') for sign in opponent]
                [f.write(f'{sign},') for sign in mask]
                [f.write(f'{sign},') for sign in quality]
                f.write(f'{winning}\n')
                f.close()

        with open("dataset_mixed_draw.csv", 'a+') as f:

            [f.write(f"{sign},") for sign in state]
            [f.write(f'{sign},') for sign in mask]
            [f.write(f"{sign},") for sign in quality]
            f.write(f'{winning}\n')
            f.close()

        with open("dataset_onehot_draw.csv", 'a+') as f:

            [f.write(f'{sign},') for sign in player]
            [f.write(f'{sign},') for sign in opponent]
            [f.write(f'{sign},') for sign in mask]
            [f.write(f'{sign},') for sign in quality]
            f.write(f'{winning}\n')
            f.close()
