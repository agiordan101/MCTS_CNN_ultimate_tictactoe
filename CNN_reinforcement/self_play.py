from mcts_rave_CNN import *

# ----- MAIN -----

def play_game():

    print_board(game, mini_game) # For console tests

    last_move = (4, 4)
    apply_move(game, mini_game, last_move, 'X')

    sign = 'O'
    i = 0
    for k in range(1000):
        reset_state()
        MCTS(last_move, sign=sign)
        i += 1
    print(f"MCTS iter {i}", file=sys.stderr, flush=True)

    print("Apply my move -> 4 4")
    print_board(game, mini_game)

    # turn_time = 0.990
    turn_time = 0.090
    while True:

        i = 0
        # while time.time() - begin_time < turn_time:
        for k in range(1000):
            reset_state()

            # print(f"MONTE CARLO BEGIN {i}", file=sys.stderr, flush=True)
            MCTS(last_move, sign=sign)
            # print(f"MONTE CARLO END {i}", file=sys.stderr, flush=True)
            i += 1

        print(f"MCTS\tlast_move {last_move}\tnbr iter {i}", file=sys.stderr, flush=True)

        reset_state()
        last_move = print_best_move(last_move, sign)
        
        print_board(game, mini_game) # For console tests
        
        winner = is_win()
        if winner or all([all([mini_game[tmpy][tmpx] != ' ' for tmpx in range(3)]) for tmpy in range(3)]):
            if winner:
                print(f"WINNER IS {winner}")
            else:
                print(f"- DRAW -")
            # print(f"Percentage escape get_moves_id() -> {100 * stat_1 / (stat_1 + stat_2)} %")
            # print(f"Percentage save get_moves_id() -> {100 * stat_3 / (stat_2)} %")
            # print(f"Values -> {stat_1}\t{stat_2}\t{stat_3}")
            exit()

        sign = 'X' if sign == 'O' else 'O'
        # turn_time = 0.095




play_game()

print(gameT)
print(qualities)
