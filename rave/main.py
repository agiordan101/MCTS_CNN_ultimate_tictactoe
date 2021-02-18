from mcts_rave import *

# ----- MAIN -----

print_board(game, mini_game) # For console tests

first_move = parsing()
turn_time = 0.990

if first_move[0] == -1:

    sign = 'X'
    sign_opp = 'O'
    last_move = (4, 4)
    apply_move(game, mini_game, last_move, sign)

    # print(f"current time -> {time.time() - begin_time}", file=sys.stderr, flush=True)
    i = 0
    # while time.time() - begin_time < turn_time:
    for k in range(600):
        reset_state()
        MCTS(last_move, sign=sign_opp)
        i += 1
    print(f"MCTS iter {i}", file=sys.stderr, flush=True)

    print("Apply my move -> 4 4")
    print_board(game, mini_game)

    last_move = parsing(sign_opp)
    turn_time = 0.090

else:
    sign = 'O'
    sign_opp = 'X'
    last_move = first_move


while True:

    # print(f"current time -> {time.time() - begin_time}", file=sys.stderr, flush=True)
    i = 0
    # while time.time() - begin_time < turn_time:
    for k in range(600):
        reset_state()

        # print(f"MONTE CARLO BEGIN {i}", file=sys.stderr, flush=True)
        MCTS(last_move, sign=sign)
        # print(f"MONTE CARLO END {i}", file=sys.stderr, flush=True)
        i += 1

    print(f"MCTS iter {i}", file=sys.stderr, flush=True)

    reset_state()
    print_best_move(last_move, sign)

    last_move = parsing(sign_opp)
    turn_time = 0.090
