/*
    Fontion qui set un parent à end=True lorque tous ses enfants le sont
    Check is_win apres tous les do_action ?? Stop les expansion. choices_len=0 ?

    Timeout ! -> segfault
    tester is_win()

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#define N_ITER_MCTS 30000
#define SEC_PER_TURN 0.090
#define SEC_FIRST_TURN 0.990
#define EMPTY_CASE 95
#define MY_SIGN 88
#define OPPONENT_SIGN 79
#define DRAW_CASE 32

typedef struct	    s_node
{
    char            grid[2];
    char            sign;
	int 		    score;
	int         	visited;
	struct s_node	*childs;
	struct s_node	*next;
	unsigned char	end;
}				    t_node;

char    opponent_row;
char    opponent_col;
char    mcts_row = -1;
char    mcts_col = -1;

char    game[9][9];
char    real_game[3][3];

char    game_save[9][9];
char    real_game_save[3][3];
// int     grid = -1; //0 - 8

char    **choices;
char    choices_len = 0;

char    valid_choices[81][2];
char    valid_choices_len = 0;

clock_t start_t;
float   t;
int     i;

int         printoupas()
{
    if (i > 1000 && i < 1100)
        return 1;
    return 0;
}

/*
------------------------------------------------------------------------------------
-	Nodes functions
*/

t_node		*new_node(char grid[2], char c)
{
	// fprintf(stderr, "New node : %d %d\n", grid[0], grid[1]);

	t_node	*tmp = (t_node *)malloc(sizeof(t_node));

	bzero(tmp, sizeof(t_node));
	tmp->grid[0] = grid[0];
	tmp->grid[1] = grid[1];
    tmp->sign = c;
	return tmp;
}

int			lst_len(t_node *nodes)
{
	int		i = 1;

	if (!nodes)
		return 0;
	while (nodes->next)
	{
		i++;
		nodes = nodes->next;
	}
	return i;
}

void		print_tree(t_node *node, int depth)
{
	t_node	*tmp;

	fprintf(stderr, "Depth %d %.*s %d %d - %d/%d (Childs:%d)\n", depth, depth, "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t", node->grid[0], node->grid[1], node->score, node->visited, lst_len(node->childs));
	tmp = node->childs;
	while (tmp)
	{
		print_tree(tmp, depth + 1);
		tmp = tmp->next;
	}
}

/*
------------------------------------------------------------------------------------
-	Game functions
*/

void        print_game()
{
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            fprintf(stderr, "%c", real_game[i][j]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
    for (int i = 0; i < 9; i++)
    {
        if (i && i % 3 == 0)
            fprintf(stderr, "------------\n");
        for (int j = 0; j < 9; j++)
        {
            if (j && j % 3 == 0)
                fprintf(stderr, "|");
            fprintf(stderr, "%c", game[i][j]);
        }
        fprintf(stderr, "\n");
    }
}

char        test_grid_values(char c00, char c01, char c02, char c10, char c11, char c12, char c20, char c21, char c22)
{
    if (c11 != EMPTY_CASE &&
        ((c01 == c11 && c11 == c21) ||
        (c10 == c11 && c11 == c12) ||
        (c20 == c11 && c11 == c02)))
    {
        // fprintf(stderr, "Grid win ! Winner: %c\n", c11);   
        return c11;
    }
    if (c00 != EMPTY_CASE &&
        ((c00 == c11 && c11 == c22) ||
        (c00 == c01 && c01 == c02) ||
        (c00 == c10 && c10 == c20)))
    {
        // fprintf(stderr, "Grid win ! Winner: %c\n", c00);
        return c00;
    }
    if (c22 != EMPTY_CASE &&
        ((c02 == c12 && c12 == c22) ||
        (c20 == c21 && c21 == c22)))
    {
        // fprintf(stderr, "Grid win ! Winner: %c\n", c22);
        return c22;
    }
    if (c00 == EMPTY_CASE || c01 == EMPTY_CASE || c02 == EMPTY_CASE ||
        c10 == EMPTY_CASE || c11 == EMPTY_CASE || c12 == EMPTY_CASE ||
        c20 == EMPTY_CASE || c21 == EMPTY_CASE || c22 == EMPTY_CASE)
        return 0;
    // fprintf(stderr, "Lose\n");
    return DRAW_CASE;
}

char        is_grid_win(char y, char x)
{
    // fprintf(stderr, "grid win ? -> ");
    //Opti -> Choper l'adresse des sous tableaux
    char    grid_y = y - y % 3;
    char    grid_x = x - x  % 3;
    // char    grid_x = x * 3;
    // char    grid_y = y * 3;
    char    c00 = game[grid_y][grid_x];
    char    c01 = game[grid_y][grid_x + 1];
    char    c02 = game[grid_y][grid_x + 2];
    char    c10 = game[grid_y + 1][grid_x];
    char    c11 = game[grid_y + 1][grid_x + 1];
    char    c12 = game[grid_y + 1][grid_x + 2];
    char    c20 = game[grid_y + 2][grid_x];
    char    c21 = game[grid_y + 2][grid_x + 1];
    char    c22 = game[grid_y + 2][grid_x + 2];

    return test_grid_values(c00, c01, c02, c10, c11, c12, c20, c21, c22);
}

char        is_win()
{
    char    c00 = real_game[0][0];
    char    c01 = real_game[0][1];
    char    c02 = real_game[0][2];
    char    c10 = real_game[1][0];
    char    c11 = real_game[1][1];
    char    c12 = real_game[1][2];
    char    c20 = real_game[2][0];
    char    c21 = real_game[2][1];
    char    c22 = real_game[2][2];

    char ret = test_grid_values(c00, c01, c02, c10, c11, c12, c20, c21, c22);
    // fprintf(stderr, "GAME FINISH !!! Winner: %c\n", ret);
    return ret;
}

void        do_action(char y, char x, char c)
{
    // fprintf(stderr, "DO ACTION y/x %d %d c %c\n", y, x, c);
    game[y][x] = c;
    if (is_grid_win(y, x))
    {
        // fprintf(stderr, "Put %c in real_game at yx %d %d\n", c, y / 3, x / 3);
        real_game[y / 3][x / 3] = c;
        // print_game();
    }
}

char        get_next_grid(char last_play_y, char last_play_x)
{
    if (last_play_y == -1)
        return -1;

    char    last_grid_x = (char)(last_play_x / 3) * 3;  //Select new grid based on opponent last play
    char    last_grid_y = (char)(last_play_y / 3) * 3;
    char    new_grid_x = last_play_x - last_grid_x;
    char    new_grid_y = last_play_y - last_grid_y;

    // fprintf(stderr, "get next grid %d %d\n", new_grid_y, new_grid_x);
    if (real_game[new_grid_y][new_grid_x] == EMPTY_CASE)
        return (new_grid_y * 3 + new_grid_x);
    else
    {
        // fprintf(stderr, "-> Grid %d %d already win, ret -1\n", new_grid_y, new_grid_x);
        return -1;
    }
}

void        get_choices_grid(char grid_y, char grid_x, char last_play_y, char last_play_x)
{
    // fprintf(stderr, "Get choices grid %d %d\n", grid_y, grid_x);
    grid_x *= 3;
    grid_y *= 3;
    // fprintf(stderr, "get_choices_grid 1 yx : %d %d\n", grid_y, grid_x);
    for (char i = 0; i < 3; i++)
    {
        for (char j = 0; j < 3; j++)
        {
            // fprintf(stderr, "New choice y/x %d/%d -> '%c' ?= '%c'\n", grid_y + i, grid_x + j, game[grid_y + i][grid_x + j], EMPTY_CASE);
            if (game[grid_y + i][grid_x + j] == EMPTY_CASE &&
                (last_play_x != grid_x + j || last_play_y != grid_y + i)) //Attention ça sert à rien : Chnager le parsing avec des do_action piur enlevre ce test de merde
            {
                choices[choices_len][0] = grid_y + i;
                choices[choices_len++][1] = grid_x + j;
                // fprintf(stderr, "New choice taken y/X %d/%d -> '%c'\n", grid_y + i, grid_x + j, game[grid_y + i][grid_x + j]);
            }
        }
    }
    // fprintf(stderr, "get_choices_grid 2 yx : %d %d / n choices : %d\n", grid_y, grid_x, *choices_len);
}

void        get_choices(char last_play_y, char last_play_x)
{
    // print_game();
    char    grid = get_next_grid(last_play_y, last_play_x);

    // fprintf(stderr, "Last pos yx %d %d, new grid %d\n", last_play_y, last_play_x, grid);
    choices_len = 0;
    if (grid == -1)
    {
        for (char i = 0; i < 3; i++)
            for (char j = 0; j < 3; j++)
                if (real_game[i][j] == EMPTY_CASE)
                    get_choices_grid(i, j, last_play_y, last_play_x);
    }
    else
        get_choices_grid(grid / 3, grid % 3, last_play_y, last_play_x);
    // fprintf(stderr, "N choices %d\n", *choices_len);
}

/*
------------------------------------------------------------------------------------
-	Selection
*/

float		compute_uct(t_node *parent, t_node *child, int debug)
{
	float	score;

    // if (debug)
    //     fprintf(stderr, "compute utc 1: parent visit %d\tchild score %d\tchild visit %d\n", parent->visited, child->score, child->visited);
	if (child->visited == 0)
		return 1000000;
	score = child->score / child->visited + M_SQRT2 * sqrt(log(parent->visited) / child->visited);
    // if (debug)
    //     fprintf(stderr, "compute utc 2: score = %f\tvisit = %d\n", score, parent->visited);
	return score;
}

t_node		*select_node_uct(t_node *node, int debug)
{
	// fprintf(stderr, "utc child addr %p\n", node->childs);
	t_node			*childs = node->childs;
	t_node			*best_nodes[81];
	int         	i_best_nodes = 0;
	float			uct;
	float			best_uct = 0;
	int				r;

    // if (debug)
    //     fprintf(stderr, "Node + childs %p %p\n", node, childs);
	while (childs)
	{
		if ((uct = compute_uct(node, childs, debug)) >= best_uct)
		{
			if (uct != best_uct)
			{
                // if (debug)
                //     fprintf(stderr, "Child new best value %f\n", uct);
				bzero(best_nodes, sizeof(t_node *) * 81);
				i_best_nodes = 0;
				best_uct = uct;
			}
			// else
            //     if (debug)
            //         fprintf(stderr, "Child same best value %f\n", uct);
			best_nodes[i_best_nodes++] = childs;
		}
		childs = childs->next;
	}

	r = rand() % i_best_nodes;
    // if (debug)
    //     fprintf(stderr, "select_node_uct: %u nodes\tbest i %d\tgrid yx %d %d\taddr %p\n", i_best_nodes, r, best_nodes[r]->grid[0], best_nodes[r]->grid[1], best_nodes[r]);
	return best_nodes[r];
}

/*
------------------------------------------------------------------------------------
-	Simulation
*/

char	    simulation(char last_y, char last_x, char c, int depth)
{
    char    winner;

	// fprintf(stderr, "Game simulation %d\n", depth);
    
    get_choices(last_y, last_x);
    if (choices_len == 0)
        return DRAW_CASE;
    char r = rand() % choices_len;

    do_action(choices[r][0], choices[r][1], c);

    if ((winner = is_win()))
        return winner;
    else
        return simulation(choices[r][0], choices[r][1], (c == MY_SIGN ? OPPONENT_SIGN : MY_SIGN), depth + 1);
}

char            mcts_setup_simulation(t_node *node, t_node *first_child_to_simul)
{
    char        win;

    if (!first_child_to_simul)
        fprintf(stderr, "IMPOSSIBLE pas d'enfants à simuler\n");

    do_action(first_child_to_simul->grid[0], first_child_to_simul->grid[1], first_child_to_simul->sign);

    // fprintf(stderr, "MCTS SIMULATION\n");
    // -- Simulation
    win = simulation(first_child_to_simul->grid[0], first_child_to_simul->grid[1], first_child_to_simul->sign, 0);

    // fprintf(stderr, "MCTS BACKPROPAGATION : win %c\n", win);
	// -- Backpropagation
	first_child_to_simul->score += (win == first_child_to_simul->sign ? 2 : (win == DRAW_CASE ? 1 : 0)); //draw
	first_child_to_simul->visited++;
    // fprintf(stderr, "my_node %d %d\topponent_node %d %d\twin %c\n", first_child_to_simul->score, first_child_to_simul->visited, second_child_to_simul->score, second_child_to_simul->visited, win);
	return win;
}

/*
------------------------------------------------------------------------------------
-	Expansion
*/

char            expansion(t_node *node, char sign)
{
	t_node		*tmp;
	t_node		*tmp2;
	t_node		*first_child_to_simul = NULL;
	char    	win;

	// fprintf(stderr, "MCTS EXPANSION (2 new depth)\n");

    get_choices(node->grid[0], node->grid[1]);
    float r = rand() % choices_len;
    for (int i = 0; i < choices_len; i++)
    {
        if (i == 0)
        {
            node->childs = new_node(choices[i], sign);
            tmp = node->childs;
        }
        else
        {
            tmp->next = new_node(choices[i], sign);
            tmp = tmp->next;
        }
        if (i == r)
            first_child_to_simul = tmp;
        // fprintf(stderr, "Create my_choice %d\n", i);
    }
    return mcts_setup_simulation(node, first_child_to_simul);
}

/*
------------------------------------------------------------------------------------
-	Backpropagation
*/

char         monte_carlo(t_node *branch, int depth)
{
	// fprintf(stderr, "MCTS SELECTION n°%d score = %d\tvisit = %d\n", depth, branch->score, branch->visited);

    // Selection lines 1 and 2
	t_node	*child = select_node_uct(branch, 0);
    char    win;

    // print_game();

    // Tree ascent
    do_action(child->grid[0], child->grid[1], child->sign);
    if (child->childs)
    {
        // fprintf(stderr, "\nMCTS go deeper\n");
        // -- Tree ascent / Backpropagation
        win = monte_carlo(child, depth + 1);
    }
    else
    {
        // fprintf(stderr, "\nMCTS leaf meet !\n");
        if (child->end || (child->end = is_win()))
            win = child->end;
        else
            win = expansion(child, child->sign == MY_SIGN ? OPPONENT_SIGN : MY_SIGN);

        // Expansion + Simulation + begin Backpropagation

        // fprintf(stderr, "backpropagation new childs into %p\n", child->childs);
    }
    // fprintf(stderr, "backpropagation '%c' += score/visit\t%d 1\n", branch->sign, (win == child->sign ? 2 : 0));
    child->score += (win == child->sign ? 2 : (win == DRAW_CASE ? 1 : 0));
    child->visited++;
    return win;
}

/*
------------------------------------------------------------------------------------
-	End turn
*/

int         is_invalid_choices(char y, char x)
{
    for (int i = 0; i < valid_choices_len; i++)
    {
        // fprintf(stderr, "Valid yx %d %d\n", valid_choices[i][0], valid_choices[i][1]);
        if (valid_choices[i][0] == y & valid_choices[i][1] == x)
            return 0;
    }
    return 1;
}

void        select_best_choice(t_node **tree)
{
    t_node  *best_node;
    t_node  *child;
    float   best_percent = 0;
    float   tmp;

    child = (*tree)->childs;
    while (child)
    {
        if ((tmp = child->score / (float)child->visited) > best_percent)
        {
            best_node = child;
            best_percent = tmp;
        }
        child = child->next;
    }

    /*if (is_invalid_choices(best_node->grid[0], best_node->grid[1]))
    {
        fprintf(stderr, "INVALID CHOICE -> %d %d\n", best_node->grid[0], best_node->grid[1]);
        print_game();
        print_tree(*tree, 0);
    }*/

    // Go forward in the path
    *tree = best_node;
    mcts_row = best_node->grid[0];
    mcts_col = best_node->grid[1];
    //printf("%d %d\n", best_node->grid[0], best_node->grid[1]);
    fprintf(stderr, "MCTS choice - > %d %d\n", best_node->grid[0], best_node->grid[1]);
}

/*
------------------------------------------------------------------------------------
-	Parsing
*/

t_node          *create_tree()
{
    char        root[2] = {'A', 'N'};   //Random things
	t_node		*tree = new_node(root, 'T');
	t_node		*tmp;
	t_node		*tmp2;

    // fprintf(stderr, "Start creating tree\n");

    //Expand
    get_choices(opponent_row, opponent_col);

    for (int i = 0; i < choices_len; i++)
    {
		// fprintf(stderr, "Create my_choice %d\n", i);
        if (i == 0)
		{
			tree->childs = new_node(choices[i], MY_SIGN);
            tmp = tree->childs;
		}
		else
		{
			tmp->next = new_node(choices[i], MY_SIGN);
            tmp = tmp->next;
		}
    }
    // fprintf(stderr, "END creating tree !\n");
	return tree;
}

void        parsing(t_node **tree)   //Faire le parsing sur les games_save
{
    char    ret;
    print_game();

    fprintf(stderr, "scanf my choice - >\n");
    scanf("%d%d", (int *)&opponent_row, (int *)&opponent_col);
    while (((ret = get_next_grid(mcts_row, mcts_col)) != -1 && ret != (int)(opponent_row / 3) * 3 + (opponent_col / 3)) || game[opponent_row][opponent_col] != EMPTY_CASE)
    {
        fprintf(stderr, "MCTS grid=%d\n", ret);
        fprintf(stderr, "scanf my choice - >");
        scanf("%d%d", (int *)&opponent_row, (int *)&opponent_col);
    }

    if (opponent_col != -1)
    {
        // fprintf(stderr, "Parsing, last opp move %d %d\n", opponent_row, opponent_col);
        do_action(opponent_row, opponent_col, OPPONENT_SIGN);
        if (tree)
        {
            // fprintf(stderr, "tree adrr %p\ttree child addr %p\n", *tree, (*tree)->childs);
            t_node  *tmp = (*tree)->childs;
            *tree = NULL;
            while (tmp)
            {
                // fprintf(stderr, "tmp = %p\n", tmp);
                if (tmp->grid[0] == opponent_row && tmp->grid[1] == opponent_col)
                {
                    // fprintf(stderr, "opponent choice find ! %p\n",  tmp);
                    *tree = tmp;
                    break ;
                }
                tmp = tmp->next;
            }
            // if (!*tree)
            //     fprintf(stderr, "WTTFF parsing wrong tree assignation !!!\n");
        }
    }
    //fprintf(stderr, "scanf valid choices\n");

    memcpy(&game_save, &game, sizeof(game));
    memcpy(&real_game_save, &real_game, sizeof(real_game));

    /*scanf("%d", (int *)&valid_choices_len);
    for (int i = 0; i < valid_choices_len; i++)
    {
        scanf("%d%d", (int *)&valid_choices[i][0], (int *)&valid_choices[i][1]);
        // fprintf(stderr, "Valid yx %d %d\n", valid_choices[i][0], valid_choices[i][1]);
    }*/
}

void        init()
{
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            game[i][j] = EMPTY_CASE;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            real_game[i][j] = EMPTY_CASE;
    // fprintf(stderr, "Init game :\n");
    // print_game();

    choices = (char **)malloc(sizeof(char *) * (81 + 1));
    choices[81] = NULL;
    for (int i = 0; i < 81; i++)
    {
        choices[i] = (char *)malloc(sizeof(char) * (2 + 1));
        choices[i][2] = '\0';
    }
    srand(time(0));
}

int		main()
{
	t_node  *tree;
    float   timeout = SEC_FIRST_TURN;
    char    ret;

    fprintf(stderr, "You  -> O\n");
    fprintf(stderr, "MCTS -> X\n");

    init();
    parsing(NULL);
    tree = create_tree();

    while (1) {

        for (i = 0; i < N_ITER_MCTS; i++)
        {
            memcpy(&game, &game_save, sizeof(game_save));
            memcpy(&real_game, &real_game_save, sizeof(real_game_save));
        
		    monte_carlo(tree, 0);
            tree->score++;
            tree->visited++;
        }
        //fprintf(stderr, "Max iter MCTS %d\n", i);
        
        memcpy(&game, &game_save, sizeof(game));
        memcpy(&real_game, &real_game_save, sizeof(real_game_save));
        
        select_best_choice(&tree);
        start_t = clock();
        timeout = SEC_PER_TURN;

        // Apply my choice
        do_action(tree->grid[0], tree->grid[1], MY_SIGN);

        if ((ret = is_win()))
        {
            print_game();
            fprintf(stderr, "END GAME -> WINNER : %c\n", ret);
            return 0;
        }
        // Get + apply opponent choice
        parsing(&tree);
        if ((ret = is_win()))
        {
            print_game();
            fprintf(stderr, "END GAME -> WINNER : %c\n", ret);
            return 0;
        }
    }
	return 0;
}
