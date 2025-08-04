import random


class TicTacToe:
    def __init__(self):
        # The board state is represented as a tuple to make it hashable
        self.board = (" ") * 9
        self.players = ["X", "O"]

    def is_winner(self, player):
        winning_combinations = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),  # Rows
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),  # Columns
            (0, 4, 8),
            (2, 4, 6),  # Diagonals
        ]
        for combo in winning_combinations:
            if (
                self.board[combo[0]]
                == self.board[combo[1]]
                == self.board[combo[2]]
                == player
            ):
                return True
        return False

    def is_draw(self):
        return " " not in self.board

    def get_available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == " "]

    def make_move(self, position, player):
        if self.board[position] == " ":
            new_board = list(self.board)
            new_board[position] = player
            self.board = tuple(new_board)
            return True
        return False


class QLearningAgent:
    def __init__(self, player):
        self.player = player
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2  # Exploration rate

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_moves):
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)  # Explore
        else:
            q_values = {move: self.get_q_value(state, move) for move in available_moves}
            max_q = max(q_values.values())
            # Choose a random action from those with the highest Q-value
            best_moves = [move for move, q_val in q_values.items() if q_val == max_q]
            return random.choice(best_moves)  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        old_q_value = self.get_q_value(state, action)

        if next_state is None:  # Game over
            next_max_q_value = 0
        else:
            available_moves = TicTacToe().get_available_moves()
            if not available_moves:
                next_max_q_value = 0
            else:
                next_q_values = [
                    self.get_q_value(next_state, move) for move in available_moves
                ]
                next_max_q_value = max(next_q_values)

        new_q_value = old_q_value + self.learning_rate * (
            reward + self.discount_factor * next_max_q_value - old_q_value
        )
        self.q_table[(state, action)] = new_q_value


def train(agent, num_games=10000):
    for i in range(num_games):
        game = TicTacToe()
        state = game.board
        while True:
            # Agent's turn
            agent_moves = game.get_available_moves()
            if not agent_moves:
                break
            action = agent.choose_action(state, agent_moves)

            # Make the move and get next state
            game.make_move(action, agent.player)
            next_state = game.board

            # Determine reward
            reward = 0
            if game.is_winner(agent.player):
                reward = 1
                agent.update_q_table(state, action, reward, None)
                break
            elif game.is_draw():
                agent.update_q_table(state, action, reward, None)
                break

            # Opponent's turn (random player)
            state_opponent = game.board
            opponent_moves = game.get_available_moves()
            if not opponent_moves:
                break
            opponent_move = random.choice(opponent_moves)
            game.make_move(opponent_move, "O" if agent.player == "X" else "X")

            # If opponent wins, the agent gets a negative reward
            if game.is_winner("O" if agent.player == "X" else "X"):
                agent.update_q_table(state, action, -1, None)
                break
            elif game.is_draw():
                agent.update_q_table(state, action, 0, None)
                break

            # Update Q-table
            agent.update_q_table(state, action, reward, game.board)
            state = game.board


if __name__ == "__main__":
    agent = QLearningAgent("X")
    train(agent)
    print("Training complete. Agent has learned to play Tic-Tac-Toe!")

    # Example of agent playing a game after training
    game = TicTacToe()
    while True:
        # Agent's turn
        agent_moves = game.get_available_moves()
        if not agent_moves:
            print("Game Over - Draw!")
            break

        # In a real game, turn off exploration
        agent.epsilon = 0.0
        best_move = agent.choose_action(game.board, agent_moves)
        game.make_move(best_move, agent.player)
        print(f"Agent ('{agent.player}') moved to position {best_move}")

        # Print board
        print(f" {game.board[0]} | {game.board[1]} | {game.board[2]} ")
        print("---+---+---")
        print(f" {game.board[3]} | {game.board[4]} | {game.board[5]} ")
        print("---+---+---")
        print(f" {game.board[6]} | {game.board[7]} | {game.board[8]} ")

        if game.is_winner(agent.player):
            print("Agent wins!")
            break
        if game.is_draw():
            print("Draw!")
            break

        # Human's turn
        opponent_moves = game.get_available_moves()
        if not opponent_moves:
            print("Game Over - Draw!")
            break

        try:
            human_move = int(input("Enter your move (0-8): "))
            while human_move not in opponent_moves:
                human_move = int(input("Invalid move. Try again: "))
            game.make_move(human_move, "O")
        except ValueError:
            print("Invalid input.")
            continue

        if game.is_winner("O"):
            print("You win!")
            break
        if game.is_draw():
            print("Draw!")
            break
