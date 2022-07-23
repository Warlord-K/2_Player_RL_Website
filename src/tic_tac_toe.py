import streamlit as st
import numpy as np
import joblib
import os
class Agent():
  def __init__(self,epsilon = 0.1,alpha = 0.1,gamma = 0.9):
    #self.state_memory = np.zeros(mem_length)
    #self.observation_memory = np.zeros(mem_length)
    #self.reward_memory = np.zeros(mem_length)
    self.Q = np.zeros((3**9,9))
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma

  def take_action(self,state):
    if np.random.random() < self.epsilon:
        return np.random.randint(0,9)
    else:
        return np.argmax(self.Q[state])

  def update_Q(self,state,action,reward,next_state):

    self.Q[state,action] = self.Q[state,action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state,:]) - self.Q[state,action])


def init(post_init=False):
    if not post_init:
        st.session_state.opponent = 'Computer'
        st.session_state.win = {'X': 0, 'O': 0}
    st.session_state.board = np.full((3, 3), '.', dtype=str)
    st.session_state.state = 0
    st.session_state.player = 'X'
    st.session_state.warning = False
    st.session_state.winner = None
    st.session_state.over = False


def check_available_moves(extra=False) -> list:
    raw_moves = [row for col in st.session_state.board.tolist() for row in col]
    num_moves = [i for i, spot in enumerate(raw_moves) if spot == '.']
    if extra:
        return [(i // 3, i % 3) for i in num_moves]
    return num_moves


def check_rows(board):
    for row in board:
        if len(set(row)) == 1:
            return row[0]
    return None


def check_diagonals(board):
    if len(set([board[i][i] for i in range(len(board))])) == 1:
        return board[0][0]
    if len(set([board[i][len(board) - i - 1] for i in range(len(board))])) == 1:
        return board[0][len(board) - 1]
    return None


def check_state():
    if st.session_state.winner:
        st.success(f"Congrats! {st.session_state.winner} won the game! 🎈")
    if st.session_state.warning and not st.session_state.over:
        st.warning('⚠️ This move already exist')
    if st.session_state.winner and not st.session_state.over:
        st.session_state.over = True
        st.session_state.win[st.session_state.winner] = (
            st.session_state.win.get(st.session_state.winner, 0) + 1
        )
    elif not check_available_moves() and not st.session_state.winner:
        st.info(f'It\'s a tie 📍')
        st.session_state.over = True


def check_win(board):
    for new_board in [board, np.transpose(board)]:
        result = check_rows(new_board)
        if result:
            return result
    return check_diagonals(board)

@st.cache()
def load_agent():
    agent = Agent(epsilon=0.1)
    try:
        path = os.path.realpath(__file__)[:-15]
        Q2 = joblib.load(f'{path}/q_table2.pkl')
        agent.Q = Q2
    except:
        pass
    return agent

def computer_player():
    moves = check_available_moves(extra=True)
    if moves:
        #i, j = np.random.choice(moves)
        agent = load_agent()
        action = agent.take_action(st.session_state.state )
        i , j = action // 3, action % 3
        
        handle_click(i, j)

def handle_click(i, j):
    if (i, j) not in check_available_moves(extra=True):
        if st.session_state.opponent == "Computer":
            computer_player()
        else:
            st.session_state.warning = True
    elif not st.session_state.winner:
        st.session_state.warning = False
        turn = 1 if st.session_state.player == 'X' else 2
        action = i*3 +j
        st.session_state.state = st.session_state.state + turn* (3**action)
        st.session_state.board[i, j] = st.session_state.player
        st.session_state.player = "O" if st.session_state.player == "X" else "X"
        winner = check_win(st.session_state.board)
        if winner != ".":
            st.session_state.winner = winner


def main():
    st.write(
        """
        # ❎🅾️ Tic Tac Toe
        """
    )

    if "board" not in st.session_state:
        init()

    reset, score, player, settings = st.columns([0.5, 0.6, 1, 1])
    reset.button('New game', on_click=init, args=(True,))

    with settings.expander('Settings'):
        st.write('**Warning**: changing this setting will restart your game')
        st.selectbox(
            'Set opponent',
            ['Computer',"Human"],
            key='opponent',
            on_change=init,
            args=(True,),
        )

    for i, row in enumerate(st.session_state.board):
        cols = st.columns([5, 1, 1, 1, 5])
        for j, field in enumerate(row):
            cols[j + 1].button(
                field,
                key=f"{i}-{j}",
                on_click=handle_click
                if st.session_state.player == 'X'
                or st.session_state.opponent == 'Human'
                else computer_player(),
                args=(i, j),
            )

    check_state()

    score.button(f'❌{st.session_state.win["X"]} 🆚 {st.session_state.win["O"]}⭕')
    player.button(
        f'{"❌" if st.session_state.player == "X" else "⭕"}\'s turn'
        if not st.session_state.winner
        else f'🏁 Game finished'
    )


if __name__ == '__main__':
    main()
