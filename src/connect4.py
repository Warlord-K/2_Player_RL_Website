import numpy as np
import random
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque 
import streamlit as st
import os
class Connect4():
    def __init__(self, board_size = (6,7)):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros(self.board_size)
        self.turn = 1
        self.done = False
        self.winner = 0
        


    def encode(self,action):
        turn = self.turn if self.turn != -1 else 2
        self.state = self.state + turn* (3**action)
    
    def encoded_state(self,state):
        return state.flatten()
    def step(self,col):
        reward = -1
        if self.is_valid_location(col):
            row = self.get_next_open_row(col)
            #action = row*self.board_size[0] + col
            #self.encode(action)
            self.board[row][col]= self.turn
            self.done = self.winning_move()
            self.winner = self.turn if self.done else 0
            self.turn *= -1
            if self.done:
                reward = 100
            if self.is_board_full():
                self.done = True
                reward = 0
            return self.encoded_state(self.board),reward,self.done,self.winner
            
        else:
            reward = -10
            return self.encoded_state(self.board),reward,self.done,self.winner
        

    def is_valid_location(self,col):
        #if this condition is true we will let the use drop self.turn here.
        #if not true that means the col is not vacant
        return self.board[5][col]==0

    def get_next_open_row(self,col):
        for r in range(self.board_size[0]):
            if self.board[r][col]==0:
                return r

    def winning_move(self):
        # Check horizontal locations for win
        for c in range(self.board_size[1]-3):
            for r in range(self.board_size[0]):
                if self.board[r][c] == self.turn and self.board[r][c+1] == self.turn and self.board[r][c+2] == self.turn and self.board[r][c+3] == self.turn:
                    return True
    
        # Check vertical locations for win
        for c in range(self.board_size[1]):
            for r in range(self.board_size[0]-3):
                if self.board[r][c] == self.turn and self.board[r+1][c] == self.turn and self.board[r+2][c] == self.turn and self.board[r+3][c] == self.turn:
                    return True
    
        # Check positively sloped diaganols
        for c in range(self.board_size[1]-3):
            for r in range(self.board_size[0]-3):
                if self.board[r][c] == self.turn and self.board[r+1][c+1] == self.turn and self.board[r+2][c+2] == self.turn and self.board[r+3][c+3] == self.turn:
                    
                    return True
    
        # Check negatively sloped diaganols
        for c in range(self.board_size[1]-3):
            for r in range(3, self.board_size[0]):
                if self.board[r][c] == self.turn and self.board[r-1][c+1] == self.turn and self.board[r-2][c+2] == self.turn and self.board[r-3][c+3] == self.turn:
                    
                    return True
        
            
        return False

    def is_board_full(self):
        for i in range(7):
            if self.is_valid_location(i):
                return False
        return True
    def render(self):
        print(np.flip(self.board,0))

class QNetwork(nn.Module):
    """ Actor (Policy) Model."""
    def __init__(self, state_size,action_size, seed, fc1_unit=64,
                 fc2_unit = 64):

        super(QNetwork,self).__init__() 
        self.seed = torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size,fc1_unit)
        self.fc2 = nn.Linear(fc1_unit,fc2_unit)
        self.fc3 = nn.Linear(fc2_unit,action_size)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent():
    """Interacts with and learns form environment."""
    
    def __init__(self, state_size, action_size, seed,buffer_size = 1e5,batch_size = 64,gamma=0.99,tau = 1e-3,lr = 5e-4,update_every = 4):
        """Initialize an Agent object.
        
        Params
        =======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.BUFFER_SIZE = int(buffer_size)  #replay buffer size
        self.BATCH_SIZE = batch_size       # minibatch size
        self.GAMMA = gamma            # discount factor
        self.TAU = tau              # for soft update of target parameters
        LR = lr              # learning rate
        self.UPDATE_EVERY = update_every        # how often to update the network
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=LR)
        
        # Replay memory 
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE,self.BATCH_SIZE,seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_step, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_step, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1)% self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory)>self.BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, self.GAMMA)
    def take_action(self, state, eps = 0):
        """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
            
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_state, dones = experiences
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # Local model is one which we need to train so it's in training mode
        self.qnetwork_local.train()
        # Target model is one with which we need to get our target so it's in evaluation mode
        # So that when we do a forward pass with target model it does not calculate gradient.
        # We will update target model weights with soft_update function
        self.qnetwork_target.eval()
        #shape of output from the model (batch_size,action_dim) = (64,4)
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
    
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        # .detach() ->  Returns a new Tensor, detached from the current graph.
        labels = rewards + (gamma* labels_next*(1-dones))
        
        loss = criterion(predicted_targets,labels).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local,self.qnetwork_target,self.TAU)
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        =======
            local model (PyTorch model): weights will be copied from
            target model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            
class ReplayBuffer:
    """Fixed -size buffe to store experience tuples."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory."""
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states,actions,rewards,next_states,dones)
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def init(post_init=False):
    if not post_init:
        st.session_state.opponent = 'Computer'
        st.session_state.win = {'Red': 0, 'Blue': 0}
        st.session_state.env = Connect4()
        st.session_state.Agent = DQNAgent(state_size=42,action_size=7,seed=0)
        try:
            path = os.path.realpath(__file__)[:-12]
            st.session_state.Agent.qnetwork_local.load_state_dict(torch.load(f'{path}/q_dict2.pt'))
        except FileNotFoundError as e:
            st.write(e)
    st.session_state.env.reset()
    st.session_state.state = 0
    st.session_state.player = 'Red'
    st.session_state.warning = False
    st.session_state.winner = None
    st.session_state.over = False

def check_state():
    if st.session_state.env.done:
        st.success(f"Congrats! {st.session_state.winner} won the game! 🎈")
    if st.session_state.warning and not st.session_state.env.done:
        st.warning('⚠️ No More Free Rows')
    if st.session_state.winner and not st.session_state.over:
        st.session_state.over = True
        st.session_state.win[st.session_state.winner] = (
            st.session_state.win.get(st.session_state.winner, 0) + 1
        )
    elif st.session_state.env.winner == 0 and st.session_state.env.is_board_full():
        st.info(f'It\'s a tie 📍')
        st.session_state.over = True

def computer_player():
    if not st.session_state.env.done:
        action = st.session_state.Agent.take_action(st.session_state.env.encoded_state(st.session_state.env.board),0.1)
        handle_click(0,action)



def handle_click(i,action):
    if not st.session_state.env.is_valid_location(action):
        if st.session_state.opponent == "Computer":
            computer_player()
        else:
            st.session_state.warning = True
    elif not st.session_state.winner:
        st.session_state.warning = False
        st.session_state.env.step(action)
        if st.session_state.env.done:
            st.session_state.winner = st.session_state.player
        st.session_state.player = "Blue" if st.session_state.player == "Red" else "Red"

def main():
    st.write(
        """
        # 🔴 Connect 4 🔵
        """
    )
    if "player" not in st.session_state:
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
    enc = [".","🔴","🔵"]
    for i, row in enumerate(np.flip(st.session_state.env.board,0)):
        cols = st.columns([2, 1, 1, 1, 1, 1, 1, 2])
        for j, field in enumerate(row):
            cols[j + 1].button(
                enc[int(field)],
                key=f"{i}-{j}",
                on_click=handle_click
                if st.session_state.player == 'Red'
                or st.session_state.opponent == 'Human'
                else computer_player(),
                args=(i,j),
            )
    check_state()

    score.button(f'🔴 {st.session_state.win["Red"]} 🆚 {st.session_state.win["Blue"]} 🔵')
    player.button(
        f'{"🔴" if st.session_state.player == "Red" else "🔵"}\'s turn'
        if not st.session_state.env.done
        else f'🏁 Game finished'
    )

if __name__ == "__main__":
    main()