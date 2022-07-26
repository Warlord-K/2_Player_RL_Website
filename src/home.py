import streamlit as st


def main():
    st.markdown(
        '''
        <h1 align="center">
            2 Player RL Games
        </h1>

        ---

        #### About
        
        <h2>
    
        I made this project for the Cynaptics Club IIT Indore, This project contains several games whose Computer AI has been trained using Reinforcement Learning.
        Currently there are two games i.e Tic-Tac-Toe and Connect 4 which are available on this website. Acess the Games by clicking the Games button in the sidebar.
        <br><br>
        I used Deep Q-Learning for this project which I implemeted using pytorch,This was my first project in Reinforcement learning so i learnt a lot in this project.
        <br><br>
        You can contact me via LinkedIn or Instagram links to which are given in my about page,you can also email me via the contact page.
        </h2>
        
        ''',
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
