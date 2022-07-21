import streamlit as st
from src import home, about, mail, tic_tac_toe


def init():
    st.session_state.page = 'Homepage'
    st.session_state.project = False
    st.session_state.game = False

    st.session_state.pages = {
        'Homepage': home.main,
        'About me': about.main,
        'Message me': mail.main,
        'Tic Tac Toe': tic_tac_toe.main,
    }


def draw_style():
    st.set_page_config(page_title='2 Player RL Games', page_icon='🎮')

    style = """
        <style>
            header {visibility: visible;}
            footer {visibility: hidden;}
        </style>
    """

    st.markdown(style, unsafe_allow_html=True)


def load_page():
    st.session_state.pages[st.session_state.page]()


def set_page(loc=None, reset=False):
    if not st.session_state.page == 'Homepage':
        for key in list(st.session_state.keys()):
            if key not in ('page', 'project', 'game', 'pages', 'set'):
                st.session_state.pop(key)

    if loc:
        st.session_state.page = loc
    else:
        st.session_state.page = st.session_state.set

    if reset:
        st.session_state.project = False
    elif st.session_state.page in ('Message me', 'About me'):
        st.session_state.project = True
        st.session_state.game = False
    else:
        pass


def change_button():
    set_page('Tic Tac Toe')
    st.session_state.game = True
    st.session_state.project = True


def main():
    if 'page' not in st.session_state:
        init()

    draw_style()

    with st.sidebar:
        project, about ,contact= st.columns([1.4, 0.9, 1.6])
        #contact = st.columns([0.4, 1])

        if not st.session_state.project:
            project.button('🎮 Games', on_click=change_button)
        else:
            project.button('🏠 Home', on_click=set_page, args=('Homepage', True))

        if st.session_state.project and st.session_state.game:
            st.selectbox(
                'Games',
                ['Tic Tac Toe'],
                key='set',
                on_change=set_page,
            )

        about.button('About', on_click=set_page, args=('About me',))

        contact.button(
            '✉️ Contact', on_click=set_page, args=('Message me',)
        )

        if st.session_state.page == 'Homepage':
            st.image('https://media.giphy.com/media/xTiIzJSKB4l7xTouE8/giphy.gif')
            

    load_page()


if __name__ == '__main__':
    main()
