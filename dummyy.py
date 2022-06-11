import streamlit as st

def main():
    st.title('Search Lyrics')
    search = st.text_input('Enter song title:')
    st.title(search)

if __name__ == '__main__':
    main()