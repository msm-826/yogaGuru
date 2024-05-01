import yaml
from yaml.loader import SafeLoader

import streamlit as st
import streamlit_authenticator as stauth

from main import main_content

st.set_page_config(layout="wide")
st.sidebar.title('YogaGuru')

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

authentication_choice = None
if 'authentication_status' not in st.session_state:
    st.session_state.authentication_status = None

if st.session_state["authentication_status"] is True:
    with st.sidebar:
        st.write(f'Welcome <span style="color:green"><i><b>{st.session_state["name"]}</b></i></span>', unsafe_allow_html=True)
        authenticator.logout(location='sidebar', key='logout_button')
    main_content()

else:
    auth_selectbox_placeholder = st.sidebar.empty()
    authentication_choice = auth_selectbox_placeholder.selectbox(
        'Choose: ',
        ('Login', 'Register')
    )

    if authentication_choice == 'Register':
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user, password_of_registered_user = authenticator.register_user(pre_authorization=False)
            if email_of_registered_user:
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
                st.success('User registered successfully')
        except Exception as e:
            st.error(e)
    elif authentication_choice == 'Login':
        authenticator.login('main')
        if st.session_state["authentication_status"] is True:
            with st.sidebar:
                st.write(f'Welcome <span style="color:green"><i><b>{st.session_state["name"]}</b></i></span>', unsafe_allow_html=True)
                authenticator.logout(location='sidebar', key='logout_button')
            main_content()
        elif st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')
        elif st.session_state["authentication_status"] is None:
            st.warning('Please enter your username and password')
    else:
        st.warning("Please choose Register or Login")

    if st.session_state["authentication_status"] is not None:
        auth_selectbox_placeholder.empty()