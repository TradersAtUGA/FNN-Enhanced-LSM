import streamlit as st
import pandas as pd
import numpy as np

DEFAULT_STATES = {
    "submitted" : False,
    "option_type": "Bermudan",
    "option_side": "Put",
    "dimensions" : 1,
    "risk_free_interest": 5.0,
    "time_to_exp": 0.25,
    "exercise_frequency" : "Monthly",
    "correlation_type" : "Identity",
    "correlation_rho" : 0.3,
    "num_of_paths" : 1_000,
    "num_of_steps" : 150,
    "poly_degree" : 3,
    "epochs" : 20
}

for k, v in DEFAULT_STATES.items():
    if k not in st.session_state:
        st.session_state.setdefault(k, v)


def main():
    st.title("Interactive Page")

    st.write("This is the interactive portion of this project. Documentation can be found below, an example is also shown")
    
    st.pills(
        "Option Type:",
        options=("Bermudan", "American", "European"),
        selection_mode="single",
        key="option_type"
    )
    
    st.pills(
        "Option Side:",
        options=("Put", "Call"),
        selection_mode="single",
        key="option_side"
    )
    
    st.slider(
        "Dimensions:",
        min_value=1,
        max_value=20,
        key="dimensions"
    )
    
    st.slider(
        "Risk-free interest:",
        min_value=0.0,
        max_value=15.0,
        key="risk_free_interest"
    )
    
    st.slider(
        "Time to expiration (years):",
        min_value=0.00,
        max_value=5.00,
        step=0.25,
        key="time_to_exp"
    )
    # add custom later
    if st.session_state.option_type == "Bermudan":
        st.pills(
            "Exercise Frequency:",
            options=("Monthly", "Quarterly", "Semi-monthly"),
            selection_mode="single",
            key="exercise_frequency"
        )

    option_info_base_df = pd.DataFrame(data={
        "Initial Stock Prices": [0.0]*st.session_state.dimensions, 
        "Strike Prices": [0.0]*st.session_state.dimensions,
        "Volatilities": [0.0]*st.session_state.dimensions}
    )
    option_information = st.data_editor(option_info_base_df)

    if st.session_state.dimensions > 1:
        st.pills(
            "Correlation Type:",
            options=("Uniform", "Identity", "Custom"),
            selection_mode="single",
            default=st.session_state.correlation_type,
            key="correlation_type"
        )

        if st.session_state.correlation_type == "Uniform":
            st.slider(
                "Correlation Rho",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.correlation_rho,
                key="correlation_rho"
            )
        
        if st.session_state.correlation_type == "Custom":
            correlation_matrix_base_df = pd.DataFrame(
                np.eye(st.session_state.dimensions),
                columns=[f"Asset: {i + 1}" for i in range(st.session_state.dimensions)],
                index=[f"Asset: {i + 1}" for i in range(st.session_state.dimensions)]
            )

            st.write("Input the correlation matrix:")
            correlation_matrix = st.data_editor(correlation_matrix_base_df)

    st.number_input(
        "Number of paths:",
        min_value=1,
        max_value=10_000,
        key="num_of_paths"
    )
    st.number_input(
        "Number of steps:",
        min_value=1,
        max_value=5_000,
        key="num_of_steps"
    )
    st.number_input(
        "Polynomial Degree", 
        min_value=1,
        max_value=15,
        key="poly_degree"
    )
    st.number_input(
        "Number of epochs",
        min_value=1,
        max_value=50,
        key="epochs"
    )

    submit = st.button(
        "Submit",
        on_click=lambda: st.session_state.update(submitted=True),
        type="primary"
    )

    st.divider()
    if st.session_state.submitted:
        # one day add validation of the inputs but
        # for now just assume user is smart
        st.write("You pressed me!")
        st.write(st.session_state.dimensions)
        st.session_state.update(submitted=False)
    

main()
