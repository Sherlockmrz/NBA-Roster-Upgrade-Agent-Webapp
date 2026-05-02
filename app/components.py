"""Reusable UI components for the Streamlit app."""

import streamlit as st


def section_header(title: str, description: str = "") -> None:
    st.subheader(title)
    if description:
        st.caption(description)
