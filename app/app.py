import streamlit as st

st.set_page_config(page_title="NBA Roster Upgrade Agent", page_icon="🏀", layout="wide")

st.title("🏀 NBA Roster Upgrade Agent Webapp")
st.caption("Streamlit interface for the original Tool A / Tool B / Tool C analytics pipeline.")

st.info(
    "Salary data is currently unavailable in the existing dataset style. "
    "All feasibility outputs should be interpreted without salary-cap constraints for now."
)

st.subheader("Planned pipeline (in order)")
pipeline_steps = [
    "Tool A: Need Diagnosis",
    "Tool B: Player Strength Profiling",
    "Tool C: Fit Ranking",
    "Feasibility Check (salary marked unavailable)",
    "Sensitivity Analysis",
]

for idx, step in enumerate(pipeline_steps, start=1):
    st.markdown(f"{idx}. {step}")

st.divider()
st.write("This is an initial scaffold. Core analytics and LLM orchestration are placeholders.")
