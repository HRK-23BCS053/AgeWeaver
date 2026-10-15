"""
State management, resource caching, and pipeline singletons for AgeWeaver Streamlit UI.
Ensures diffusion models and face analyzers are cached in RAM via @st.cache_resource
to avoid redundant loading penalties on CPU.
"""

import os
import sys
import streamlit as st
from PIL import Image

# Ensure src and root are in sys.path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))

from ageweaver.pipeline import AgeWeaverPipeline
from ageweaver.identity import FaceIdentityAnalyzer
from ageweaver.dataset import DatasetManager
from ageweaver.evaluation import AgeWeaverEvaluator
from ageweaver.experiment import ExperimentRunner
from ageweaver.config import PipelineConfig

@st.cache_resource(show_spinner="Loading Stable Diffusion v1-5 Pipeline into RAM...")
def get_pipeline() -> AgeWeaverPipeline:
    """Loads and caches the diffusion pipeline singleton on CPU."""
    cfg = PipelineConfig(device="cpu", local_files_only=True)
    pipe = AgeWeaverPipeline(config=cfg)
    pipe.load_pipeline()
    return pipe

@st.cache_resource(show_spinner="Initializing InsightFace buffalo_l Analyzer...")
def get_analyzer() -> FaceIdentityAnalyzer:
    """Initializes and caches the FaceIdentityAnalyzer singleton."""
    return FaceIdentityAnalyzer()

@st.cache_resource
def get_dataset_manager() -> DatasetManager:
    """Returns the DatasetManager singleton."""
    return DatasetManager()

@st.cache_resource
def get_evaluator() -> AgeWeaverEvaluator:
    """Returns the AgeWeaverEvaluator singleton."""
    analyzer = get_analyzer()
    return AgeWeaverEvaluator(face_analyzer=analyzer)

@st.cache_resource
def get_experiment_runner() -> ExperimentRunner:
    """Returns the ExperimentRunner singleton."""
    pipe = get_pipeline()
    evaluator = get_evaluator()
    return ExperimentRunner(pipeline=pipe, evaluator=evaluator)

def init_session_state():
    """Initializes UI session state defaults."""
    if "selected_source_mode" not in st.session_state:
        st.session_state.selected_source_mode = "benchmark"
    if "selected_identity_id" not in st.session_state:
        st.session_state.selected_identity_id = "ID001_RAJESH_KHANNA"
    if "current_source_image" not in st.session_state:
        st.session_state.current_source_image = None
    if "current_source_meta" not in st.session_state:
        st.session_state.current_source_meta = None
    if "last_progression_result" not in st.session_state:
        st.session_state.last_progression_result = None
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
