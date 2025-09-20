import plotly.io as pio
import streamlit as st

# Definir la paleta personalizada
PALETA_PASTEL = [
    "#0070C0",
    "#358ac2",
    "#62a8d7",
]


def aplicar_tema_plotly():
    tema_personalizado = dict(
        layout=dict(
            colorway=PALETA_PASTEL,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Arial", size=14, color="#333333"),
            title=dict(font=dict(size=20, color="#333333")),
            xaxis=dict(showgrid=True, gridcolor="#eeeeee"),
            yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
        )
    )
    pio.templates["tema_pastel"] = tema_personalizado
    pio.templates.default = "tema_pastel"
