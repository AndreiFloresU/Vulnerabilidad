import streamlit as st
from typing import Optional


def tarjeta_metrica(
    titulo: str,
    valor: str | int | float,
    color: str = "#1f77b4",
    icono: Optional[str] = None,
    descripcion: Optional[str] = None,
):
    """
    Crea una tarjeta minimalista para mostrar métricas

    Args:
        titulo: Título de la métrica
        valor: Valor numérico o texto a mostrar
        color: Color de la tarjeta en formato hex (por defecto azul)
        icono: Emoji o icono opcional
        descripcion: Descripción adicional opcional
    """

    # Formatear el valor si es numérico
    if isinstance(valor, (int, float)):
        valor_formateado = f"{valor:,}"
    else:
        valor_formateado = str(valor)

    # Crear el icono si se proporciona
    icono_html = (
        f'<span style="margin-right: 8px; font-size: 1.2em;">{icono}</span>'
        if icono
        else ""
    )

    # Crear descripción si se proporciona
    descripcion_html = (
        f'<p style="margin: 4px 0 0 0; font-size: 0.7em; color: rgba(255,255,255,0.8); font-weight: 300;">{descripcion}</p>'
        if descripcion
        else ""
    )

    # HTML de la tarjeta
    html_tarjeta = f"""
    <div style="
        background: linear-gradient(135deg, {color} 0%, {color}dd 100%);
        padding: 16px 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: none;
        min-height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease;
    ">
        <div style="margin-bottom: 4px;">
            {icono_html}<span style="font-size: 0.8em; font-weight: 500; opacity: 0.9;">{titulo}</span>
        </div>
        <div style="font-size: 1.8em; font-weight: 700; margin: 2px 0;">
            {valor_formateado}
        </div>
        {descripcion_html}
    </div>
    """

    st.markdown(html_tarjeta, unsafe_allow_html=True)


def tarjeta_simple(titulo: str, valor: str | int | float, color: str = "#1f77b4"):
    """
    Versión simplificada de la tarjeta para uso básico

    Args:
        titulo: Título de la métrica
        valor: Valor a mostrar
        color: Color de la tarjeta en formato hex
    """
    tarjeta_metrica(titulo=titulo, valor=valor, color=color)


# Colores predefinidos para uso común
COLORES = {
    "azul": "#1f77b4",
    "verde": "#2ca02c",
    "rojo": "#d62728",
    "naranja": "#ff7f0e",
    "morado": "#9467bd",
    "rosa": "#e377c2",
    "gris": "#7f7f7f",
    "amarillo": "#bcbd22",
    "cyan": "#17becf",
}
