import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple


def obtener_facultades_por_grupo(
    df_personas: pd.DataFrame, grupo_seleccionado: str
) -> List[str]:
    """
    Obtiene las facultades disponibles según el grupo de interés seleccionado
    """
    # Filtrar por grupo de interés primero
    df_filtrado = df_personas[df_personas["tipo"] == grupo_seleccionado]

    facultades = sorted(df_filtrado["facultad"].dropna().unique().tolist())
    return ["Todos"] + facultades


def obtener_carreras_por_grupo_y_facultad(
    df_personas: pd.DataFrame, grupo_seleccionado: str, facultad_seleccionada: str
) -> List[str]:
    """
    Obtiene las carreras disponibles según el grupo de interés y facultad seleccionados
    """
    # Filtrar por grupo de interés primero
    df_filtrado = df_personas[df_personas["tipo"] == grupo_seleccionado]

    if facultad_seleccionada != "Todos":
        df_filtrado = df_filtrado[df_filtrado["facultad"] == facultad_seleccionada]

    carreras = sorted(df_filtrado["carrera_homologada"].dropna().unique().tolist())
    return ["Todos"] + carreras


def mostrar_filtros(
    df_personas: pd.DataFrame, key_suffix: str = ""
) -> Tuple[str, str, str]:
    """
    Muestra los filtros y retorna los valores seleccionados

    Args:
        df_personas: DataFrame con los datos de personas
        key_suffix: Sufijo único para las keys de los widgets (para evitar conflictos entre páginas)

    Returns:
        Tupla con (grupo_seleccionado, facultad_seleccionada, carrera_seleccionada)
    """
    st.header("🔍 Filtros")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Grupo de interés
        opciones_grupo = {"E": "Enrollment", "A": "Afluentes", "G": "Graduados"}

        grupo_seleccionado = st.selectbox(
            "Grupo de interés:",
            options=list(opciones_grupo.keys()),
            format_func=lambda x: opciones_grupo[x],
            index=0,  # Por defecto "E" (Enrollment)
            key=f"grupo_interes_{key_suffix}",
        )

    with col2:
        # Facultad (filtrada según el grupo de interés)
        facultades_disponibles = obtener_facultades_por_grupo(
            df_personas, grupo_seleccionado
        )

        facultad_seleccionada = st.selectbox(
            "Facultad:",
            options=facultades_disponibles,
            index=0,  # Por defecto "Todos"
            key=f"facultad_{key_suffix}",
        )

    with col3:
        # Carrera (depende del grupo de interés y facultad seleccionada)
        carreras_disponibles = obtener_carreras_por_grupo_y_facultad(
            df_personas, grupo_seleccionado, facultad_seleccionada
        )

        carrera_seleccionada = st.selectbox(
            "Carrera:",
            options=carreras_disponibles,
            index=0,  # Por defecto "Todos"
            key=f"carrera_{key_suffix}",
        )

    return grupo_seleccionado, facultad_seleccionada, carrera_seleccionada


def aplicar_filtros(
    df_vulnerabilidad: Dict[str, pd.DataFrame],
    grupo_seleccionado: str,
    facultad_seleccionada: str,
    carrera_seleccionada: str,
) -> Dict[str, pd.DataFrame]:
    """
    Aplica los filtros seleccionados únicamente a la hoja Personas

    Args:
        df_vulnerabilidad: Diccionario con todas las hojas del Excel
        grupo_seleccionado: Grupo de interés seleccionado
        facultad_seleccionada: Facultad seleccionada
        carrera_seleccionada: Carrera seleccionada

    Returns:
        Diccionario con los DataFrames donde solo Personas está filtrado
    """
    df_filtrado = df_vulnerabilidad.copy()

    # Filtrar personas según los criterios seleccionados
    df_personas = df_filtrado["Personas"].copy()

    # Filtro por grupo de interés (tipo)
    if grupo_seleccionado:
        df_personas = df_personas[df_personas["tipo"] == grupo_seleccionado]

    # Filtro por facultad
    if facultad_seleccionada != "Todos":
        df_personas = df_personas[df_personas["facultad"] == facultad_seleccionada]

    # Filtro por carrera
    if carrera_seleccionada != "Todos":
        df_personas = df_personas[
            df_personas["carrera_homologada"] == carrera_seleccionada
        ]

    # Actualizar solo el DataFrame de personas filtrado
    df_filtrado["Personas"] = df_personas

    # Las demás hojas permanecen sin cambios (datos completos)
    return df_filtrado
