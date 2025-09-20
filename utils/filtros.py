import streamlit as st
import pandas as pd
from typing import Dict, List, Optional


def inicializar_filtros(df_vulnerabilidad: Dict[str, pd.DataFrame]):
    """
    Inicializa los filtros en session_state si no existen

    Args:
        df_vulnerabilidad: Diccionario con todas las hojas del Excel
    """
    if "filtros_inicializados" not in st.session_state:
        # Obtener valores únicos de las columnas relevantes
        df_personas = df_vulnerabilidad["Personas"]

        # Grupo de interés - valores por defecto
        st.session_state.grupo_interes = "E"  # Enrollment por defecto

        # Facultad - obtener valores únicos y agregar "Todos"
        facultades_unicas = sorted(df_personas["facultad"].dropna().unique().tolist())
        st.session_state.facultades_disponibles = ["Todos"] + facultades_unicas
        st.session_state.facultad_seleccionada = "Todos"  # Todos por defecto

        # Carrera - obtener valores únicos y agregar "Todos"
        carreras_unicas = sorted(
            df_personas["carrera_homologada"].dropna().unique().tolist()
        )
        st.session_state.carreras_disponibles = ["Todos"] + carreras_unicas
        st.session_state.carrera_seleccionada = "Todos"  # Todos por defecto

        st.session_state.filtros_inicializados = True


def mostrar_filtros():
    """
    Muestra los widgets de filtros en la sidebar y actualiza session_state
    """
    st.sidebar.header("🔍 Filtros")

    # Grupo de interés
    opciones_grupo = {"E": "Enrollment", "A": "Afluentes", "G": "Graduados"}

    grupo_seleccionado = st.sidebar.selectbox(
        "Grupo de interés:",
        options=list(opciones_grupo.keys()),
        format_func=lambda x: f"{x} - {opciones_grupo[x]}",
        index=list(opciones_grupo.keys()).index(st.session_state.grupo_interes),
        key="grupo_interes",
    )

    # Facultad
    facultad_seleccionada = st.sidebar.selectbox(
        "Facultad:",
        options=st.session_state.facultades_disponibles,
        index=st.session_state.facultades_disponibles.index(
            st.session_state.facultad_seleccionada
        ),
        key="facultad_seleccionada",
    )

    # Carrera (depende de la facultad seleccionada)
    carreras_filtradas = obtener_carreras_por_facultad(
        st.session_state.facultad_seleccionada
    )

    # Si cambia la facultad, resetear carrera a "Todos"
    if "facultad_anterior" not in st.session_state:
        st.session_state.facultad_anterior = facultad_seleccionada

    if st.session_state.facultad_anterior != facultad_seleccionada:
        st.session_state.carrera_seleccionada = "Todos"
        st.session_state.facultad_anterior = facultad_seleccionada

    # Asegurar que la carrera seleccionada esté en las opciones disponibles
    if st.session_state.carrera_seleccionada not in carreras_filtradas:
        st.session_state.carrera_seleccionada = "Todos"

    carrera_seleccionada = st.sidebar.selectbox(
        "Carrera:",
        options=carreras_filtradas,
        index=carreras_filtradas.index(st.session_state.carrera_seleccionada),
        key="carrera_seleccionada",
    )


def obtener_carreras_por_facultad(facultad_seleccionada: str) -> List[str]:
    """
    Obtiene las carreras disponibles según la facultad seleccionada
    """
    if "df_vulnerabilidad" not in st.session_state:
        return ["Todos"]

    df_personas = st.session_state.df_vulnerabilidad["Personas"]

    if facultad_seleccionada == "Todos":
        carreras = sorted(df_personas["carrera_homologada"].dropna().unique().tolist())
    else:
        carreras = sorted(
            df_personas[df_personas["facultad"] == facultad_seleccionada][
                "carrera_homologada"
            ]
            .dropna()
            .unique()
            .tolist()
        )

    return ["Todos"] + carreras


def mostrar_filtros_en_pagina():
    """
    Muestra los filtros en la página principal con lógica de cascada completa
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
            index=list(opciones_grupo.keys()).index(st.session_state.grupo_interes),
            key="grupo_interes",
        )

    with col2:
        # Facultad (filtrada según el grupo de interés)
        facultades_filtradas = obtener_facultades_por_grupo(
            st.session_state.grupo_interes
        )

        # Si cambia el grupo, resetear facultad a "Todos"
        if "grupo_anterior" not in st.session_state:
            st.session_state.grupo_anterior = st.session_state.grupo_interes

        if st.session_state.grupo_anterior != st.session_state.grupo_interes:
            st.session_state.facultad_seleccionada = "Todos"
            st.session_state.carrera_seleccionada = "Todos"
            st.session_state.grupo_anterior = st.session_state.grupo_interes

        # Asegurar que la facultad seleccionada esté en las opciones disponibles
        if st.session_state.facultad_seleccionada not in facultades_filtradas:
            st.session_state.facultad_seleccionada = "Todos"

        facultad_seleccionada = st.selectbox(
            "Facultad:",
            options=facultades_filtradas,
            index=facultades_filtradas.index(st.session_state.facultad_seleccionada),
            key="facultad_seleccionada",
        )

    with col3:
        # Carrera (depende del grupo de interés y facultad seleccionada)
        carreras_filtradas = obtener_carreras_por_grupo_y_facultad(
            st.session_state.grupo_interes, st.session_state.facultad_seleccionada
        )

        # Si cambia la facultad, resetear carrera a "Todos"
        if "facultad_anterior" not in st.session_state:
            st.session_state.facultad_anterior = st.session_state.facultad_seleccionada

        if st.session_state.facultad_anterior != st.session_state.facultad_seleccionada:
            st.session_state.carrera_seleccionada = "Todos"
            st.session_state.facultad_anterior = st.session_state.facultad_seleccionada

        # Asegurar que la carrera seleccionada esté en las opciones disponibles
        if st.session_state.carrera_seleccionada not in carreras_filtradas:
            st.session_state.carrera_seleccionada = "Todos"

        carrera_seleccionada = st.selectbox(
            "Carrera:",
            options=carreras_filtradas,
            index=carreras_filtradas.index(st.session_state.carrera_seleccionada),
            key="carrera_seleccionada",
        )


def obtener_facultades_por_grupo(grupo_seleccionado: str) -> List[str]:
    """
    Obtiene las facultades disponibles según el grupo de interés seleccionado
    """
    if "df_vulnerabilidad" not in st.session_state:
        return ["Todos"]

    df_personas = st.session_state.df_vulnerabilidad["Personas"]

    # Filtrar por grupo de interés primero
    df_filtrado = df_personas[df_personas["tipo"] == grupo_seleccionado]

    facultades = sorted(df_filtrado["facultad"].dropna().unique().tolist())
    return ["Todos"] + facultades


def obtener_carreras_por_grupo_y_facultad(
    grupo_seleccionado: str, facultad_seleccionada: str
) -> List[str]:
    """
    Obtiene las carreras disponibles según el grupo de interés y facultad seleccionados
    """
    if "df_vulnerabilidad" not in st.session_state:
        return ["Todos"]

    df_personas = st.session_state.df_vulnerabilidad["Personas"]

    # Filtrar por grupo de interés primero
    df_filtrado = df_personas[df_personas["tipo"] == grupo_seleccionado]

    if facultad_seleccionada != "Todos":
        df_filtrado = df_filtrado[df_filtrado["facultad"] == facultad_seleccionada]

    carreras = sorted(df_filtrado["carrera_homologada"].dropna().unique().tolist())
    return ["Todos"] + carreras


def aplicar_filtros(
    df_vulnerabilidad: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Aplica los filtros seleccionados únicamente a la hoja Personas

    Args:
        df_vulnerabilidad: Diccionario con todas las hojas del Excel

    Returns:
        Diccionario con los DataFrames donde solo Personas está filtrado
    """
    df_filtrado = df_vulnerabilidad.copy()

    # Filtrar personas según los criterios seleccionados
    df_personas = df_filtrado["Personas"].copy()

    # Filtro por grupo de interés (tipo)
    if st.session_state.grupo_interes:
        df_personas = df_personas[df_personas["tipo"] == st.session_state.grupo_interes]

    # Filtro por facultad
    if st.session_state.facultad_seleccionada != "Todos":
        df_personas = df_personas[
            df_personas["facultad"] == st.session_state.facultad_seleccionada
        ]

    # Filtro por carrera
    if st.session_state.carrera_seleccionada != "Todos":
        df_personas = df_personas[
            df_personas["carrera_homologada"] == st.session_state.carrera_seleccionada
        ]

    # Actualizar solo el DataFrame de personas filtrado
    df_filtrado["Personas"] = df_personas

    # Las demás hojas permanecen sin cambios (datos completos)

    return df_filtrado
