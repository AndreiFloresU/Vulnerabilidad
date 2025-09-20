import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.carga_datos import cargar_datos_vulnerabilidad
from utils.filtros import (
    inicializar_filtros,
    aplicar_filtros,
    mostrar_filtros_en_pagina,
)
from utils.tarjetas import tarjeta_simple, COLORES
import numpy as np


def obtener_familiares_periodo(datos_filtrados, periodo):
    """
    Obtiene las cédulas únicas de familiares para un periodo específico
    """
    df_personas = datos_filtrados["Personas"]
    estudiantes_periodo = df_personas[df_personas["periodo"] == periodo][
        "identificacion"
    ].drop_duplicates()

    df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())
    if df_universo.empty:
        return pd.Series(dtype=object)

    universo_periodo = df_universo[
        df_universo["identificacion"].isin(estudiantes_periodo)
    ]

    cedulas_padres = universo_periodo["ced_padre"][
        universo_periodo["ced_padre"] != "0"
    ].drop_duplicates()
    cedulas_madres = universo_periodo["ced_madre"][
        universo_periodo["ced_madre"] != "0"
    ].drop_duplicates()

    return pd.concat([cedulas_padres, cedulas_madres]).drop_duplicates()


def crear_pie_chart_tipos_deuda(datos_filtrados, periodo, grupo_seleccionado):
    """
    Crea un pie chart con la proporción de tipos de deuda (top 10)

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        periodo: Periodo específico
        grupo_seleccionado: Tipo de grupo (G, A, E)

    Returns:
        plotly.graph_objects.Figure: Gráfico pie chart
    """
    # Obtener datos de deudas
    df_deudas = datos_filtrados.get("Deudas", pd.DataFrame())
    if df_deudas.empty:
        return None

    # Filtrar por año 2025 y mes 7
    df_deudas_mes7 = df_deudas[(df_deudas["anio"] == 2025) & (df_deudas["mes"] == 7)]

    if df_deudas_mes7.empty:
        return None

    # Filtrar por grupo de interés
    if grupo_seleccionado == "G":  # Graduados
        df_personas = datos_filtrados["Personas"]
        graduados_periodo = df_personas[df_personas["periodo"] == periodo][
            "identificacion"
        ].drop_duplicates()

        df_deudas_filtrado = df_deudas_mes7[
            df_deudas_mes7["identificacion"].isin(graduados_periodo)
        ]

    else:  # Afluentes o Enrollment
        familiares_unicos = obtener_familiares_periodo(datos_filtrados, periodo)
        if len(familiares_unicos) == 0:
            return None

        df_deudas_filtrado = df_deudas_mes7[
            df_deudas_mes7["identificacion"].isin(familiares_unicos)
        ]

    if df_deudas_filtrado.empty:
        return None

    # Agrupar por tipo (nombre descriptivo), sumar valores y obtener top 10
    # Si el campo 'tipo' está vacío, usar 'cod_tipo'
    df_deudas_filtrado["tipo_final"] = df_deudas_filtrado.apply(
        lambda row: (
            row["tipo"]
            if (pd.notna(row["tipo"]) and row["tipo"].strip() != "")
            else row["cod_tipo"]
        ),
        axis=1,
    )

    top_deudas = df_deudas_filtrado.groupby("tipo_final")["valor"].sum().reset_index()
    top_deudas = top_deudas.sort_values("valor", ascending=False).head(10)

    if top_deudas.empty:
        return None

    # Crear pie chart
    fig = px.pie(
        top_deudas,
        values="valor",
        names="tipo_final",
        title=f"Top 10 Tipos de Deuda - {grupo_seleccionado} {periodo} (Julio 2025)",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    # Personalizar el gráfico
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>"
        + "Valor: $%{value:,.0f}<br>"
        + "Porcentaje: %{percent}<br>"
        + "<extra></extra>",
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
    )

    return fig


def crear_stacked_bar_calificacion(datos_filtrados, periodo, grupo_seleccionado):
    """
    Crea un gráfico de barras apiladas con deuda por calificación crediticia

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        periodo: Periodo específico
        grupo_seleccionado: Tipo de grupo (G, A, E)

    Returns:
        plotly.graph_objects.Figure: Gráfico de barras apiladas
    """
    # Obtener datos de deudas
    df_deudas = datos_filtrados.get("Deudas", pd.DataFrame())
    if df_deudas.empty:
        return None

    # Filtrar por año 2025 y mes 7
    df_deudas_mes7 = df_deudas[(df_deudas["anio"] == 2025) & (df_deudas["mes"] == 7)]

    if df_deudas_mes7.empty:
        return None

    # Filtrar por grupo de interés
    if grupo_seleccionado == "G":  # Graduados
        df_personas = datos_filtrados["Personas"]
        graduados_periodo = df_personas[df_personas["periodo"] == periodo][
            "identificacion"
        ].drop_duplicates()

        df_deudas_filtrado = df_deudas_mes7[
            df_deudas_mes7["identificacion"].isin(graduados_periodo)
        ]

    else:  # Afluentes o Enrollment
        familiares_unicos = obtener_familiares_periodo(datos_filtrados, periodo)
        if len(familiares_unicos) == 0:
            return None

        df_deudas_filtrado = df_deudas_mes7[
            df_deudas_mes7["identificacion"].isin(familiares_unicos)
        ]

    if df_deudas_filtrado.empty:
        return None

    # Usar calificacion (descripción) o cod_calificacion si está vacío
    df_deudas_filtrado["calificacion_final"] = df_deudas_filtrado.apply(
        lambda row: (
            row["calificacion"]
            if (pd.notna(row["calificacion"]) and row["calificacion"].strip() != "")
            else row["cod_calificacion"]
        ),
        axis=1,
    )

    # Agrupar por calificación y sumar valores
    calificacion_deudas = (
        df_deudas_filtrado.groupby("calificacion_final")["valor"].sum().reset_index()
    )

    if calificacion_deudas.empty:
        return None

    # Ordenar por calificación (A1, A2, A3, AL, B1, B2, C1, C2, D, E)
    orden_calificacion = ["A1", "A2", "A3", "AL", "B1", "B2", "C1", "C2", "D", "E"]
    calificacion_deudas["orden"] = calificacion_deudas["calificacion_final"].apply(
        lambda x: orden_calificacion.index(x) if x in orden_calificacion else 999
    )
    calificacion_deudas = calificacion_deudas.sort_values("orden")

    # Definir colores por nivel de riesgo según clasificación oficial
    colores = {
        # Riesgo Normal (Verde)
        "A1": "#006400",
        "A2": "#228B22",
        "A3": "#32CD32",
        # Sin calificación (Gris)
        "AL": "#808080",
        # Riesgo Potencial (Amarillo/Naranja)
        "B1": "#FFD700",
        "B2": "#FF8C00",
        # Deficientes (Rojo)
        "C1": "#FF6347",
        "C2": "#DC143C",
        # Dudoso recaudo (Rojo oscuro)
        "D": "#8B0000",
        # Pérdidas (Negro)
        "E": "#000000",
    }

    # Crear gráfico de barras apiladas horizontal usando plotly express
    # Preparar los datos para plotly express
    calificacion_deudas["categoria"] = "Deuda Total"

    # Crear el gráfico
    fig = px.bar(
        calificacion_deudas,
        x="valor",
        y="categoria",
        color="calificacion_final",
        orientation="h",
        title=f"Deuda por Calificación Crediticia - {grupo_seleccionado} {periodo} (Julio 2025)",
        color_discrete_map=colores,
        text="valor",
    )

    # Personalizar el gráfico
    fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="inside",
        textfont=dict(color="white", size=10),
    )

    # Actualizar hover manualmente para cada trace
    for i, trace in enumerate(fig.data):
        calificacion = calificacion_deudas.iloc[i]["calificacion_final"]
        fig.data[i].update(
            hovertemplate=f"<b>{calificacion}</b><br>Valor: $%{{x:,.0f}}<extra></extra>"
        )

    fig.update_layout(
        xaxis_title="Valor de Deuda (USD)",
        yaxis_title="",
        height=400,  # Aumentar altura para dar más espacio
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Mover leyenda más abajo para evitar choques
            xanchor="center",
            x=0.5,
            font=dict(size=9),
            title="Calificación",
        ),
        xaxis=dict(tickformat="$,.0f"),
        margin=dict(l=80, r=50, t=100, b=120),  # Más margen superior e inferior
        plot_bgcolor="white",
        title=dict(x=0.5, xanchor="center", font=dict(size=14)),  # Centrar título
    )

    return fig


def crear_bar_chart_top_deudas(top_deudas, periodo, grupo_seleccionado):
    """
    Función legacy - ya no se usa
    """
    pass


# Configuración de la página
st.set_page_config(page_title="Ingresos y Deudas", page_icon="💰", layout="wide")

# Título principal
st.title("💰 Ingresos y Deudas")

# Cargar datos
df_vulnerabilidad = cargar_datos_vulnerabilidad()

# Inicializar filtros
inicializar_filtros(df_vulnerabilidad)

# Mostrar filtros en la página principal
mostrar_filtros_en_pagina()

# Aplicar filtros
datos_filtrados = aplicar_filtros(df_vulnerabilidad)

# Obtener el grupo seleccionado
grupo_seleccionado = st.session_state.grupo_interes

# Obtener periodos únicos del grupo seleccionado, ordenados alfabéticamente
df_personas_filtrado = datos_filtrados["Personas"]
periodos = sorted(df_personas_filtrado["periodo"].dropna().unique().tolist())

# Lógica condicional según el grupo seleccionado
if grupo_seleccionado in ["G", "A"]:  # Graduados o Afluentes
    st.subheader("📊 Análisis por Periodos")

    # Si hay al menos 2 periodos, mostrar en columnas
    if len(periodos) >= 2:
        # Dividir en dos columnas para mostrar los periodos
        col1, col_divider, col2 = st.columns([1, 0.05, 1])

        with col1:
            st.write(f"### {periodos[0]}")

            # Mostrar pie chart directo
            fig_pie = crear_pie_chart_tipos_deuda(
                datos_filtrados, periodos[0], grupo_seleccionado
            )
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)

                # Mostrar gráfico de barras apiladas por calificación
                fig_stack = crear_stacked_bar_calificacion(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                if fig_stack:
                    st.plotly_chart(fig_stack, use_container_width=True)
            else:
                st.info("No hay datos de deudas disponibles para este periodo")

        with col_divider:
            st.markdown(
                """
            <div style="
                border-left: 2px solid #e0e0e0;
                height: 300px;
                margin: 20px 0;
            "></div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.write(f"### {periodos[1]}")

            # Mostrar pie chart directo
            fig_pie = crear_pie_chart_tipos_deuda(
                datos_filtrados, periodos[1], grupo_seleccionado
            )
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)

                # Mostrar gráfico de barras apiladas por calificación
                fig_stack = crear_stacked_bar_calificacion(
                    datos_filtrados, periodos[1], grupo_seleccionado
                )
                if fig_stack:
                    st.plotly_chart(fig_stack, use_container_width=True)
            else:
                st.info("No hay datos de deudas disponibles para este periodo")

    else:
        # Si hay un solo periodo o ninguno
        if periodos:
            st.write(f"### {periodos[0]}")

            # Mostrar pie chart directo
            fig_pie = crear_pie_chart_tipos_deuda(
                datos_filtrados, periodos[0], grupo_seleccionado
            )
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)

                # Mostrar gráfico de barras apiladas por calificación
                fig_stack = crear_stacked_bar_calificacion(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                if fig_stack:
                    st.plotly_chart(fig_stack, use_container_width=True)
            else:
                st.info("No hay datos de deudas disponibles para este periodo")
        else:
            st.write("No hay periodos disponibles para este grupo")

else:  # Enrollment (E)
    st.subheader("📊 Análisis General - Enrollment")

    # Mostrar periodo también para Enrollment
    if periodos:
        st.write(f"### {periodos[0]}")

        # Mostrar pie chart directo
        fig_pie = crear_pie_chart_tipos_deuda(
            datos_filtrados, periodos[0], grupo_seleccionado
        )
        if fig_pie:
            st.plotly_chart(fig_pie, use_container_width=True)

            # Mostrar gráfico de barras apiladas por calificación
            fig_stack = crear_stacked_bar_calificacion(
                datos_filtrados, periodos[0], grupo_seleccionado
            )
            if fig_stack:
                st.plotly_chart(fig_stack, use_container_width=True)
        else:
            st.info("No hay datos de deudas disponibles para este periodo")
    else:
        st.write("No hay periodos disponibles para este grupo")
