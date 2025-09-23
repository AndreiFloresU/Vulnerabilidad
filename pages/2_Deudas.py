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
        ].copy()

    else:  # Afluentes o Enrollment
        familiares_unicos = obtener_familiares_periodo(datos_filtrados, periodo)
        if len(familiares_unicos) == 0:
            return None

        df_deudas_filtrado = df_deudas_mes7[
            df_deudas_mes7["identificacion"].isin(familiares_unicos)
        ].copy()

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


def crear_bar_chart_calificacion_descriptiva(
    datos_filtrados, periodo, grupo_seleccionado
):
    """
    Gráfico de barras horizontal con calificaciones crediticias,
    agrupadas en 3 categorías (Riesgo estable, Riesgo moderado, Alto Riesgo)
    según el código en 'cod_calificacion'.
    """
    df_deudas = datos_filtrados.get("Deudas", pd.DataFrame())
    if df_deudas.empty:
        return None

    # Julio 2025
    df_deudas_mes7 = df_deudas[(df_deudas["anio"] == 2025) & (df_deudas["mes"] == 7)]
    if df_deudas_mes7.empty:
        return None

    # Filtrar universo según grupo
    if grupo_seleccionado == "G":
        df_personas = datos_filtrados["Personas"]
        graduados_periodo = df_personas.loc[
            df_personas["periodo"] == periodo, "identificacion"
        ].drop_duplicates()
        df_deudas_filtrado = df_deudas_mes7[
            df_deudas_mes7["identificacion"].isin(graduados_periodo)
        ].copy()
    else:
        familiares_unicos = obtener_familiares_periodo(datos_filtrados, periodo)
        if len(familiares_unicos) == 0:
            return None
        df_deudas_filtrado = df_deudas_mes7[
            df_deudas_mes7["identificacion"].isin(familiares_unicos)
        ].copy()

    if df_deudas_filtrado.empty:
        return None

    # 🔑 Normalizar y usar SIEMPRE el código 'cod_calificacion' para mapear
    df_deudas_filtrado["cod_cal_norm"] = (
        df_deudas_filtrado["cod_calificacion"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"": np.nan, "NAN": np.nan})
    )
    # opcional: no contar vacías
    df_deudas_filtrado = df_deudas_filtrado.dropna(subset=["cod_cal_norm"])

    # Mapeo código → descripción (3 grupos)
    mapa_desc = {
        "A1": "Riesgo estable",
        "A2": "Riesgo moderado",
        "A3": "Riesgo moderado",
        "AL": "Riesgo moderado",
        "B1": "Riesgo moderado",
        "B2": "Riesgo moderado",
        "C1": "Riesgo moderado",
        "C2": "Riesgo moderado",
        "D": "Alto Riesgo",
        "E": "Alto Riesgo",
    }

    df_deudas_filtrado["calificacion_desc"] = (
        df_deudas_filtrado["cod_cal_norm"].map(mapa_desc).fillna("Desconocido")
    )

    # Agrupar por las 3 categorías (más "Desconocido" si hubiera códigos raros)
    df_grouped = df_deudas_filtrado.groupby("calificacion_desc", as_index=False).agg(
        conteo_deudas=("valor", "count"), valor_total=("valor", "sum")
    )

    # Orden por severidad
    orden = {
        "Riesgo estable": 1,
        "Riesgo moderado": 2,
        "Alto Riesgo": 3,
        "Desconocido": 99,
    }
    df_grouped["orden"] = df_grouped["calificacion_desc"].map(orden)
    df_grouped = df_grouped.sort_values("orden")

    if df_grouped.empty:
        return None

    # Colores por categoría
    colores = {
        "Riesgo estable": "#2ecc71",  # verde
        "Riesgo moderado": "#f1c40f",  # amarillo
        "Alto Riesgo": "#e74c3c",  # rojo
        "Desconocido": "#95a5a6",  # gris
    }

    fig = px.bar(
        df_grouped,
        x="conteo_deudas",
        y="calificacion_desc",
        orientation="h",
        title=f"Deudas por Calificación Crediticia - {grupo_seleccionado} {periodo} (Julio 2025)",
        text="conteo_deudas",
        color="calificacion_desc",
        color_discrete_map=colores,
        hover_data={"valor_total": ":$,.0f", "conteo_deudas": True},
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="inside",
        hovertemplate="<b>%{y}</b><br>"
        "Número de deudas: %{x:,}<br>"
        "Valor total: %{customdata[0]:$,.0f}<extra></extra>",
    )
    fig.update_layout(
        xaxis_title="Número de Deudas",
        yaxis_title="Calificación Crediticia",
        height=350,
        showlegend=False,
        margin=dict(l=80, r=50, t=80, b=60),
        plot_bgcolor="white",
        title=dict(x=0.5, xanchor="center", font=dict(size=14)),
    )
    return fig


# Configuración de la página
st.set_page_config(page_title="Deudas", page_icon="💰", layout="wide")

# Título principal
st.title("💰 Deudas")

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

                # Mostrar gráfico de barras top 5
                fig_top5 = crear_bar_chart_calificacion_descriptiva(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                if fig_top5:
                    st.plotly_chart(fig_top5, use_container_width=True)

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

                # Mostrar gráfico de barras top 5
                fig_top5 = crear_bar_chart_calificacion_descriptiva(
                    datos_filtrados, periodos[1], grupo_seleccionado
                )
                if fig_top5:
                    st.plotly_chart(fig_top5, use_container_width=True)

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

                # Mostrar gráfico de barras top 5
                fig_top5 = crear_bar_chart_calificacion_descriptiva(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                if fig_top5:
                    st.plotly_chart(fig_top5, use_container_width=True)

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

            # Mostrar gráfico de barras top 5
            fig_top5 = crear_bar_chart_calificacion_descriptiva(
                datos_filtrados, periodos[0], grupo_seleccionado
            )
            if fig_top5:
                st.plotly_chart(fig_top5, use_container_width=True)

        else:
            st.info("No hay datos de deudas disponibles para este periodo")
    else:
        st.write("No hay periodos disponibles para este grupo")
