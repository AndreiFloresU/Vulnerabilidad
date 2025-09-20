import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.carga_datos import cargar_datos_vulnerabilidad
from utils.filtros import (
    inicializar_filtros,
    aplicar_filtros,
    obtener_facultades_por_grupo,
    obtener_carreras_por_grupo_y_facultad,
)
from utils.tarjetas import tarjeta_simple, COLORES
import numpy as np


def calcular_vulnerabilidad_estudiantes(datos_filtrados, periodo):
    """
    Calcula la vulnerabilidad de estudiantes basándose en criterios específicos

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        periodo: Periodo específico

    Returns:
        pd.DataFrame: Estudiantes con flag de vulnerabilidad
    """
    # Obtener estudiantes del periodo
    df_personas = datos_filtrados["Personas"]
    estudiantes_periodo = df_personas[df_personas["periodo"] == periodo].copy()

    if estudiantes_periodo.empty:
        return pd.DataFrame()

    # Inicializar flags de vulnerabilidad
    estudiantes_periodo["vulnerable"] = False
    estudiantes_periodo["en_riesgo"] = False
    estudiantes_periodo["motivos_vulnerabilidad"] = ""
    estudiantes_periodo["contador_riesgos"] = 0

    # Obtener datos de familias
    df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())
    df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
    df_deudas = datos_filtrados.get("Deudas", pd.DataFrame())

    if not df_universo.empty:
        # Unir con información familiar
        estudiantes_con_familia = estudiantes_periodo.merge(
            df_universo, on="identificacion", how="left"
        )

        # Criterio 1: Familia sin empleo (padre y madre sin ingresos)
        if not df_ingresos.empty:
            # Filtrar ingresos de junio 2025
            ingresos_mes6 = df_ingresos[
                (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
            ]

            if not ingresos_mes6.empty and "salario" in ingresos_mes6.columns:
                # Personas con ingresos
                personas_con_ingresos = set(
                    ingresos_mes6[ingresos_mes6["salario"] > 0]["identificacion"]
                )

                for idx, estudiante in estudiantes_con_familia.iterrows():
                    motivos = []
                    contador_condiciones = 0

                    # Verificar si familia no tiene ingresos
                    padre_sin_empleo = (
                        estudiante["ced_padre"] != "0"
                        and estudiante["ced_padre"] not in personas_con_ingresos
                    )
                    madre_sin_empleo = (
                        estudiante["ced_madre"] != "0"
                        and estudiante["ced_madre"] not in personas_con_ingresos
                    )

                    # Si ambos padres (si existen) están sin empleo
                    tiene_padre = estudiante["ced_padre"] != "0"
                    tiene_madre = estudiante["ced_madre"] != "0"

                    if tiene_padre and tiene_madre:
                        # Ambos padres existen
                        if padre_sin_empleo and madre_sin_empleo:
                            motivos.append("Familia sin empleo")
                            contador_condiciones += 1
                    elif tiene_padre and not tiene_madre:
                        # Solo padre existe
                        if padre_sin_empleo:
                            motivos.append("Familia sin empleo")
                            contador_condiciones += 1
                    elif not tiene_padre and tiene_madre:
                        # Solo madre existe
                        if madre_sin_empleo:
                            motivos.append("Familia sin empleo")
                            contador_condiciones += 1

                    # Criterio 2: Deudas familiares en calificación D o E
                    if not df_deudas.empty:
                        deudas_mes6 = df_deudas[
                            (df_deudas["anio"] == 2025) & (df_deudas["mes"] == 6)
                        ]

                        if not deudas_mes6.empty:
                            # Deudas de la familia
                            cedulas_familia = []
                            if estudiante["ced_padre"] != "0":
                                cedulas_familia.append(estudiante["ced_padre"])
                            if estudiante["ced_madre"] != "0":
                                cedulas_familia.append(estudiante["ced_madre"])

                            if cedulas_familia:
                                deudas_familia = deudas_mes6[
                                    deudas_mes6["identificacion"].isin(cedulas_familia)
                                ]

                                # Verificar si hay deudas en calificación D o E
                                if "cod_calificacion" in deudas_familia.columns:
                                    calificaciones_riesgo = deudas_familia[
                                        "cod_calificacion"
                                    ].isin(["D", "E"])
                                    if calificaciones_riesgo.any():
                                        motivos.append("Deuda familiar crítica (D/E)")
                                        contador_condiciones += 1

                    # Actualizar vulnerabilidad según el contador de condiciones
                    if contador_condiciones > 0:
                        idx_original = estudiantes_periodo.index[
                            estudiantes_periodo["identificacion"]
                            == estudiante["identificacion"]
                        ][0]
                        
                        # Asignar categoría según el número de condiciones
                        if contador_condiciones >= 2:
                            # Rojo: Alta vulnerabilidad (2 o más condiciones)
                            estudiantes_periodo.loc[idx_original, "vulnerable"] = True
                            estudiantes_periodo.loc[idx_original, "en_riesgo"] = False
                        else:
                            # Amarillo: En situación de riesgo (1 condición)
                            estudiantes_periodo.loc[idx_original, "vulnerable"] = False
                            estudiantes_periodo.loc[idx_original, "en_riesgo"] = True
                        
                        estudiantes_periodo.loc[idx_original, "contador_riesgos"] = contador_condiciones
                        estudiantes_periodo.loc[
                            idx_original, "motivos_vulnerabilidad"
                        ] = "; ".join(motivos)

    return estudiantes_periodo


def crear_barras_facultades_vulnerables(estudiantes_vulnerables, periodo):
    """
    Crea gráfico de barras con top 5 facultades con más estudiantes vulnerables

    Args:
        estudiantes_vulnerables: DataFrame con estudiantes y flag de vulnerabilidad
        periodo: Periodo específico

    Returns:
        plotly.graph_objects.Figure: Gráfico de barras
    """
    if estudiantes_vulnerables.empty:
        return None

    # Calcular estadísticas por facultad
    stats_facultad = (
        estudiantes_vulnerables.groupby("facultad")
        .agg(
            {
                "identificacion": "count",  # Total estudiantes
                "vulnerable": "sum",  # Estudiantes vulnerables
                "en_riesgo": "sum",  # Estudiantes en riesgo
            }
        )
        .reset_index()
    )

    stats_facultad.columns = ["facultad", "total_estudiantes", "vulnerables", "en_riesgo"]

    # Agregar total de estudiantes con algún tipo de riesgo
    stats_facultad["total_con_riesgo"] = stats_facultad["vulnerables"] + stats_facultad["en_riesgo"]

    # Filtrar solo facultades con estudiantes en algún tipo de riesgo
    stats_facultad = stats_facultad[stats_facultad["total_con_riesgo"] > 0]

    if stats_facultad.empty:
        return None

    # Ordenar por total con riesgo y tomar top 5
    top_facultades = stats_facultad.sort_values("total_con_riesgo", ascending=False).head(5)

    # Crear gráfico de barras
    fig = px.bar(
        top_facultades,
        x="total_con_riesgo",
        y="facultad",
        orientation="h",
        title=f"Top 5 Facultades con Más Estudiantes en Situación de Riesgo - Enrollment {periodo}",
        labels={
            "total_con_riesgo": "Número de Estudiantes en Riesgo",
            "facultad": "Facultad",
        },
        text="total_con_riesgo",
        color="vulnerables",
        color_continuous_scale="Reds",
    )

    # Personalizar
    fig.update_traces(
        textposition="inside",
        texttemplate="%{text}",
        hovertemplate="<b>%{y}</b><br>"
        + "Total en riesgo: %{x}<br>"
        + "Alta vulnerabilidad: %{marker.color}<br>"
        + "<extra></extra>",
    )

    fig.update_layout(
        height=400,
        yaxis=dict(categoryorder="total ascending"),
        xaxis=dict(title="Número de Estudiantes en Riesgo"),
        coloraxis_colorbar=dict(title="Estudiantes con Alta Vulnerabilidad"),
    )

    return fig


# Configuración de la página
st.set_page_config(page_title="Análisis de Riesgos", page_icon="⚠️", layout="wide")

# Título principal
st.title("⚠️ Análisis de Riesgos")

# Cargar datos
df_vulnerabilidad = cargar_datos_vulnerabilidad()

# Inicializar filtros
inicializar_filtros(df_vulnerabilidad)

# Mostrar filtros personalizados (solo Enrollment)
st.header("🔍 Filtros")
col1, col2, col3 = st.columns(3)

with col1:
    # Solo mostrar Enrollment
    st.session_state.grupo_interes = "E"
    st.selectbox(
        "Grupo de interés:",
        options=["E"],
        format_func=lambda x: "Enrollment",
        index=0,
        disabled=True,
        key="grupo_interes_riesgos",
    )

with col2:
    # Facultad (filtrada según enrollment)
    facultades_filtradas = obtener_facultades_por_grupo("E")
    if st.session_state.facultad_seleccionada not in facultades_filtradas:
        st.session_state.facultad_seleccionada = "Todos"
    st.session_state.facultad_seleccionada = st.selectbox(
        "Facultad:",
        options=facultades_filtradas,
        index=facultades_filtradas.index(st.session_state.facultad_seleccionada),
        key="facultad_seleccionada_riesgos",
    )

with col3:
    # Carrera (filtrada según enrollment y facultad)
    carreras_filtradas = obtener_carreras_por_grupo_y_facultad(
        "E", st.session_state.facultad_seleccionada
    )
    if st.session_state.carrera_seleccionada not in carreras_filtradas:
        st.session_state.carrera_seleccionada = "Todos"
    st.session_state.carrera_seleccionada = st.selectbox(
        "Carrera:",
        options=carreras_filtradas,
        index=carreras_filtradas.index(st.session_state.carrera_seleccionada),
        key="carrera_seleccionada_riesgos",
    )

# Aplicar filtros
datos_filtrados = aplicar_filtros(df_vulnerabilidad)

# Obtener periodos únicos de enrollment
df_personas_filtrado = datos_filtrados["Personas"]
periodos = sorted(df_personas_filtrado["periodo"].dropna().unique().tolist())

if periodos:
    st.subheader("📊 Análisis de Vulnerabilidad - Enrollment")

    # Análisis para el primer periodo disponible
    periodo = periodos[0]
    st.write(f"### {periodo}")

    # Calcular vulnerabilidad
    estudiantes_vulnerables = calcular_vulnerabilidad_estudiantes(
        datos_filtrados, periodo
    )

    if not estudiantes_vulnerables.empty:
        # Mostrar métricas generales
        total_estudiantes = len(estudiantes_vulnerables)
        estudiantes_alta_vulnerabilidad = estudiantes_vulnerables["vulnerable"].sum()
        estudiantes_en_situacion_riesgo = estudiantes_vulnerables["en_riesgo"].sum()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            tarjeta_simple("Total Estudiantes", f"{total_estudiantes}", COLORES["azul"])

        with col2:
            tarjeta_simple(
                "Alta Vulnerabilidad", f"{estudiantes_alta_vulnerabilidad}", COLORES["rojo"]
            )

        with col3:
            tarjeta_simple(
                "En Situación de Riesgo",
                f"{estudiantes_en_situacion_riesgo}",
                COLORES["naranja"],
            )

        with col4:
            estudiantes_seguros = total_estudiantes - estudiantes_alta_vulnerabilidad - estudiantes_en_situacion_riesgo
            tarjeta_simple(
                "Sin Riesgo Identificado", f"{estudiantes_seguros}", COLORES["verde"]
            )

        st.markdown("---")

        # Gráfico de barras de facultades vulnerables
        fig_barras = crear_barras_facultades_vulnerables(
            estudiantes_vulnerables, periodo
        )
        if fig_barras:
            st.plotly_chart(fig_barras, use_container_width=True)
        else:
            st.info(
                "No se encontraron estudiantes en situación vulnerable para mostrar el análisis por facultades"
            )

        # Mostrar detalles de vulnerabilidad si hay estudiantes en riesgo
        if estudiantes_alta_vulnerabilidad > 0 or estudiantes_en_situacion_riesgo > 0:
            st.markdown("---")
            st.subheader("📋 Detalle de Estudiantes Vulnerables")

            # Mostrar estudiantes con alta vulnerabilidad
            estudiantes_riesgo_alto = estudiantes_vulnerables[
                estudiantes_vulnerables["vulnerable"] == True
            ][
                [
                    "identificacion",
                    "facultad",
                    "carrera_homologada",
                    "motivos_vulnerabilidad",
                ]
            ]

            # Mostrar estudiantes en situación de riesgo
            estudiantes_riesgo_medio = estudiantes_vulnerables[
                estudiantes_vulnerables["en_riesgo"] == True
            ][
                [
                    "identificacion",
                    "facultad",
                    "carrera_homologada",
                    "motivos_vulnerabilidad",
                ]
            ]

            if not estudiantes_riesgo_alto.empty:
                st.write("#### 🔴 Estudiantes con Alta Vulnerabilidad (2 o más condiciones)")
                estudiantes_riesgo_alto.columns = [
                    "Identificación",
                    "Facultad",
                    "Carrera",
                    "Motivos de Vulnerabilidad",
                ]
                st.dataframe(estudiantes_riesgo_alto, use_container_width=True)

            if not estudiantes_riesgo_medio.empty:
                st.write("#### 🟡 Estudiantes en Situación de Riesgo (1 condición)")
                estudiantes_riesgo_medio.columns = [
                    "Identificación",
                    "Facultad",
                    "Carrera",
                    "Motivos de Vulnerabilidad",
                ]
                st.dataframe(estudiantes_riesgo_medio, use_container_width=True)

    else:
        st.info("No hay datos de estudiantes disponibles para este periodo")

else:
    st.write("No hay periodos disponibles para Enrollment")
