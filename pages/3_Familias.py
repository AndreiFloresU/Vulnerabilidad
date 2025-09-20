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


def obtener_datos_familias(datos_filtrados, periodo, grupo_seleccionado):
    """
    Obtiene los datos de familias combinando información de estudiantes, padres y madres

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        periodo: Periodo específico
        grupo_seleccionado: Tipo de grupo (A o E)

    Returns:
        pd.DataFrame: Datos de familias con información completa
    """
    # Obtener estudiantes del periodo
    df_personas = datos_filtrados["Personas"]
    estudiantes_periodo = df_personas[df_personas["periodo"] == periodo].copy()

    # Obtener universo de familiares
    df_universo = datos_filtrados.get("Universo Familiares", pd.DataFrame())
    if df_universo.empty:
        return pd.DataFrame()

    # Unir estudiantes con información de familiares
    familias = estudiantes_periodo.merge(df_universo, on="identificacion", how="left")

    # Filtrar solo familias que tienen al menos un padre o madre válido
    familias = familias[(familias["ced_padre"] != "0") | (familias["ced_madre"] != "0")]

    return familias


def obtener_datos_empleabilidad_familia(datos_filtrados, familias_df):
    """
    Obtiene información de empleabilidad (desempleo) de las familias

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        familias_df: DataFrame con datos de familias

    Returns:
        pd.DataFrame: Familias con información de empleabilidad
    """
    # Obtener datos de ingresos para determinar empleabilidad
    df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())

    # Inicializar columnas de empleabilidad
    familias_df["padre_desempleado"] = True  # Inicialmente asumir desempleado
    familias_df["madre_desempleado"] = True  # Inicialmente asumir desempleado
    familias_df["familia_con_desempleo"] = False

    if not df_ingresos.empty:
        # Filtrar por año 2025 mes 6
        ingresos_mes6 = df_ingresos[
            (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
        ]

        if not ingresos_mes6.empty and "salario" in ingresos_mes6.columns:
            # Obtener lista de personas que SÍ tienen ingresos (están empleados)
            personas_empleadas = ingresos_mes6[ingresos_mes6["salario"] > 0][
                "identificacion"
            ].unique()

            # Verificar empleo por familia
            for idx, familia in familias_df.iterrows():
                # Verificar si el padre está empleado
                if familia["ced_padre"] != "0":
                    familias_df.loc[idx, "padre_desempleado"] = (
                        familia["ced_padre"] not in personas_empleadas
                    )
                else:
                    familias_df.loc[idx, "padre_desempleado"] = (
                        False  # No aplica si no existe padre
                    )

                # Verificar si la madre está empleada
                if familia["ced_madre"] != "0":
                    familias_df.loc[idx, "madre_desempleado"] = (
                        familia["ced_madre"] not in personas_empleadas
                    )
                else:
                    familias_df.loc[idx, "madre_desempleado"] = (
                        False  # No aplica si no existe madre
                    )

                # Familia tiene desempleo si al menos un padre/madre válido está desempleado
                tiene_padre_valido = familia["ced_padre"] != "0"
                tiene_madre_valida = familia["ced_madre"] != "0"

                padre_desempleado = (
                    tiene_padre_valido and familias_df.loc[idx, "padre_desempleado"]
                )
                madre_desempleada = (
                    tiene_madre_valida and familias_df.loc[idx, "madre_desempleado"]
                )

                # La familia tiene desempleo si algún familiar válido está desempleado
                familias_df.loc[idx, "familia_con_desempleo"] = (
                    padre_desempleado or madre_desempleada
                )

    return familias_df


def obtener_datos_salario_deuda_familia(datos_filtrados, familias_df):
    """
    Combina salarios y deudas familiares (suma de padre + madre)

    Args:
        datos_filtrados: Diccionario con los datos filtrados
        familias_df: DataFrame con datos de familias

    Returns:
        pd.DataFrame: Familias con salario total y deuda total familiar
    """
    # Obtener datos de salarios e ingresos
    df_ingresos = datos_filtrados.get("Ingresos", pd.DataFrame())
    df_deudas = datos_filtrados.get("Deudas", pd.DataFrame())

    result_familias = familias_df.copy()
    result_familias["salario_familiar"] = 0
    result_familias["deuda_familiar"] = 0

    if not df_ingresos.empty:
        # Filtrar por año 2025 mes 6
        ingresos_mes6 = df_ingresos[
            (df_ingresos["anio"] == 2025) & (df_ingresos["mes"] == 6)
        ]

        # La columna correcta para ingresos es 'salario'
        if "salario" in ingresos_mes6.columns and not ingresos_mes6.empty:
            # Sumar salarios por familia
            for idx, familia in result_familias.iterrows():
                salario_total = 0

                # Salario del padre
                if familia["ced_padre"] != "0":
                    salario_padre = ingresos_mes6[
                        ingresos_mes6["identificacion"] == familia["ced_padre"]
                    ]["salario"].sum()
                    salario_total += salario_padre

                # Salario de la madre
                if familia["ced_madre"] != "0":
                    salario_madre = ingresos_mes6[
                        ingresos_mes6["identificacion"] == familia["ced_madre"]
                    ]["salario"].sum()
                    salario_total += salario_madre

                result_familias.loc[idx, "salario_familiar"] = salario_total

    if not df_deudas.empty:
        # Filtrar por año 2025 mes 6
        deudas_mes6 = df_deudas[(df_deudas["anio"] == 2025) & (df_deudas["mes"] == 6)]

        # La columna correcta para deudas es 'valor'
        if "valor" in deudas_mes6.columns and not deudas_mes6.empty:
            # Sumar deudas por familia
            for idx, familia in result_familias.iterrows():
                deuda_total = 0

                # Deuda del padre
                if familia["ced_padre"] != "0":
                    deuda_padre = deudas_mes6[
                        deudas_mes6["identificacion"] == familia["ced_padre"]
                    ]["valor"].sum()
                    deuda_total += deuda_padre

                # Deuda de la madre
                if familia["ced_madre"] != "0":
                    deuda_madre = deudas_mes6[
                        deudas_mes6["identificacion"] == familia["ced_madre"]
                    ]["valor"].sum()
                    deuda_total += deuda_madre

                result_familias.loc[idx, "deuda_familiar"] = deuda_total

    return result_familias


def crear_treemap_desempleo(familias_df, periodo, grupo_seleccionado):
    """
    Crea un treemap mostrando % de familias con al menos un padre/madre desempleado
    """
    if familias_df.empty:
        return None

    # Calcular estadísticas de desempleo
    total_familias = len(familias_df)
    familias_con_desempleo = (
        familias_df["familia_con_desempleo"].sum()
        if "familia_con_desempleo" in familias_df.columns
        else 0
    )
    familias_sin_desempleo = total_familias - familias_con_desempleo

    # Preparar datos para treemap
    data_treemap = pd.DataFrame(
        {
            "categoria": ["Con Desempleo", "Sin Desempleo"],
            "cantidad": [familias_con_desempleo, familias_sin_desempleo],
            "porcentaje": [
                (
                    (familias_con_desempleo / total_familias * 100)
                    if total_familias > 0
                    else 0
                ),
                (
                    (familias_sin_desempleo / total_familias * 100)
                    if total_familias > 0
                    else 0
                ),
            ],
        }
    )

    # Crear treemap
    fig = px.treemap(
        data_treemap,
        path=["categoria"],
        values="cantidad",
        title=f"Distribución de Desempleo Familiar - {grupo_seleccionado} {periodo}",
        color="porcentaje",
        color_continuous_scale="RdYlGn_r",
        hover_data={"porcentaje": ":.1f"},
    )

    fig.update_layout(height=400)
    return fig


def crear_donut_desempleo(familias_df, periodo, grupo_seleccionado):
    """
    Crea un donut chart mostrando % de familias con desempleo
    """
    if familias_df.empty:
        return None

    # Calcular estadísticas
    total_familias = len(familias_df)
    familias_con_desempleo = (
        familias_df["familia_con_desempleo"].sum()
        if "familia_con_desempleo" in familias_df.columns
        else 0
    )
    familias_sin_desempleo = total_familias - familias_con_desempleo

    # Preparar datos
    data_donut = pd.DataFrame(
        {
            "categoria": ["Con Desempleo", "Sin Desempleo"],
            "cantidad": [familias_con_desempleo, familias_sin_desempleo],
        }
    )

    # Crear donut chart
    fig = px.pie(
        data_donut,
        values="cantidad",
        names="categoria",
        title=f"Proporción de Familias con Desempleo - {grupo_seleccionado} {periodo}",
        hole=0.4,
        color_discrete_sequence=["#FF6B6B", "#4ECDC4"],
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Familias: %{value}<br>Porcentaje: %{percent}<extra></extra>",
    )

    fig.update_layout(height=400)
    return fig


def crear_scatter_salario_deuda(familias_df, periodo, grupo_seleccionado):
    """
    Crea un scatter plot mostrando relación entre salario familiar y deuda familiar
    """
    if familias_df.empty:
        return None

    # Filtrar familias con datos válidos
    familias_validas = familias_df[
        (familias_df["salario_familiar"] > 0) | (familias_df["deuda_familiar"] > 0)
    ].copy()

    if familias_validas.empty:
        return None

    # Crear scatter plot
    fig = px.scatter(
        familias_validas,
        x="salario_familiar",
        y="deuda_familiar",
        title=f"Relación Salario vs Deuda Familiar - {grupo_seleccionado} {periodo} (Junio 2025)",
        labels={
            "salario_familiar": "Salario Familiar (USD)",
            "deuda_familiar": "Deuda Familiar (USD)",
        },
        opacity=0.6,
        color_discrete_sequence=["#1f77b4"],
    )

    # Personalizar
    fig.update_traces(
        marker=dict(size=8),
        hovertemplate="<b>Familia</b><br>"
        + "Salario: $%{x:,.0f}<br>"
        + "Deuda: $%{y:,.0f}<extra></extra>",
    )

    fig.update_layout(
        height=500, xaxis=dict(tickformat="$,.0f"), yaxis=dict(tickformat="$,.0f")
    )

    return fig


# Configuración de la página
st.set_page_config(
    page_title="Análisis de Familias", page_icon="👨‍👩‍👧‍👦", layout="wide"
)

# Título principal
st.title("👨‍👩‍👧‍👦 Análisis de Familias")

# Cargar datos
df_vulnerabilidad = cargar_datos_vulnerabilidad()

# Inicializar filtros
inicializar_filtros(df_vulnerabilidad)

# Mostrar filtros en la página principal (sin 'Graduados')
col1, col2, col3 = st.columns(3)
with col1:
    opciones_grupo = {"E": "Enrollment", "A": "Afluentes"}
    if st.session_state.grupo_interes not in opciones_grupo:
        st.session_state.grupo_interes = "E"
    st.session_state.grupo_interes = st.selectbox(
        "Grupo de interés:",
        options=list(opciones_grupo.keys()),
        format_func=lambda x: opciones_grupo[x],
        index=list(opciones_grupo.keys()).index(st.session_state.grupo_interes),
        key="grupo_interes_familias",
    )
with col2:
    # Facultad (filtrada según el grupo de interés)
    from utils.filtros import obtener_facultades_por_grupo

    facultades_filtradas = obtener_facultades_por_grupo(st.session_state.grupo_interes)
    if st.session_state.facultad_seleccionada not in facultades_filtradas:
        st.session_state.facultad_seleccionada = "Todos"
    st.session_state.facultad_seleccionada = st.selectbox(
        "Facultad:",
        options=facultades_filtradas,
        index=facultades_filtradas.index(st.session_state.facultad_seleccionada),
        key="facultad_seleccionada_familias",
    )
with col3:
    from utils.filtros import obtener_carreras_por_grupo_y_facultad

    carreras_filtradas = obtener_carreras_por_grupo_y_facultad(
        st.session_state.grupo_interes, st.session_state.facultad_seleccionada
    )
    if st.session_state.carrera_seleccionada not in carreras_filtradas:
        st.session_state.carrera_seleccionada = "Todos"
    st.session_state.carrera_seleccionada = st.selectbox(
        "Carrera:",
        options=carreras_filtradas,
        index=carreras_filtradas.index(st.session_state.carrera_seleccionada),
        key="carrera_seleccionada_familias",
    )

# Aplicar filtros
datos_filtrados = aplicar_filtros(df_vulnerabilidad)

# Obtener el grupo seleccionado
grupo_seleccionado = st.session_state.grupo_interes

# Validar que solo se muestren Afluentes y Enrollment
if grupo_seleccionado == "G":
    st.warning(
        "⚠️ El análisis de familias solo está disponible para **Afluentes (A)** y **Enrollment (E)**"
    )
    st.info("Por favor, cambie el grupo de interés en la barra lateral.")
else:
    # Obtener periodos únicos del grupo seleccionado
    df_personas_filtrado = datos_filtrados["Personas"]
    periodos = sorted(df_personas_filtrado["periodo"].dropna().unique().tolist())

    if periodos:
        st.subheader(f"📊 Análisis Familiar - {grupo_seleccionado}")

        # Si hay al menos 2 periodos, mostrar en columnas
        if len(periodos) >= 2:
            # Dividir en dos columnas para mostrar los periodos
            col1, col_divider, col2 = st.columns([1, 0.05, 1])

            with col1:
                st.write(f"### {periodos[0]}")

                # Obtener datos de familias para el primer periodo
                familias_df = obtener_datos_familias(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )

                if not familias_df.empty:
                    # Procesar datos de empleabilidad
                    familias_df = obtener_datos_empleabilidad_familia(
                        datos_filtrados, familias_df
                    )

                    # Procesar datos de salario y deuda
                    familias_df = obtener_datos_salario_deuda_familia(
                        datos_filtrados, familias_df
                    )

                    # Mostrar métricas para primer periodo
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        tarjeta_simple(
                            "Total Familias", f"{len(familias_df)}", COLORES["azul"]
                        )
                    with subcol2:
                        salario_promedio = familias_df["salario_familiar"].mean()
                        tarjeta_simple(
                            "Salario Promedio",
                            f"${salario_promedio:,.0f}",
                            COLORES["morado"],
                        )

                    # Gráficos de desempleo
                    fig_tree = crear_treemap_desempleo(
                        familias_df, periodos[0], grupo_seleccionado
                    )
                    if fig_tree:
                        st.plotly_chart(fig_tree, use_container_width=True)

                    fig_donut = crear_donut_desempleo(
                        familias_df, periodos[0], grupo_seleccionado
                    )
                    if fig_donut:
                        st.plotly_chart(fig_donut, use_container_width=True)

                else:
                    st.info("No hay datos de familias disponibles para este periodo")

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

                # Obtener datos de familias para el segundo periodo
                familias_df2 = obtener_datos_familias(
                    datos_filtrados, periodos[1], grupo_seleccionado
                )

                if not familias_df2.empty:
                    # Procesar datos de empleabilidad
                    familias_df2 = obtener_datos_empleabilidad_familia(
                        datos_filtrados, familias_df2
                    )

                    # Procesar datos de salario y deuda
                    familias_df2 = obtener_datos_salario_deuda_familia(
                        datos_filtrados, familias_df2
                    )

                    # Mostrar métricas para segundo periodo
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        tarjeta_simple(
                            "Total Familias", f"{len(familias_df2)}", COLORES["azul"]
                        )
                    with subcol2:
                        salario_promedio2 = familias_df2["salario_familiar"].mean()
                        tarjeta_simple(
                            "Salario Promedio",
                            f"${salario_promedio2:,.0f}",
                            COLORES["morado"],
                        )

                    # Gráficos de desempleo
                    fig_tree2 = crear_treemap_desempleo(
                        familias_df2, periodos[1], grupo_seleccionado
                    )
                    if fig_tree2:
                        st.plotly_chart(fig_tree2, use_container_width=True)

                    fig_donut2 = crear_donut_desempleo(
                        familias_df2, periodos[1], grupo_seleccionado
                    )
                    if fig_donut2:
                        st.plotly_chart(fig_donut2, use_container_width=True)

                else:
                    st.info("No hay datos de familias disponibles para este periodo")

            # Scatter plot de salario vs deuda para ambos periodos
            st.markdown("---")
            st.subheader("📈 Relación Salario vs Deuda Familiar")

            # Crear dos columnas para los scatter plots
            scatter_col1, scatter_col2 = st.columns(2)

            with scatter_col1:
                # Scatter plot para el primer periodo
                familias_df_scatter1 = obtener_datos_familias(
                    datos_filtrados, periodos[0], grupo_seleccionado
                )
                if not familias_df_scatter1.empty:
                    familias_df_scatter1 = obtener_datos_salario_deuda_familia(
                        datos_filtrados, familias_df_scatter1
                    )
                    fig_scatter1 = crear_scatter_salario_deuda(
                        familias_df_scatter1, periodos[0], grupo_seleccionado
                    )
                    if fig_scatter1:
                        st.plotly_chart(fig_scatter1, use_container_width=True)
                    else:
                        st.info(
                            "No hay datos suficientes para mostrar la relación salario-deuda"
                        )

            with scatter_col2:
                # Scatter plot para el segundo periodo
                familias_df_scatter2 = obtener_datos_familias(
                    datos_filtrados, periodos[1], grupo_seleccionado
                )
                if not familias_df_scatter2.empty:
                    familias_df_scatter2 = obtener_datos_salario_deuda_familia(
                        datos_filtrados, familias_df_scatter2
                    )
                    fig_scatter2 = crear_scatter_salario_deuda(
                        familias_df_scatter2, periodos[1], grupo_seleccionado
                    )
                    if fig_scatter2:
                        st.plotly_chart(fig_scatter2, use_container_width=True)
                    else:
                        st.info(
                            "No hay datos suficientes para mostrar la relación salario-deuda"
                        )

        else:
            # Si hay un solo periodo o ninguno
            periodo = periodos[0]
            st.write(f"### {periodo}")

            # Obtener datos de familias
            familias_df = obtener_datos_familias(
                datos_filtrados, periodo, grupo_seleccionado
            )

            if not familias_df.empty:
                # Procesar datos de empleabilidad
                familias_df = obtener_datos_empleabilidad_familia(
                    datos_filtrados, familias_df
                )

                # Procesar datos de salario y deuda
                familias_df = obtener_datos_salario_deuda_familia(
                    datos_filtrados, familias_df
                )

                # Mostrar métricas generales
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    tarjeta_simple(
                        "Total Familias", f"{len(familias_df)}", COLORES["azul"]
                    )

                with col2:
                    familias_con_padre = len(
                        familias_df[familias_df["ced_padre"] != "0"]
                    )
                    tarjeta_simple(
                        "Con Padre", f"{familias_con_padre}", COLORES["verde"]
                    )

                with col3:
                    familias_con_madre = len(
                        familias_df[familias_df["ced_madre"] != "0"]
                    )
                    tarjeta_simple(
                        "Con Madre", f"{familias_con_madre}", COLORES["naranja"]
                    )

                with col4:
                    salario_promedio = familias_df["salario_familiar"].mean()
                    tarjeta_simple(
                        "Salario Promedio",
                        f"${salario_promedio:,.0f}",
                        COLORES["morado"],
                    )

                st.markdown("---")

                # Gráficos de desempleo
                col_tree, col_donut = st.columns(2)

                with col_tree:
                    fig_tree = crear_treemap_desempleo(
                        familias_df, periodo, grupo_seleccionado
                    )
                    if fig_tree:
                        st.plotly_chart(fig_tree, use_container_width=True)

                with col_donut:
                    fig_donut = crear_donut_desempleo(
                        familias_df, periodo, grupo_seleccionado
                    )
                    if fig_donut:
                        st.plotly_chart(fig_donut, use_container_width=True)

                # Scatter plot de salario vs deuda
                st.markdown("---")
                fig_scatter = crear_scatter_salario_deuda(
                    familias_df, periodo, grupo_seleccionado
                )
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.info(
                        "No hay datos suficientes para mostrar la relación salario-deuda"
                    )

            else:
                st.info("No hay datos de familias disponibles para este periodo")
    else:
        st.write("No hay periodos disponibles para este grupo")
